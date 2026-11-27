import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Optional

import numpy as np

import math


EPS = 1e-8

def get_w_regularization(w1, w2):
    """
    Competition regularization on router weights.
    w1, w2: Router weights with shape [B, K, 1].
    Encourages polarized weights and variation across slots.
    """
    # Combine router weights from both views.
    w = torch.cat([w1, w2], dim=0) # [2B, K, 1]

    # Penalize intermediate weights.
    # w * (1 - w) peaks at 0.5 and reaches zero at 0 and 1.
    # Encourage weights toward either endpoint.
    loss_polar = torch.mean(w * (1 - w))

    # Encourage diversity across slots.
    # Discourage identical weights across slots.
    # Maximize the variance of w across K slots.
    loss_div = - torch.var(w, dim=1).mean()

    return 0.1 * loss_polar + 0.1 * loss_div

def batched_cosine_sim(A: torch.Tensor, B: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    # A, B: (B, K, D)
    A_n = F.normalize(A, p=2, dim=-1, eps=eps)
    B_n = F.normalize(B, p=2, dim=-1, eps=eps)
    return torch.einsum('bik,bjk->bij', A_n, B_n)  # (B, K, K)


def attention_overlap(A_att: torch.Tensor, B_att: torch.Tensor) -> torch.Tensor:
    # A_att, B_att: (B, H*W, K) attention maps of slots over spatial tokens
    # returns affinity by computing JSD or dot-overlap between slot attention vectors
    # Here we use simple dot product of normalized attention vectors
    # Output: (B, K, K) where entry (i,j) is overlap between slot_i in A and slot_j in B
    # We assume A_att and B_att are L2-normalized along spatial dim
    A_n = F.normalize(A_att, p=2, dim=1, eps=EPS)  # (B,HW,K)
    B_n = F.normalize(B_att, p=2, dim=1, eps=EPS)
    # compute KxK overlap per batch: einsum over spatial dim
    # result shape: (B, K, K)
    return torch.einsum('bsk,btk->bst', A_n, B_n)


# -------------------------
# Graph construction
# -------------------------
def build_slot_graph(slots: torch.Tensor,
                     mode: str = 'cosine',
                     att_maps: Optional[torch.Tensor] = None,
                     sigma: float = 0.1) -> torch.Tensor:
    """
    Construct a symmetric affinity matrix G for each batch.
    slots: (B, K, D)
    mode: 'cosine' | 'rbf' | 'attention'
      - 'cosine': pairwise cosine similarity
      - 'rbf': RBF (gaussian) kernel on euclidean distance
      - 'attention': uses att_maps (B, HW, K) for cross-slot overlap
    returns G: (B, K, K) symmetric, non-negative
    """
    B, K, D = slots.shape
    if mode == 'cosine':
        G = batched_cosine_sim(slots, slots)  # [-1,1]
        # shift to non-negative: (x+1)/2
        G = (G + 1.0) / 2.0
        G = torch.clamp(G, min=0.0)
    elif mode == 'rbf':
        # compute pairwise squared distances
        # slots: (B,K,D)
        x_norm2 = (slots ** 2).sum(dim=-1, keepdim=True)  # (B, K, 1)
        # pairwise dist squared: (B, K, K)
        dist2 = x_norm2 + x_norm2.transpose(1, 2) - 2.0 * torch.einsum('bik,bjk->bij', slots, slots)
        G = torch.exp(-dist2 / (2.0 * sigma * sigma))
    elif mode == 'attention':
        if att_maps is None:
            raise ValueError("att_maps required for mode='attention'")
        # att_maps: (B, HW, K) -> compute overlap
        # convert to (B, K, HW) for convenience
        A = att_maps.permute(0, 2, 1)
        # normalize
        A_n = F.normalize(A, p=2, dim=-1, eps=EPS)
        G = torch.einsum('bip,bjp->bij', A_n, A_n)
        G = torch.clamp(G, min=0.0)
    else:
        raise ValueError(f"unknown mode {mode}")
    # optionally set diagonal to zero or keep as self-affinity (we'll keep diag as 1)
    # set diagonal to 1 for stable Laplacian
    eye = torch.eye(K, device=G.device, dtype=G.dtype).unsqueeze(0)
    G = G * (1.0 - eye) + eye  # ensure diagonal ones
    return G


# -------------------------
# Laplacian & spectral embedding
# -------------------------
def compute_graph_laplacian(G: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """
    G: (B, K, K) affinity (non-negative, symmetric)
    returns L: (B, K, K) Laplacian (symmetric)
    """
    deg = G.sum(dim=-1)  # (B, K)
    B, K = deg.shape
    Dinv_sqrt = torch.diag_embed(torch.pow(deg + EPS, -0.5))  # (B, K, K)
    if normalized:
        # normalized Laplacian: L = I - D^{-1/2} G D^{-1/2}
        I = torch.eye(K, device=G.device, dtype=G.dtype).unsqueeze(0)
        DGD = torch.einsum('bij,bjk,bkl->bil', Dinv_sqrt, G, Dinv_sqrt)  # (B,K,K)
        L = I - DGD
    else:
        D = torch.diag_embed(deg)
        L = D - G
    # ensure symmetry
    L = 0.5 * (L + L.transpose(1, 2))
    return L


class _StableEigenvectors(torch.autograd.Function):
    """Differentiate eigenvectors of a real symmetric matrix across resolved eigenvalue gaps."""

    @staticmethod
    def forward(ctx, matrix):
        values, vectors = torch.linalg.eigh(matrix)
        ctx.save_for_backward(values, vectors)
        return vectors

    @staticmethod
    def backward(ctx, grad_vectors):
        values, vectors = ctx.saved_tensors
        with torch.autocast(vectors.device.type, enabled=False):
            gaps = values.unsqueeze(-2) - values.unsqueeze(-1)
            tolerance = (torch.finfo(values.dtype).eps * values.shape[-1]
                         * values.abs().amax(dim=-1, keepdim=True).unsqueeze(-1))
            # Rotations inside an unresolved eigenspace have no unique derivative.
            inverse_gaps = torch.where(gaps.abs() > tolerance, gaps.reciprocal(), 0.0)
            adjoint = (vectors.mT @ grad_vectors.to(vectors.dtype)) * inverse_gaps
            symmetric_adjoint = (adjoint + adjoint.mT) * 0.5
            return vectors @ symmetric_adjoint @ vectors.mT


def spectral_embedding(L: torch.Tensor, r: int = 4, stable_eigenvectors: bool = False) -> torch.Tensor:
    """
    Compute r smallest non-trivial eigenvectors of Laplacian per batch.
    L: (B, K, K) symmetric positive semidef.
    Returns U: (B, K, r) spectral embedding.
    Note: smallest eigenvector is often constant (trivial), so we skip the first eigenvector (index 0).
    We'll take eigenvectors 1..r (if available), else take first r.
    """
    B, K, _ = L.shape
    L = L.float()
    # use torch.linalg.eigh (returns ascending eigenvalues)
    # To save memory, compute batchwise; torch.linalg.eigh supports batch
    if stable_eigenvectors:
        eigvecs = _StableEigenvectors.apply(L)
    else:
        _, eigvecs = torch.linalg.eigh(L)  # eigvals: (B,K) ascending
    # skip first trivial eigenvector if K > r
    # choose indices start_idx..start_idx+r-1
    if K > r:
        start = 1  # skip eigenvector 0
        end = start + r
        if end <= K:
            U = eigvecs[:, :, start:end]  # (B, K, r)
        else:
            # fallback: take last r eigenvectors if not enough after skipping
            U = eigvecs[:, :, -r:]
    else:
        # K <= r, return all eigenvectors (may include trivial)
        U = eigvecs  # (B, K, K)
        if U.shape[-1] != r:
            # pad with zeros
            pad = r - U.shape[-1]
            U = F.pad(U, (0, pad), 'constant', 0.0)
    return U.to(L.dtype)  # (B, K, r)


# -------------------------
# Spectral alignment loss (orthogonal Procrustes)
# -------------------------
def orthogonal_procrustes_loss(Ua: torch.Tensor, Ub: torch.Tensor) -> torch.Tensor:
    """
    Ua, Ub: (B, K, r) spectral embeddings
    minimizes min_Q || Ua - Ub Q ||_F^2 over orthogonal Q.
    Closed-form: Q* = V U^T where SVD of (Ub^T Ua) = U S V^T.
    Loss = ||Ua||^2 + ||Ub||^2 - 2 * trace(S)
    We'll compute per-batch and average.
    """
    B, K, r = Ua.shape
    # center columns? spectral embeddings may be sign/scale ambiguous; we can optionally normalize columns
    # compute BtA = Ub^T Ua -> shape (B, r, r)
    BtA = torch.einsum('bkr,bkp->brp', Ub, Ua)  # (B, r, r)
    BtA = BtA.float()
    # SVD per batch
    # torch.linalg.svd supports batch and returns U, S, Vh (Vh is V^H)
    U_svd, S_svd, Vh_svd = torch.linalg.svd(BtA)  # U: (B,r,r), S: (B,r), Vh: (B,r,r)
    # trace(S) = sum S
    traceS = S_svd.sum(dim=1)  # (B,)
    # norms
    Ua_norm2 = (Ua ** 2).sum(dim=(1, 2))
    Ub_norm2 = (Ub ** 2).sum(dim=(1, 2))
    loss_per_batch = Ua_norm2 + Ub_norm2 - 2.0 * traceS
    loss = loss_per_batch.mean()
    return loss


# -------------------------
# SPOT loss integrator
# -------------------------
def reg_InfoNCE(Sa: torch.Tensor,
              Sb: torch.Tensor,
              att_maps_a: Optional[torch.Tensor] = None,
              att_maps_b: Optional[torch.Tensor] = None,
              graph_mode: str = 'cosine',
              r: int = 4,
              lambda_struct: float = 1.0,
              stable_eigenvectors: bool = False) -> torch.Tensor:
    """
    Compute SPOT losses for a batch:
     - spectral structural alignment loss L_struct
    Sa, Sb: (B, K, D) slot embeddings per view
    att_maps_* optional: (B, HW, K) for 'attention' graph_mode
    """
    # 1) build graphs
    Ga = build_slot_graph(Sa, mode=graph_mode, att_maps=att_maps_a)
    Gb = build_slot_graph(Sb, mode=graph_mode, att_maps=att_maps_b)

    # 2) Laplacian
    La = compute_graph_laplacian(Ga, normalized=True)
    Lb = compute_graph_laplacian(Gb, normalized=True)

    # 3) spectral embedding
    Ua = spectral_embedding(La, r=r, stable_eigenvectors=stable_eigenvectors)  # (B, K, r)
    Ub = spectral_embedding(Lb, r=r, stable_eigenvectors=stable_eigenvectors)

    # 4) spectral alignment loss
    L_struct = orthogonal_procrustes_loss(Ua, Ub)

    loss = lambda_struct * L_struct

    return loss  

def slot_infoNCE(slots1, slots2, temperature=0.07):
    """
    slots1: B × K × D
    slots2: B × K × D
    Returns scalar loss
    """

    # 1) Normalize along feature dimension
    slots1 = F.normalize(slots1, dim=-1)
    slots2 = F.normalize(slots2, dim=-1)

    B, K, D = slots1.shape

    # 2) Flatten slots: B*K × D
    slots1_flat = slots1.reshape(B*K, D)
    slots2_flat = slots2.reshape(B*K, D)

    # 3) Compute similarity matrix: (B*K) × (B*K)
    logits = torch.matmul(slots1_flat, slots2_flat.t()) / temperature

    # 4) Positive pairs are on the diagonal
    labels = torch.arange(B*K, device=logits.device)

    # 5) Cross-entropy as InfoNCE
    loss = F.cross_entropy(logits, labels)

    return loss

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        logits_per_image1 = logit_scale * image_features1 @ image_features1.T
        
        logits_per_image1_self = logit_scale * image_features1 @ image_features1.T
        logits_per_image2_self = logit_scale * image_features2 @ image_features2.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss_ssl = (
            self.loss_function(logits_per_image1_self, labels) + 
            self.loss_function(logits_per_image2_self, labels)
        ) / 2
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2 + loss_ssl

        # loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss

class WeightedInfoNCE(nn.Module):
    def __init__(self, loss_function, label_smoothing, k=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.loss_function = loss_function
        self.label_smoothing = label_smoothing
        self.device = device
        self.k = k

    def loss(self, similarity_matrix, eps_all):
        n = similarity_matrix.shape[0]
        total_loss = 0.0
        for i in range(n):
            eps = eps_all[i]
            total_loss += (1 - eps) * (-1. * similarity_matrix[i, i] + torch.logsumexp(similarity_matrix[i, :], dim=0))
            total_loss += eps * (-1. / n * similarity_matrix[i, :].sum() + torch.logsumexp(similarity_matrix[i, :], dim=0))
        total_loss /= n
        return total_loss

    def forward(self, image_features1, image_features2, logit_scale, positive_weights=None):
        # Normalize the image features
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        # Compute similarity logits
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        # Apply positive weights if provided
        if positive_weights is not None:
            eps = 1. - 1. / (1 + torch.exp(-self.k * positive_weights))
        else:
            eps = [self.label_smoothing for _ in range(image_features1.shape[0])]
        
        logits_per_image2 = logits_per_image1.T
        
        # Generate labels
        # labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)

        loss1 = self.loss(logits_per_image1, eps)
        loss2 = self.loss(logits_per_image2, eps)
        # # Compute loss
        logits_per_image1_self = logit_scale * image_features1 @ image_features1.T
        logits_per_image2_self = logit_scale * image_features2 @ image_features2.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss_ssl = (
            self.loss_function(logits_per_image1_self, labels) + 
            self.loss_function(logits_per_image2_self, labels)
        ) / 2
        loss = (loss1 + loss2) / 2 + loss_ssl

        return loss


def relational_distillation(student_features, teacher_features, metric='euclidean', temperature=1.0, normalize=True):
    """
    Compute Relational Distillation loss between student and teacher features.

    Args:
        student_features: Tensor of shape (batch_size, dim_s)
        teacher_features: Tensor of shape (batch_size, dim_t)
        metric: 'euclidean' or 'cosine' to measure relational structure.
        temperature: scaling factor for distances or similarities (if applicable).
        normalize: whether to normalize distances by teacher's mean distance.

    Returns:
        loss: scalar Tensor representing the relational distillation loss.
    """
    assert student_features.dim() == 2 and teacher_features.dim() == 2, "Features should be 2D tensors"
    N = student_features.size(0)

    if metric == 'euclidean':
        # Normalize features to unit vectors
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        # Pairwise differences for teacher and student (shape: N x N x D)
        teacher_diffs = teacher_features.unsqueeze(1) - teacher_features.unsqueeze(0)
        student_diffs = student_features.unsqueeze(1) - student_features.unsqueeze(0)
        # Pairwise Euclidean distances (shape: N x N)
        teacher_dists = torch.norm(teacher_diffs, dim=2, p=2)
        student_dists = torch.norm(student_diffs, dim=2, p=2)

        if normalize:
            # Normalize by mean teacher distance to focus on relative structure
            eps = 1e-8
            teacher_rel = teacher_dists / teacher_dists.mean() + eps
            student_rel = student_dists / student_dists.mean() + eps
        else:
            teacher_rel = teacher_dists
            student_rel = student_dists

        # Apply temperature scaling if needed
        if temperature != 1.0:
            teacher_rel = teacher_rel / temperature
            student_rel = student_rel / temperature

        # Compute MSE loss between relational structures
        loss = F.mse_loss(student_rel, teacher_rel)

    elif metric == 'cosine':
        # Normalize features to unit vectors
        student_norm = F.normalize(student_features, p=2, dim=1)
        teacher_norm = F.normalize(teacher_features, p=2, dim=1)
        # Compute pairwise cosine similarity matrices (shape: N x N)
        teacher_sim = torch.matmul(teacher_norm, teacher_norm.t())
        student_sim = torch.matmul(student_norm, student_norm.t())

        # Apply temperature scaling to similarities (logits) if needed
        if temperature != 1.0:
            teacher_sim = teacher_sim / temperature
            student_sim = student_sim / temperature

        # Compute MSE loss between relational structures
        loss = F.mse_loss(student_sim, teacher_sim)

    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")

    return loss
