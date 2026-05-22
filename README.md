<p align="center">

  <h3 align="center">InfoGeo: Information-Theoretic Object-Centric Learning for Cross-View Generalizable UAV Geo-Localization</h3>

</p>

<h5 align="center">
  If you like our project, please give us a star ⭐️ for the continuous updates.
</h5>

<p align="center">
  By <a href="https://hrt00.github.io/hyzhang.github.io/" target="_blank">Hongyang Zhang<sup>1,*</sup></a>,&nbsp;
  Maonan Wang<sup>2,3,*</sup>,&nbsp;
  Ziyao Wang<sup>1</sup>,&nbsp;
  Hongrui Yin<sup>1</sup>,&nbsp;
  Man On Pun<sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup>CUHK(SZ); 
  <sup>2</sup>CUHK; 
  <sup>3</sup>Shanghai AI Lab
</p>

<p align="center">
  <sup>*</sup>Equal contribution. <sup>†</sup>Corresponding author.
</p>

## <a id="news"></a> 🔥 News
- [May 01, 2026]: InfoGeo is accepted by ICML'26 🎉

## 📝 Overview

<div align="center">
  <img src="assets/overview.png" width="800"/>
  <br>
  <em>Overview Pipeline of InfoGeo & Details on  Cross-view Visual Concept Reasoner </em>
</div>

Cross-view geo-localization (CVGL) is fundamental for precise localization and navigation in GPS-denied environments, aiming to match ground or UAV imagery with satellite views. Existing approaches often rely on global feature alignment, but they suffer from substantial domain shifts induced by varying regional textures and weather conditions. This issue becomes even more pronounced in UAV-based scenarios, where the broader perspective inevitably introduces dense, fine-grained objects, creating significant visual clutter. To address this, we draw inspiration from Object-Centric Learning (OCL) and propose InfoGeo, an information-theoretic framework designed to enhance robustness and generalization. InfoGeo reformulates the optimization as an information bottleneck process with two core objectives: (i) maximizing view-invariant information by aligning the object-centric structural relations across views, and (ii) minimizing view-specific noisy signals through cross-view knowledge constraints. Extensive evaluations across diverse benchmarks and challenging scenarios demonstrate that InfoGeo significantly outperforms state-of-the-art methods.

---
