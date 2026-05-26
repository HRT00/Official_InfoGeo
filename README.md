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
  Hongrui Yin<sup>1,2</sup>,&nbsp;
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
- [May 08, 2026]: The preprint version has been released in [Paper Link](https://arxiv.org/pdf/2605.07099v3) 🚩
- [May 01, 2026]: InfoGeo is accepted by ICML'26 🎉

## 📝 Overview

<div align="center">
  <img src="assets/overview.png" width="700"/>
  <br>
  <em>Overview Pipeline of InfoGeo</em>
</div>

Cross-view geo-localization (CVGL) is fundamental for precise localization and navigation in GPS-denied environments, aiming to match ground or UAV imagery with satellite views. Existing approaches often rely on global feature alignment, but they suffer from substantial domain shifts induced by varying regional textures and weather conditions. This issue becomes even more pronounced in UAV-based scenarios, where the broader perspective inevitably introduces dense, fine-grained objects, creating significant visual clutter. To address this, we draw inspiration from Object-Centric Learning (OCL) and propose InfoGeo, an information-theoretic framework designed to enhance robustness and generalization. InfoGeo reformulates the optimization as an information bottleneck process with two core objectives: (i) maximizing view-invariant information by aligning the object-centric structural relations across views, and (ii) minimizing view-specific noisy signals through cross-view knowledge constraints.

## Acknowledgement
We gratefully acknowledge the open-source community and the authors of the cross-view geo-localization. This repository is built using the [Sample4Geo](https://github.com/Skyy93/Sample4Geo) and [CV-cities](https://github.com/GaoShuang98/CVCities).

## Cite
If you find our paper and code useful in your research, please consider citing our work 📝:
```bibtex
@article{zhang2026infogeo,
  title        = {InfoGeo: Information-Theoretic Object-Centric Learning for Cross-View Generalizable UAV Geo-Localization},
  author       = {Zhang, Hongyang and Wang, Maonnan and Wang, Ziyao and Yin, Hongrui and Pun, Man On},
  year         = {2026},
  eprint       = {2605.07099},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  doi          = {10.48550/arXiv.2605.07099},
  url          = {https://arxiv.org/abs/2605.07099}
}
```

## Contact
If you have any questions about this project, please feel free to contact hongyangzhang1@link.cuhk.edu.cn.
