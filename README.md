# Towards-Balanced-RGB-TSDF-Fusion-for-Consistent-Semantic-Scene-Completion

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in:

**Towards Balanced RGB-TSDF Fusion for Consistent Semantic Scene Completion by 3D RGB Feature Completion and a Classwise Entropy Loss Function**

Laiyan Ding, Panwen Hu, Jie Li, Rui Huang

[PRCV 2023 (arXiv pdf)](https://arxiv.org/abs/2403.16888)

We are sorry that the code has been too old to carry on again. Thus, we only provide the core component of our project as in network.py and loss.py. network.py has a class named baseline_resnet50_init_reuse which implements of the our idea that estimate the scene semantic and geometry with two stages. loss.py is a class that implements the entropy-based loss to tackle with hard-to-distinguish classes, e.g., sofa and chair.

This code is for non-commercial use.

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{ding2023towards,
  title={Towards Balanced RGB-TSDF Fusion for Consistent Semantic Scene Completion by 3D RGB Feature Completion and a Classwise Entropy Loss Function},
  author={Ding, Laiyan and Hu, Panwen and Li, Jie and Huang, Rui},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages={128--141},
  year={2023},
  organization={Springer}
}
```

# Setup

Please refer to the codebase of [TorchSSC](https://github.com/charlesCXK/TorchSSC) for data preparation. Our code also largely borrows from [TorchSSC](https://github.com/charlesCXK/TorchSSC).
