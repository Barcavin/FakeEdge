# FakeEdge

## About
This repository supports the following paper:
>

FakeEdge is a model-agnostic technique to alleviate dataset shift issue in link prediction. It aligns the target link's environment by deliberately adding
or removing the target edge from the graph during training and testing stages.

This repository contains the implementation of FakeEdge in PyTorch for several existing link prediction models. The code is adapted from their original implementation. The models include:

- GAE-like models [link](https://github.com/facebookresearch/SEAL_OGB)
- SEAL [link](https://github.com/facebookresearch/SEAL_OGB)
- WalkPool [link](https://github.com/DaDaCheng/WalkPooling)
- PLNLP [link](https://github.com/zhitao-wang/PLNLP)


## Requirements

- Python 3.8
- PyTorch 1.11.0
- PyTorch\_Geometric 2.0.4

Other libraries include numpy, scipy, sklearn, tqdm, ogb etc.


## Usages

## Reference

If you find this repository useful, please cite our paper:

