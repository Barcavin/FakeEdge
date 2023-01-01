# FakeEdge

## About
This repository supports the following paper:
[FakeEdge: Alleviate Dataset Shift in Link Prediction](https://arxiv.org/abs/2211.15899)

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
To reproduce the results in the paper, run the following commands:

```bash
bash main.sh --data=<data> --method=<method> --fuse=<fuse>
```
`data` can be 
"cora", "citeseer", "pubmed", "Celegans", "Ecoli", "NS", "PB", "Power", "Router", "USAir", "Yeast"

`method` can be "GCN", "SAGE", "GIN", "SEAL", WalkPool", "PLNLP"

`fuse` can be "plus", "minus", "mean", "att", "original"(which means no fake edge)

## Reference

If you find this repository useful, please cite our paper:
> @inproceedings{
dong2022fakeedge,
title={FakeEdge: Alleviate Dataset Shift in Link Prediction},
author={Kaiwen Dong and Yijun Tian and Zhichun Guo and Yang Yang and Nitesh Chawla},
booktitle={The First Learning on Graphs Conference},
year={2022},
url={https://openreview.net/forum?id=QDN0jSXuvtX}
}
