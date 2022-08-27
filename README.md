# Learning Heterogeneous Interaction Strengths by Trajectory Prediction with Graph Neural Network

This repository contains the official PyTorch implementation of the architecture from:

**Learning Heterogeneous Interaction Strengths by Trajectory Prediction with Graph Neural Network**
by Seungwoong Ha and Hawoong Jeong.

**Abstract** : Dynamical systems with interacting agents are universal in nature, commonly modeled by a graph of relationships between their constituents. Recently, various works have been presented to tackle the problem of inferring those relationships from the system trajectories via deep neural networks, but most of the studies assume binary or discrete types of interactions for simplicity. In the real world, the interaction kernels often involve continuous interaction strengths, which cannot be accurately approximated by discrete relations. In this work, we propose the relational attentive inference network (RAIN) to infer continuously weighted interaction graphs without any ground-truth interaction strengths. Our model employs a novel pairwise attention (PA) mechanism to refine the trajectory representations and a graph transformer to extract heterogeneous interaction weights for each pair of agents. We show that our RAIN model with the PA mechanism accurately infers continuous interaction strengths for simulated physical systems in an unsupervised manner. Further, RAIN with PA successfully predicts trajectories from motion capture data with an interpretable interaction graph, demonstrating the virtue of modeling unknown dynamics with continuous weights.

## Requirements
- Python 3.6+
- Pytorch 1.0+ (written for 1.8)

## Run experiments
TBA

