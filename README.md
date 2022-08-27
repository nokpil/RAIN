# Learning Heterogeneous Interaction Strengths by Trajectory Prediction with Graph Neural Network

This repository contains the official PyTorch implementation of the architecture from:

**Learning Heterogeneous Interaction Strengths by Trajectory Prediction with Graph Neural Network**
by Seungwoong Ha and Hawoong Jeong.

![image](https://user-images.githubusercontent.com/58059577/187023302-464a0805-947f-49dd-8d93-7c45a8d26a04.png)

**Abstract** : Dynamical systems with interacting agents are universal in nature, commonly modeled by a graph of relationships between their constituents. Recently, various works have been presented to tackle the problem of inferring those relationships from the system trajectories via deep neural networks, but most of the studies assume binary or discrete types of interactions for simplicity. In the real world, the interaction kernels often involve continuous interaction strengths, which cannot be accurately approximated by discrete relations. In this work, we propose the relational attentive inference network (RAIN) to infer continuously weighted interaction graphs without any ground-truth interaction strengths. Our model employs a novel pairwise attention (PA) mechanism to refine the trajectory representations and a graph transformer to extract heterogeneous interaction weights for each pair of agents. We show that our RAIN model with the PA mechanism accurately infers continuous interaction strengths for simulated physical systems in an unsupervised manner. Further, RAIN with PA successfully predicts trajectories from motion capture data with an interpretable interaction graph, demonstrating the virtue of modeling unknown dynamics with continuous weights.

## Requirements
- Python 3.6+
- Pytorch 1.0+ (written for 1.8)

## Run experiments
Here is a typical parameter settings for training RAIN for spring-ball systems with 10 balls.  Note that agent_num, dt, input_length, and output_length should be match to the dataset you have previously created.
```
torchrun RAIN.py --epochs 100000 --system spring --model-type RAIN --agent-num 10 --dt 5 --heads-dim 32 --heads-num 4 --lstm-num 1 --batch-size 128 --lr 0.0001 --input-length 50 --output-length 50 --pa T --gt F --ww F --diff T --checkpoint -1 --indicator test 
```

* model_ type : RAIN, JointLSTM, SingleLSTM, NRI
* heads_dim, heads_num : controls the number of heads and its dimension of the pairwise attention (PA) module.
* pa : determines whether to use PA mechanism or not. T : True, F : False
* gt : if this turns into T (True), the model uses ground-truth weights for its prediction. (Do not enable this option with the motion data, since it has no true edges.)
* ww : copying intrinsic frequency to every steps. Only enable this for training of Kuramoto systems.
* diff : determines whether to predict the state differences (T), or the raw state value of the next timestep (F).
* checkpoint : -1 as a default, {epoch}*{lr}_{new indicator} will refer the saved folders (at the result/run) with the same settings, load the modle checkpoints of epoch {epoch}, setting learning rate to new {lr}, and {new indicator} will be concatenated to the original indiacor. (example : 400*1e-5_E400)
* indicator : use short words for describing your current trial, and it will be attached to the file and folder names.

For dataset genearation,
* spring
```
python generate_dataset.py -n 10 -dt 0.005 -sp 0.3 -sf charge_10_t5 -st charge -is 2
```
* kuramoto
```
python generate_dataset_kuramoto.py -n 15 -dt 0.01 -sp 0.3 -sf kuramoto_15_t10_uniform -sm uniform
```
* motion 
(WIP)
