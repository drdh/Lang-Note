*讨论在整个ray系列中出现的算法*

## Tune

### 1. Tune Trial Schedulers

#### 1.1. Popilation Based Training (PBT)

[DeepMind Blog-PBT](https://deepmind.com/blog/population-based-training-neural-networks/)

[Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846)

本论文首先介绍当前的hyperparameter调整的方式分为如下

- Paralle Search

  并行多个进程，每个进程不同的hyperparameter. 最后将结果比较，选择最好的那个

  - Grid Search
  - Random Search

- Sequential Optimisation

  每次运行一个，然后根据这个的结果，重新选择hyperparameter, 循环往复，直到比较好的结果

用流程图来表示就是：

![1544539375126](Algorithms-in-Ray/1544539375126.png)

而PBT就是，并行一系列hyperparameter不同的模型，然后在中间将那些结果不好的模型的parameter和hyperparameter换成较好的那个(exploit), 并且加上一些随机的噪声(explore).

![1544540487814](Algorithms-in-Ray/1544540487814.png)

#### 1.2. Asynchronous HyperBand



#### 1.3. HyperBand



#### 1.4. Median Stopping Rule





### 2. Tune Search Algorithms

#### 2.1. Variant Generation (Grid Search/Random Search)



#### 2.2. HyperOpt Search (Tree-structured Parzen Estimators)



## RLlib

### 1. RLlib Algorithms

#### 1.1. High-throughput architectures

##### 1.1.1. Distributed Prioritized Experience Replay (Ape-X)



##### 1.1.2. Importance Weighted Actor-Learner Architecture (IMPALA)



#### 1.2. Gradient-based

##### 1.2.1. Advantage Actor-Critic (A2C, A3C)



##### 1.2.2. Deep Deterministic Policy Gradients (DDPG, TD3)



##### 1.2.3. Deep Q Networks (DQN, Rainbow, Parametric DQN)



##### 1.2.4. Policy Gradients



##### 1.2.5. Proximal Policy Optimization (PPO)



#### 1.3. Derivative-free

##### 1.3.1. Augmented Random Search (ARS)



##### 1.3.2. Evolution Strategies

