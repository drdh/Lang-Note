*讨论在整个ray系列中出现的算法*

## Tune

注意Trial Schedulers与Search Algorithms是不同的。前者安排一系列Trials如何执行执行顺序，后者确定每次的Hyperparameter Configuration.

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

[Massively Parallel Hyperparameter Tuning](https://openreview.net/forum?id=S1Y7OOlRZ)

这个算法是下面的HyperBand异步推广的结果。它的算法为

![1544603332687](Algorithms-in-Ray/1544603332687.png)

论文中只给出了Successive Halving算法(SHA),　由于下面的HyperBand 使用了SHA的子过程，所以很容易补充成为Asynchronous HyperBand.

输入的参数为$r$ 最小资源，$\eta >0$ 表示reduction factor, $s$ 表示最小的early-stop rate.

注意到rung越大，表示这个超参的设定越有前景，则分配到的资源越大。

同时由于get_job()函数的存在，使得算法是异步的，当发现存在promotable设定的时候，就返回这个设定，并且将rung加一，如果不存在，也不用等待其他的结束，而是直接生成一个新的设定，且rung=0

#### 1.3. HyperBand

[standard version of HyperBand](https://arxiv.org/abs/1603.06560)

https://people.eecs.berkeley.edu/~kjamieson/hyperband.html

这个的实际目标是，资源B是有限的，尝试的次数n可以选择，每次尝试分配的资源是B/n, 那么如何分配资源，n可大可小。

提出的算法如下

![1544601981800](Algorithms-in-Ray/1544601981800.png)

设定一个案例如下：

![1544602008924](Algorithms-in-Ray/1544602008924.png)

将trial分成很多部分，每次trial最大资源为$R$, 分成$(s_{max}=\lfloor \log _{\eta} (R) \rfloor)+1$ 个阶段，每个阶段总资源为$B=(s_{max}+1)R$ 每个阶段又分成多个子部分，其中当尝试的个数$n_i$越大，分配的资源$r_i$越小，越会提早结束探索。每个SHA子过程都使用不同的early-stop rate.(由于现代的超参调试问题都有高维的搜索空间，并且模型有很大的训练代价，所以提前终止是很有必要的)

其中get_hyperparameter_configuration(n) 表示从超参的设定集合中采样n个独立同分布的样本。

run_then_return_val_loss(t,ri)表示超参为t, 资源为ri时的validation loss

python 代码如下

```python
# you need to write the following hooks for your custom problem
from problem import get_random_hyperparameter_configuration,run_then_return_val_loss

max_iter = 81  # maximum iterations/epochs per configuration
eta = 3 # defines downsampling rate (default=3)
logeta = lambda x: log(x)/log(eta)
s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

#### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
for s in reversed(range(s_max+1)):
    n = int(ceil(int(B/max_iter/(s+1))*eta**s)) # initial number of configurations
    r = max_iter*eta**(-s) # initial number of iterations to run configurations for

    #### Begin Finite Horizon Successive Halving with (n,r)
    T = [ get_random_hyperparameter_configuration() for i in range(n) ] 
    for i in range(s+1):
        # Run each of the n_i configs for r_i iterations and keep best n_i/eta
        n_i = n*eta**(-i)
        r_i = r*eta**(i)
        val_losses = [ run_then_return_val_loss(num_iters=r_i,hyperparameters=t) for t in T ]
        T = [ T[i] for i in argsort(val_losses)[0:int( n_i/eta )] ]
    #### End Finite Horizon Successive Halving with (n,r)
```

#### 1.4. Median Stopping Rule

[Google Vizier: A Service for Black-Box Optimization](https://ai.google/research/pubs/pub46180)

这篇文章介绍的是google研发的Black0Box Optimization系统。主要介绍了系统的组成。略。

### 2. Tune Search Algorithms

#### 2.1. Variant Generation (Grid Search/Random Search)

无需多言。

#### 2.2. HyperOpt Search (Tree-structured Parzen Estimators)

[Hyperopt Distributed Asynchronous Hyperparameter Optimization in Python](http://hyperopt.github.io/hyperopt/)

这其实是一个Python库：

> `hyperopt` is a Python library for optimizing over awkward search spaces with real-valued, discrete, and conditional dimensions.

目前实现的算法有

- Random Search
- Tree of Parzen Estimators (TPE)

使用案例

安装`pip install hyperopt`

```python
from hyperopt import hp

# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
from hyperopt import fmin, tpe, space_eval
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print(best)
print(space_eval(space,best))
```

输出为

```python
{'a': 1, 'c2': -0.08088083656564893}
('case 2', -0.08088083656564893)
```

注意到用法其实很简单，定义objective, 设定search space, 选择search aalgorithms, 设定evaluations数目。

ray里面其实是调用这个库来实现的。

## RLlib

### 1. RLlib Algorithms

#### 1.1. High-throughput architectures

##### 1.1.1. Distributed Prioritized Experience Replay (Ape-X)

[Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933)

是DQN与DDPG关于Apx-X的变体。使用一个GPU learner与多个CPU workers, 用于experience collection.

> Experience collection can scale to hundreds of CPU workers due to the distributed prioritization of experience prior to storage in replay buffers



##### 1.1.2. Importance Weighted Actor-Learner Architecture (IMPALA)

[IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)

在IMPALA中，一个中心的learner在一个很紧凑的循环里执行SGD, 同时异步地从许多actor processes里面拉取样本batches



#### 1.2. Gradient-based

##### 1.2.1. Advantage Actor-Critic (A2C, A3C)

[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)





##### 1.2.2. Deep Deterministic Policy Gradients (DDPG, TD3)

[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)

DDPG实现与DQN实现很像。



##### 1.2.3. Deep Q Networks (DQN, Rainbow, Parametric DQN)

[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)

Rainbow是作为DQN的改进



##### 1.2.4. Policy Gradients

[Policy Gradient Methods for Reinforcement Learning with Function Approximation ](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)

vabilla policy gradients.下面的PPO表现更好



##### 1.2.5. Proximal Policy Optimization (PPO)

[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)



#### 1.3. Derivative-free

##### 1.3.1. Augmented Random Search (ARS)

[Simple random search provides a competitive approach to reinforcement learning](https://arxiv.org/abs/1803.07055)





##### 1.3.2. Evolution Strategies

[Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)