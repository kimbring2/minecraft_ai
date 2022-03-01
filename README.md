# Introduction
Code for playing the Minecraft using the Deep Learning. 

# Python Dependencies
1. Minerl 0.3.7
2. Tensorflow 2.4.1
3. Tensorflow_probability 0.11.0
4. ZeroMQ
5. Gym
6. OpenCV
7. Matplotlib

# Reference
1. [Hierarchical Deep Q-Network from Imperfect Demonstrations in Minecraft, A Skrynnik, 2019](https://arxiv.org/abs/1912.08664)
2. [Sample Efficient Reinforcement Learning through Learning From Demonstrations in Minecraft, C Scheller, 2020](https://arxiv.org/abs/2003.06066)

# Action, Observation of Minecraft
<img src="image/minerl_env_space.png" width="1000">

# Learning-Based Model architecture
<img src="image/learning_model.png" width="1000">

# Rule-Based Model architecture
<img src="image/rule_model.png" width="1000">

# Supervised Learning method
For Minecraft games, agent can not learn every behaviour for high level playing only using Reinforcment Learning becaue of complexity of task. In such cases, the agent must first learn through human expert data. Try to train network for MineRLTreechop-v0 first using below command.

```
$ python run_supervised_learning --workspace_path [your path]/minecraft_ai/ --data_path [your path]/minerl_data/ --gpu_use True
```

The loss should fall to almost 0 as shown in the graph below for good performance. Model is saved under folder named model of workspace path.

<img src="image/sl_reward_graph.png" width="500">

After finishing training, you can test trained model using below command.

```
$ python run_evaluation.py --workspace_path [your path]/minecraft_ai/ --model_name [trained model name] --gpu_use True
```

# Reinforcement Learning method
Because of long game play time, normal A2C method can not be used because it should use whole episode once. Therefore, off-policy A2C such as [IMPALA](https://deepmind.com/research/publications/2019/impala-scalable-distributed-deep-rl-importance-weighted-actor-learner-architectures) is needed. It can restore trajectory data from buffer for training like a DQN.

You can run the IMPALA for MineRL by below command. 

```
$ ./run_reinforcement_learning.sh [number of envs]
```

In the simplest task like a MineRLNavigateDense-v0, agent can find a direction very easily like a below graph when using only 3 action.

<img src="image/rl_reward_graph.png" width="500">

# Detailed inforamtion
Please check Medium article(https://medium.com/@dohyeongkim/deep-q-learning-from-demonstrations-dqfd-for-minecraft-tutorial-1-4b462a18de5a) for more information.
