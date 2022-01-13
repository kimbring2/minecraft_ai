# Introduction
Code for playing the Minecraft using the Deep Learning. 

# Dependencies
1. minerl
2. tensorflow 2

# Network architecture
<img src="image/minecraft_network.png" width="1000">

# Supervised Learning method
I just use common the cross enthropy method for calculating loss between action of replay file and network. Loss should fall to almost 0 as shown in the graph below when the policy network is trained by the Supervised Learning manner.

<img src="image/treechop_sl_loss.png" width="500">

# Reinforcement Learning method
Because of long game play time, normal A2C method can not be due to memory problem. Therefore, [IMPALA](https://deepmind.com/research/publications/2019/impala-scalable-distributed-deep-rl-importance-weighted-actor-learner-architectures) is used which store trajectory into buffer and use it for training like a DQN. To verify alrorithm works well in Minecraft, Cartpole and Pong environment is used.

You can run the [IMPALA for Pong](https://github.com/kimbring2/minecraft_ai/blob/master/Pong_v0(IMPALA).ipynb) and check how it works.

# How to run 
First, you need to check everything about MineRL package works well in your PC. Please visit https://minerl.readthedocs.io/en/latest/.

If everything is fine, open and run the cell of MineRL-TreeChop(LSTM).ipynb file. It is simple A2C agent for TreeChop task of Minecraft. Firat, agent is trained via Supervised Learning. Next, it begins Reinforcement Learning phase from pretrained model.

# Detailed inforamtion
Please check Medium article(https://medium.com/@dohyeongkim/deep-q-learning-from-demonstrations-dqfd-for-minecraft-tutorial-1-4b462a18de5a) for more information.
