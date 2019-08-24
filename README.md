# MineRL Imitation&Reinforcement Learning
This github code is for solving a MineRLTreechop-v0, MineRLNavigate-v0 environment of MineRL. I use a Convolution Neural Network for traning a agent. 

# How to use it
First, you need to install dependent package by using a requirement.txt file. 
Second, you need to modify a path for model and summary of tensorflow in MineRL_IL.ipynb file.
Third, you should check code operate properly by running IPython Notebook.

# Imitation Learning
The data that people played directly on the game is provided, so I tried to solve the problem by using Imitation Learning first.

## Network Structure
The network structure for Imitation Learning seems to be very simple CNN extracts features on the game screen, processes them through flatten and FC, and finally outputs the probability for each action.

![Structure image](https://github.com/kimbring2/MineRL/blob/master/image/03-17-22.png)

## Result
After finishing a training, agent can go to a two tree in environment and attack it for collecting wood. But, it stops after then. Thus, agent can collect only 2 reward.

![Treechop-v0 agent video](https://github.com/kimbring2/MineRL/blob/master/monitor/ezgif.com-video-to-gif.gif)

I need to solve this problem for collection more wood in one environment.

![Treechop-v0 traning loss graph](https://github.com/kimbring2/MineRL/blob/master/image/14-47-20.png)

Loss graph shows that there is no ploblem in traning process. However, it looks like it need more epoch.

# Reinforcment Learning
Changing the number and combination of learning behaviors to follow or changing the number of networks did not improve performance any more. So we examined how to use reinforcement learning additionally.

## Network Structure
The reinforcement learning network structure is the same as the supervised learning network structure, but the loss for learning uses the reward value that the agent earns.

## Result
For imitation learning, we first called the weights of the learned networks and used them in reinforcement learning. When experimenting with these procedures, we were unable to confirm an improvement in performance that should be unique.

# What is real problem of two approach
At this point, we decided that changes in network type and structure could no longer improve performance. So, other approaches were devised, but in the case of the TreeChop task, the necessary actions were considered first.
