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
After completing the learning, the agent can go to the two trees in the environment and attack it to collect the woods. However, it stops. Therefore, the agent cannot collect more wood.

![Treechop-v0 agent video](https://github.com/kimbring2/MineRL/blob/master/monitor/ezgif.com-video-to-gif.gif)

To collect more wood in one environment, I need to solve this problem.

![Treechop-v0 traning loss graph](https://github.com/kimbring2/MineRL/blob/master/image/14-47-20.png)

Loss graph shows that there is no ploblem in traning process. However, it looks like it need more training step.

# Reinforcment Learning
Changing the number and combination of learning behaviors to follow or changing the number of networks did not improve performance any more. So we examined how to use reinforcement learning additionally.

## Network Structure
The reinforcement learning network structure is the same as the supervised learning network structure, but the loss for learning uses the reward value that the agent earns.

## Result
For imitation learning, we first called the weights of the learned networks and used them in reinforcement learning. When experimenting with these procedures, we were unable to confirm an improvement in performance that should be unique.

# What is problem of two approach
At this point, we decided that changes in network type and structure could no longer improve performance. So, other approaches were devised, but in the case of the TreeChop task, the necessary actions were considered first.

## Solution
The first thing to consider was that in order to collect items like wood, you had to take attack action for a certain amount of time. If you train agents to take one action per frame without considering the duration of these specific actions, you can easily see that the number has increased significantly.

''' python
if (action2_index == 0):
  action['camera'][0] = 0; action['camera'][1] = -1; action['forward'] = 0; action['jump'] = 0; 
  action['attack'] = 1
elif (action2_index == 1):
  action['camera'][0] = 0; action['camera'][1] = 1; action['forward'] = 0; action['jump'] = 0;
  action['attack'] = 1
elif (action2_index == 2):
  action['camera'][0] = 1; action['camera'][1] = 0; action['forward'] = 0; action['jump'] = 0;  
  action['attack'] = 1
elif (action2_index == 3):
  action['camera'][0] = -1; action['camera'][1] = 0; action['forward'] = 0; action['jump'] = 0; 
  action['attack'] = 1
elif (action2_index == 4):
  action['camera'][0] = 0; action['camera'][1] = 0; action['forward'] = 0; action['jump'] = 0; 
  action['attack'] = 1
elif (action2_index == 5):
  action['camera'][0] = 0; action['camera'][1] = 0; action['forward'] = 1; action['jump'] = 1; 
  action['attack'] = 0
'''
