<img src="image/18-28-41.png" width="350">

The code uploaded here is for MineRL Competition 2019. The purpose of the competition is to find an effective reinforcement learning algorithm using human play datasets. 

# Introduction
I started participating at the beginning of this competition, but I wasn't able to participate correctly in the Retro Contest held at OpenAI last year, so I wanted to finish it this time. What is required in the competition is a method for efficiently learning a reinforcement learning model using given data. For these reasons, we approached the most basic network by using supervised learning and learning according to a given data set.

# MineRL environment
Requesting in the competition is to resolve MineRLObtainDiamond-v0. In order to solve this, advance work such as moving to a specific place or collecting trees is necessary.

<img src="image/19-14-57.png" width="600">

The agent obtains information on items currently possessed, including screen information on the screen during game play. In addition, actions such as camera rotation, advancement, attack, item creation, item drop, and item equipment can be performed.

<img src="image/22-02-25.png" width="400"> <img src="image/22-02-49.png" width="400">

## Network Structure

The network structure is largely composed of a CNN part that receives the current state value of the agent and an FC part that outputs the next action value.

![Structure image](https://github.com/kimbring2/MineRL/blob/master/image/03-17-22.png)

# Preprosseing
The agent obtains information on items currently possessed, including screen information on the screen during game play. In addition, actions such as camera rotation, advancement, attack, item creation, item drop, and item equipment can be performed. Moreover, it is necessary to select the action of the agent by the value output from the network. In the first output, an action related to an item is selected, and in the second output, an attack, jump, and camera rotation action are selected. The details of the contents mentioned pevious can be confirmed with the uploaded code.

## Imitation Learning Result
After completing the learning, the agent can go to the two trees in the environment and attack it to collect the woods. However, it stops. Therefore, the agent cannot collect more wood.

![Treechop-v0 CNN agent video](https://github.com/kimbring2/MineRL/blob/master/monitor/ezgif.com-video-to-gif.gif)

Loss graph shows that there is no ploblem in traning process. However, in the model that used only the test results CNN and FC many times, we finally concluded that there was a limit to learning.
<img src="image/14-47-20.png" width="600">

![Treechop-v0 CNN + RNN agent video](http://i3.ytimg.com/vi/5bMTUvPmCuQ/hqdefault.jpg)(https://youtu.be/5bMTUvPmCuQ)

Because of the nature of the game, I thought that it would not be possible to fit all the information on one screen, so I introduced RNN and added it between CNN and FC for learning.
<img src="image/13-22-49.png" width="600">

# Reinforcment Learning
Changing the number and combination of actions or the number of networks did not improve performance any more. Thus, we use Reinforcement Learning additionally. The Reinforcement Learning network structure is the same as the supervised learning network structure except the loss for learning uses the reward value that the agent earns. We first called the weights of the learned networks and used them in Reinforcement Learning. When experimenting with these procedures, we were unable to confirm an improvement in performance that should be unique.

# What is problem of two approach
At this point, we decided that changes in network type and structure could no longer improve performance. So, other approaches were devised in the case of the TreeChop task.

## Solution
The first thing to consider was that in order to collect items like wood, you had to take attack action for a certain amount of time. If you train agents to take one action per frame without considering the duration of these specific actions, you can easily see that the number has increased significantly.

## Action Example
The agent selects one action using the final value output from the network.

```
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
```

Here, in order to collect trees, we added a section that repeats actions based on the output of other networks, considering that the attack action must be continued several times.
```
if (action2_index == 4):
  for q in range(0, action3_index[0]):
    obs1, reward, done, info = env.step(action)
    netr += reward

    if (done == True):
      break
    else:
      obs1, reward, done, info = env.step(action)
```

# Result
![Result1](http://i3.ytimg.com/vi/CAzVF_lgOK4/maxresdefault.jpg)](https://www.youtube.com/watch?v=CAzVF_lgOK4)
![Result2](http://i3.ytimg.com/vi/gV9UNLkQkFE/maxresdefault.jpg)](https://www.youtube.com/watch?v=gV9UNLkQkFE)

# How to use a code
First, you need to install dependent package by using a requirement.txt file. 
Second, you need to modify a path for model and summary of tensorflow in MineRL_IL.ipynb file.
Third, you should check code operate properly by running IPython Notebook.
