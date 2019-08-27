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
The agent obtains information on items currently possessed, including screen information on the screen during game play. In addition, actions such as camera rotation, advancement, attack, item creation, item drop, and item equipment can be performed.

```
pov = obs['pov'].astype(np.float32) / 255.0 - 0.5
inventory = obs['inventory']
        
coal = inventory['coal']
cobblestone = inventory['cobblestone']
crafting_table = inventory['crafting_table']
dirt = inventory['dirt']
furnace = inventory['furnace']
iron_axe = inventory['iron_axe']
iron_ingot = inventory['iron_ingot']
iron_ore = inventory['iron_ore']
iron_pickaxe = inventory['iron_pickaxe']
log = inventory['log']
planks = inventory['planks']
stick = inventory['stick']
stone = inventory['stone']
stone_axe = inventory['stone_axe']
stone_pickaxe = inventory['stone_pickaxe']
torch = inventory['torch']
wooden_axe = inventory['wooden_axe']
wooden_pickaxe = inventory['wooden_pickaxe']
```

Moreover, it is necessary to select the action of the agent by the value output from the network. In the first output, an action related to an item is selected, and in the second output, an attack, jump, and camera rotation action are selected.

```
if (action1_index == 0):
  action['place'] = 1; action['craft'] = 0;
  action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
elif (action1_index == 1):
  action['place'] = 2; action['craft'] = 0;
  action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
elif (action1_index == 2):
  action['place'] = 3; action['craft'] = 0;
  action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
elif (action1_index == 3):
  action['place'] = 4; action['craft'] = 0;
  action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
elif (action1_index == 4):
  action['place'] = 5; action['craft'] = 0;
  action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
elif (action1_index == 5):
  action['place'] = 0; action['craft'] = 1; 
  action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
elif (action1_index == 6):
  action['place'] = 0; action['craft'] = 2;
  action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
elif (action1_index == 7):
  action['place'] = 0; action['craft'] = 3;
  action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
elif (action1_index == 8):
  action['place'] = 0; action['craft'] = 4;
  action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
elif (action1_index == 9):
  action['place'] = 0; action['craft'] = 0;
  action['nearbyCraft'] = 1; action['nearbySmelt'] = 0
elif (action1_index == 10):
  action['place'] = 0; action['craft'] = 0;
  action['nearbyCraft'] = 2; action['nearbySmelt'] = 0
elif (action1_index == 11):
  action['place'] = 0; action['craft'] = 0;
  action['nearbyCraft'] = 3; action['nearbySmelt'] = 0
elif (action1_index == 12):
  action['place'] = 0; action['craft'] = 0;
  action['nearbyCraft'] = 4; action['nearbySmelt'] = 0
elif (action1_index == 13):
  action['place'] = 0; action['craft'] = 0;
  action['nearbyCraft'] = 5; action['nearbySmelt'] = 0
elif (action1_index == 14):
  action['place'] = 0; action['craft'] = 0;
  action['nearbyCraft'] = 6; action['nearbySmelt'] = 0
elif (action1_index == 15):
  action['place'] = 0; action['craft'] = 0;
  action['nearbyCraft'] = 7; action['nearbySmelt'] = 0
elif (action1_index == 16):
  action['place'] = 0; action['craft'] = 0;
  action['nearbyCraft'] = 0; action['nearbySmelt'] = 1
elif (action1_index == 17):
  action['place'] = 0; action['craft'] = 0;
  action['nearbyCraft'] = 0; action['nearbySmelt'] = 2
elif (action1_index == 18):
  action['place'] = 0; action['craft'] = 0;
  action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
            
          
if (action2_index == 0):
  action['camera'][0] = 0; action['camera'][1] = -0.5; action['forward'] = 0; action['jump'] = 0; 
  action['attack'] = 0
elif (action2_index == 1):
  action['camera'][0] = 0; action['camera'][1] = 0.5; action['forward'] = 0; action['jump'] = 0;
  action['attack'] = 0
elif (action2_index == 2):
  action['camera'][0] = 0.5; action['camera'][1] = 0; action['forward'] = 0; action['jump'] = 0;  
  action['attack'] = 0
elif (action2_index == 3):
  action['camera'][0] = -0.5; action['camera'][1] = 0; action['forward'] = 0; action['jump'] = 0; 
  action['attack'] = 0
elif (action2_index == 4):
  action['camera'][0] = 0; action['camera'][1] = 0; action['forward'] = 0; action['jump'] = 0; 
  action['attack'] = 1
elif (action2_index == 5):
  action['camera'][0] = 0; action['camera'][1] = 0; action['forward'] = 1; action['jump'] = 1; 
  action['attack'] = 0
```

## Imitation Learning Result
After completing the learning, the agent can go to the two trees in the environment and attack it to collect the woods. However, it stops. Therefore, the agent cannot collect more wood.

![Treechop-v0 agent video](https://github.com/kimbring2/MineRL/blob/master/monitor/ezgif.com-video-to-gif.gif)

Loss graph shows that there is no ploblem in traning process.

<img src="image/14-47-20.png" width="600">

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
<iframe width="560" height="315"
src="https://www.youtube.com/watch?v=CAzVF_lgOK4&t=6s"
frameborder="0"; encrypted-media" allowfullscreen></iframe>

# How to use a code
First, you need to install dependent package by using a requirement.txt file. 
Second, you need to modify a path for model and summary of tensorflow in MineRL_IL.ipynb file.
Third, you should check code operate properly by running IPython Notebook.
