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
![CNN structure image](https://github.com/kimbring2/MineRL/blob/master/image/03-17-22.png)


Minecraft does not allow you to see all game information at once. Therefore, it can be predicted that the behavior in the current frame is affected by the information in the previous frame. Therefore, we could use the RNN network additionally to account for these temporal depencies and see better performance in the Treechop task.
![RNN+CNN structure image](https://github.com/kimbring2/MineRL/blob/master/image/19-16-46.png)

# Preprosseing
The agent obtains information on items currently possessed, including screen information on the screen during game play. In addition, actions such as camera rotation, advancement, attack, item creation, item drop, and item equipment can be performed. Moreover, it is necessary to select the action of the agent by the value output from the network. In the first output, an action related to an item is selected, and in the second output, an attack, jump, and camera rotation action are selected. The details of the contents mentioned pevious can be confirmed with the uploaded code.

## Imitation Learning Result
After completing the learning, the agent can go to the two trees in the environment and attack it to collect the woods. However, it stops. Therefore, the agent cannot collect more wood.

![Treechop-v0 CNN agent video](https://github.com/kimbring2/MineRL/blob/master/monitor/ezgif.com-video-to-gif.gif)

Loss graph shows that there is no ploblem in traning process. However, in the model that used only the test results CNN and FC many times, we finally concluded that there was a limit to learning.
<img src="image/14-47-20.png" width="600">

Because of the nature of the game, I thought that it would not be possible to fit all the information on one screen, so I introduced RNN and added it between CNN and FC for learning.
<img src="image/13-22-49.png" width="600">

Performance video after adding RNN : https://youtu.be/5bMTUvPmCuQ

## Making item task
We were able to train the network by extracting only a part of a making specific item in the provided data set. However, because the learning result was not as good as Treechop, we decided to use rule base method here.

```
if (place_flag == 0):
  if (planks < 5):
    action['place'] = 0; action['craft'] = 3; 
    action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
    action['attack'] = 0; action['camera'][0] = 0; action['camera'][1] = 0;
    action['forward'] = 0; action['jump'] = 0
  elif (stick < 2):
    action['place'] = 0; action['craft'] = 2; 
    action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
    action['attack'] = 0; action['camera'][0] = 0; action['camera'][1] = 0;
    action['forward'] = 0; action['jump'] = 0
  elif (crafting_table == 0):
    action['place'] = 0; action['craft'] = 4; 
    action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
    action['attack'] = 0; action['camera'][0] = 0; action['camera'][1] = 0;
    action['forward'] = 0; action['jump'] = 0
            
if ( (crafting_table >= 1) & (stick >= 2) & (planks >= 3) ):
  if (place_flag == 0):
    action['place'] = 0; action['craft'] = 0; 
    action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
    action['attack'] = 0; action['camera'][0] = -10; action['camera'][1] = 0;
    action['forward'] = 0; action['jump'] = 0;
    action['equip'] = 0
    place_flag = place_flag + 1
  elif (place_flag == 1):
    action['place'] = 4; action['craft'] = 0; 
    action['nearbyCraft'] = 0; action['nearbySmelt'] = 0
    action['attack'] = 0; action['camera'][0] = 0; action['camera'][1] = 0;
    action['forward'] = 0; action['jump'] = 0;
    action['equip'] = 0
    place_flag = place_flag + 1
  else:
    action['place'] = 0; action['craft'] = 0; 
    action['nearbyCraft'] = 2; action['nearbySmelt'] = 0
    action['attack'] = 0; action['camera'][0] = 0; action['camera'][1] = 0;
    action['forward'] = 0; action['jump'] = 0;
    action['equip'] = 0
```

For making a wooden pickaxe, we need three planks, two sticks, and a crafting table. All three materials can basically be made in log, so the need to collect wood well through the tree chop task can proceed to the next task.

![Making wooden pickaxe](https://github.com/kimbring2/MineRL/blob/master/image/make_wooden_pickaxe.png)

In the case of a treechop task, you can get a decent performance with imitation learning, so if you use the deep learning method and the method does not work correctly like item production, use the rule base method like this Decided to do.
