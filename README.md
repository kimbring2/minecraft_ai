<img src="image/18-28-41.png" width="350">

The code uploaded here is for MineRL Competition 2019. The purpose of the competition is to find an effective reinforcement learning algorithm using human play datasets. 

# Introduction
I started participating at the beginning of this competition, but I wasn't able to participate correctly in the Retro Contest held at OpenAI last year, so I wanted to finish it this time. What is required in the competition is a method for efficiently learning a reinforcement learning model using given data. For these reasons, we approached the most basic network by using supervised learning and learning according to a given data set.

# MineRL environment
Requesting in the competition is to resolve MineRLObtainDiamond-v0. In order to solve this, advance work such as moving to a specific place or collecting trees is necessary.

<img src="image/19-14-57.png" width="600">

The agent obtains information on items currently possessed, including screen information on the screen during game play. In addition, actions such as camera rotation, advancement, attack, item creation, item drop, and item equipment can be performed.

<img src="image/22-02-25.png" width="400"> <img src="image/22-02-49.png" width="400">

## How to use human play dataset
Since it is efficient reinforcement learning using game play data of the target person of the competition, a large capacity play data set is given. Rather than learning everything from the beginning in Reinforcement Learning, using this data to let the network learn in advance is a faster way to get diamonds. 

<img src="image/22-02-25.png" width="400"> <img src="image/05-53-53.png" width="600">

Fortunately, in addition to providing a data set from the organizer, it also provides a viewer in the form of a GUI, so that for the first time like Minecraft, participants can easily understand the goals of the game. In particular, it is possible to check not only the screen but also the behavior of the agent and the change in the compensation value associated therewith in units of frames.

## Network Structure
The network structure is largely composed of a CNN part that receives the current state value of the agent and an FC part that outputs the next action value.
![CNN structure image](https://github.com/kimbring2/MineRL/blob/master/image/03-17-22.png)


Minecraft does not allow you to see all game information at once. Therefore, it can be predicted that the behavior in the current frame is affected by the information in the previous frame. Therefore, we could use the RNN network additionally to account for these temporal depencies and see better performance in the Treechop task.
![RNN+CNN structure image](https://github.com/kimbring2/MineRL/blob/master/image/19-16-46.png)

## Preprosseing
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

![Making wooden pickaxe](https://github.com/kimbring2/MineRL/blob/master/image/make_wooden_pickaxe.png)

For making a wooden pickaxe, we need three planks, two sticks, and a crafting table. All three materials can basically be made in log, so the need to collect wood well through the tree chop task can proceed to the next task. In other words, only the treechop task is executed until 5 or more logs are collected, and then the action to create the required number of planks, sticks, and crafting_table is set directly using the if statement. The detail code can be found in the uploaded file.

## Combine various tasks into one 
In order to solve the MineRLObtainIronPickaxe-v0 environment, solving a simple environment is needed. Especially in the case of the competition, since the starting position is random, the character can be activated first in an area where there are no trees around. In this case, you need to move to the area where the tree is, instead of trying to do the treechop task right away.

![Combining agent video](https://github.com/kimbring2/MineRL/blob/master/monitor/navi_tree.gif)

Once you start the game, it will move for a certain amount of time, as you learned in the MineRLNavigate-v0 environment to walk around a specific place on the map. After that, if there is no increase in the compensation value over a certain time to act as learned in the MineRLTreechop-v0 environment, the two tasks are repeatedly used like Navigate again.

## Pretrain model file
In addition to code sharing, Tensorflow weights file that is trained from Imitation Learning is shared.
MineRLNavigate-v0 : https://drive.google.com/drive/folders/17vVjFu0P1gd6rXRFSwfze5gvgutApemo?usp=sharing
MineRLTreechop-v0 : https://drive.google.com/drive/folders/1pIBxe5G0x_NU85S3wxYUDDhhHNlSRArQ?usp=sharing
