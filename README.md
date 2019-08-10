# MineRL Imitation Learning

This github code is for solving a MineRLTreechop-v0, MineRLNavigate-v0 environment of MineRL. I use a Convolution Neural Network for traning a agent. 

# How to use it

First, you need to install dependent package by using a requirement.txt file. 

Second, you need to modify a path for model and summary of tensorflow in MineRL_IL.ipynb file.

Third, you should check code operate properly by running IPython Notebook.

# Result
![Treechop-v0 agent video](https://github.com/kimbring2/MineRL/blob/master/monitor/ezgif.com-video-to-gif.gif)

After finishing a training, agent can go to a two tree in environment and attack it for collecting wood. But, it stops after then. Thus, agent can collect only 2 reward.

I need to solve this problem for collection more wood in one environment.

![Treechop-v0 traning loss graph](https://github.com/kimbring2/MineRL/blob/master/image/14-47-20.png)

Loss graph shows that there is no ploblem in traning process. However, it looks like it need more epoch.
