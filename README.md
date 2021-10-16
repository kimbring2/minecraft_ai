# Introduction
Code for playing the Minecraft using the Deep Learning 

# Dependencies
1. minerl
2. tensorflow 2

# Network architecture
<img src="image/minecraft_network.png" width="1000">

# Reference
1. 

# How to run 
First, you need to check everything about MineRL package works well in your PC. Please visit https://minerl.readthedocs.io/en/latest/.

If everything is fine, open and run the cell of MineRL-TreeChop(LSTM).ipynb file. It is simple A2C agent for TreeChop task of Minecraft. Firat, agent is trained via Supervised Learning. Next, it begins Reinforcement Learning phase from pretrained model.

# How to check training goes well
1. Loss check : Loss should fall to almost 0 as shown in the graph above when the policy network is trained by the Supervised Learning manner.

# Detailed inforamtion
Please check Medium article(https://medium.com/@dohyeongkim/deep-q-learning-from-demonstrations-dqfd-for-minecraft-tutorial-1-4b462a18de5a) for more information.
