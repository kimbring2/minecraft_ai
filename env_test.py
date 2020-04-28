'''
import minerl
import gym
env = gym.make('MineRLNavigateDense-v0')


obs = env.reset()

done = False
while not done:
    action = env.action_space.sample() 
 
    # One can also take a no_op action with
    # action =env.action_space.noop()
    
 
    obs, reward, done, info = env.step(
        action)
'''

import minerl

# YOU ONLY NEED TO DO THIS ONCE!W
#minerl.data.download('/your/local/path')

data = minerl.data.make('MineRLObtainDiamond-v0', data_dir='/media/kimbring2/6224AA7924AA5039/minerl_data')

# Iterate through a single epoch gathering sequences of at most 32 steps
for current_state, action, reward, next_state, done \
    in data.sarsd_iter(
        num_epochs=1, max_sequence_len=32):

        # Print the POV @ the first step of the sequence
        print(current_state['pov'][0])W

        # Print the final reward pf the sequence!
        print(reward[-1])

        # Check if final (next_state) is terminal.
        print(done[-1])

        # ... do something with the data.
        print("At the end of trajectories the length"
              "can be < max_sequence_len", len(reward))