# Reinforcement learning for route planning in restaurant
##### Tao-Sen Chang  s442720

##### We did the route planning by special algorithm on last task. In this machine learning sub-project I try to show different apporach for the agent who can traversal multiple destinations on the grid system, and of course, get the shortest path of the route. The method is called reinforcement learning. 

## What is reinforcement learning?
##### Reinforcement learning is how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward. The agent makes a sequence of decisions, and learn to perform best actions every step. For example, in my project there is a waiter in the grid and he have to reach many tables for serving the meal, so he must learn the shortest path to get a table.

## How to do that?
##### The idea of how to complete the reinforcement learning is not quite easy. However, there is an example - rat in a maze. Instead of writing the whole codes at beginning, I use the existing codes(tools) by adjusting some parameters to train the agent on our 16x16 grid.
https://www.samyzaf.com/ML/rl/qmaze.html
##### I train the agent(waiter) with rewards and penalties, the waiter in the above grid gets a small penalty for every legal move. The reason is that we want it to get to the target table in the shortest possible path. However, the shortest path to the target table is sometimes long and winding, and our agent (the waiter) may have to endure many errors until he gets to the table.
##### For example, one of the training parameters(rewards) is:
```
if rat_row == win_target_x and rat_col == win_target_y: # if reach the final target
    return 1.0
if mode == 'blocked':   # move to the block in the grid (blocks are tables or kitchen in our grid)
    return self.min_reward - 1
if (rat_row, rat_col) in self.visited: # when get to the visited grid point
    return -10.0/256.0    
if mode == 'invalid': # when move to the boundary
    return -4.0/16.0    
if mode == 'valid': # to make the route shorter, we give a penalty by moving to valid grid point
    return -2.5/256.0
if (rat_row, rat_col) in self.curr_win_targets: # if reach any table
    return 1.0
```


