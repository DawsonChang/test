# Reinforcement learning for route planning in restaurant
##### Tao-Sen Chang  s442720

##### We did the route planning by special algorithm on last task. In this machine learning sub-project I try to show different apporach for the agent who can traversal multiple destinations on the grid system, and of course, get the shortest path of the route. The method is called reinforcement learning. 

## What is reinforcement learning?
##### Reinforcement learning is how software agents ought to take actions in an environment in order to maximize the notion of cumulative reward. The agent makes a sequence of decisions, and learn to perform best actions every step. For example, in my project there is a waiter in the grid and he have to reach many tables for serving the meal, so he must learn the shortest path to get a table.

## How to do that?
##### The idea of how to complete the reinforcement learning is not quite easy. However, there is an example - rat in a maze. Instead of writing the whole algorithm at beginning, I use the existing codes(tools) by adjusting some parameters to train the agent on our 16x16 grid.
https://www.samyzaf.com/ML/rl/qmaze.html
##### I train the agent(waiter) with rewards and penalties, the waiter in the above grid gets a small penalty for every legal move. The reason is that we want it to get to the target table in the shortest possible path. However, the shortest path to the target table is sometimes long and winding, and our agent (the waiter) may have to endure many errors until he gets to the table.
##### For example, one of the training parameters(rewards) are:
```
if rat_row == win_target_x and rat_col == win_target_y: # if reach the final target
    return 1.0
if mode == 'blocked':   # move to the block in the grid (blocks are tables or kitchen in our grid)
    return -1.0
if (rat_row, rat_col) in self.visited: # when get to the visited grid point
    return -0.5    
if mode == 'invalid': # when move to the boundary
    return -0.75    
if mode == 'valid': # to make the route shorter, we give a penalty by moving to valid grid point
    return -0.04
if (rat_row, rat_col) in self.curr_win_targets: # if reach any table
    return 1.0
```
## Q-learning
##### We want to get the maximum reward from each action in a state. Here defines action=π(s).
##### Q(s,a) = the maximum total reward we can get by choosing action a in state s
##### Hence it's obvious that we get the function π(s)=argmaxQ(s,ai)
##### Now the question is how to get Q(s,a)?
##### There is a solution called Bellman's Equation:
##### Q(s,a) = R(s,a) + maxQ(s′,ai)
##### R(s,a) is the reward in current state s, action a. And s′ means the next state, so maxQ(s′,ai) means the maximum reward made by 1 of 4 actions from next state - it's a recursive property.
