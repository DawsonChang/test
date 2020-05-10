# Reinforcement learning for route planning in restaurant
##### Tao-Sen Chang    s442720

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
```
self.min_reward = -0.5 * self.maze.size
```
## Q-learning
##### We want to get the maximum reward from each action in a state. Here defines action=π(s).
##### Q(s,a) = the maximum total reward we can get by choosing action a in state s. Hence it's obvious that we get the function π(s)=argmaxQ(s,ai)   Now the question is how to get Q(s,a)?
##### There is a solution called Bellman's Equation: Q(s,a) = R(s,a) + maxQ(s′,ai)
##### R(s,a) is the reward in current state s, action a. And s′ means the next state, so maxQ(s′,ai) means the maximum reward in 4 actions from next state. In the code we have the Experience Class to memorize each "episode", but the memory is limited, therefore if reach the max_memory, then delete the old episode which has lower effect to current episode.
##### There is a coefficient called discount factor, usually denoted by γ which is required for the Bellman equation for stochastic environments. So the new Bellman's Equation can be written as Q(s,a) = R(s,a) + γ * maxQ(s′,ai). This discount factor is to diminish the effects which are far from current state.
```
class Experience(object):
    def __init__(self, model, max_memory=100, discount=0.95):
        self.model = model
        self.max_memory = max_memory
        self.discount = discount
        self.memory = list()
        self.num_actions = model.output_shape[-1]

    def remember(self, episode):
        # episode = [envstate, action, reward, envstate_next, game_over]
        # memory[i] = episode
        # envstate == flattened 1d maze cells info, including rat cell (see method: observe)
        self.memory.append(episode)
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def predict(self, envstate):
        return self.model.predict(envstate)[0]

    def get_data(self, data_size=10):
        env_size = self.memory[0][0].shape[1]   # envstate 1d size (1st element of episode)
        mem_size = len(self.memory)
        data_size = min(mem_size, data_size)
        inputs = np.zeros((data_size, env_size))
        targets = np.zeros((data_size, self.num_actions))
        for i, j in enumerate(np.random.choice(range(mem_size), data_size, replace=False)):
            envstate, action, reward, envstate_next, game_over = self.memory[j]
            inputs[i] = envstate
            # There should be no target values for actions not taken.
            targets[i] = self.predict(envstate)
            # Q_sa = derived policy = max quality env/action = max_a' Q(s', a')
            Q_sa = np.max(self.predict(envstate_next))
            if game_over:
                targets[i, action] = reward
            else:
                # reward + gamma * max_a' Q(s', a')
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets
```

## Training
##### Following is the algorithm for training neural network model to solve the problem. One epoch means one loop of the training, and in each epoch the agent will finally become either "win" or "lose". 
##### Another coefficient "epsilon" is exploration factor which decides the probability of whether the agent will perform new actions instead of following the previous experiences (which is called exploitation). By this way the agent could not only collect better rewards from previous experiences, but also have the chances to explore unknow area where might get more rewards. If one of the strategy is determined, then let's start training it by neural network. (inputs: size equals to the maze size, targets: size is the same as the number of actions (4 in our case)).
```
# Exploration factor
epsilon = 0.1
def qtrain(model, maze, **opt):
    global epsilon
    n_epoch = opt.get('n_epoch', 15000)
    max_memory = opt.get('max_memory', 1000)
    data_size = opt.get('data_size', 50)
    weights_file = opt.get('weights_file', "")
    name = opt.get('name', 'model')
    start_time = datetime.datetime.now()

    # If you want to continue training from a previous model,
    # just supply the h5 file name to weights_file option
    if weights_file:
        print("loading weights from file: %s" % (weights_file,))
        model.load_weights(weights_file)

    # Construct environment/game from numpy array: maze (see above)
    qmaze = Qmaze(maze)

    # Initialize experience replay object
    experience = Experience(model, max_memory=max_memory)

    win_history = []   # history of win/lose game
    n_free_cells = len(qmaze.free_cells)
    hsize = qmaze.maze.size//2   # history window size
    win_rate = 0.0
    imctr = 1
    pre_episodes = 2**31 - 1

    for epoch in range(n_epoch):
        loss = 0.0
        #rat_cell = random.choice(qmaze.free_cells)
        #rat_cell = (0, 0)
        rat_cell = (12, 12)

        qmaze.reset(rat_cell)
        game_over = False

        # get initial envstate (1d flattened canvas)
        envstate = qmaze.observe()

        n_episodes = 0
        while not game_over:
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            prev_envstate = envstate
            # Get next action
            if np.random.rand() < epsilon:
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(prev_envstate))

            # Apply action, get reward and new envstate
            envstate, reward, game_status = qmaze.act(action)
            if game_status == 'win':
                print("win")
                win_history.append(1)
                game_over = True
                # save_pic(qmaze)
                if n_episodes <= pre_episodes:
                    # output_route(qmaze)
                    print(qmaze.visited)
                    with open('res.data', 'wb') as filehandle:
                        pickle.dump(qmaze.visited, filehandle)
                    pre_episodes = n_episodes
                    
            elif game_status == 'lose':
                print("lose")
                win_history.append(0)
                game_over = True
                # save_pic(qmaze)
            else:
                game_over = False

            # Store episode (experience)
            episode = [prev_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes += 1

            # Train neural network model
            inputs, targets = experience.get_data(data_size=data_size)
            h = model.fit(
                inputs,
                targets,
                epochs=8,
                batch_size=16,
                verbose=0,
            )
            loss = model.evaluate(inputs, targets, verbose=0)
            
        
        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize
    
        dt = datetime.datetime.now() - start_time
        t = format_time(dt.total_seconds())
        
        template = "Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}"
        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))
```

## Testing
##### Use this algorithm to our 16x16 grid and train.
```
grid = [[1 for x in range(16)] for y in range(16)]
table1 = Table(2, 2)
table2 = Table (2,7)
table3 = Table(2, 12)
table4 = Table(7, 2)
table5 = Table(7, 7)
table6 = Table(7, 12)
table7 = Table(12, 2)
table8 = Table(12, 7)

kitchen = Kitchen(13, 13)
maze = np.array(grid)
model = build_model(maze)
qtrain(model, maze, epochs=1000, max_memory=8*maze.size, data_size=32)
```
##### Also I create a list called win_targets to put the position of tables in the grid.
```
win_targets = [(4, 4),(4, 9),(4, 14),(9, 4),(9, 9),(9, 14),(14, 4),(14, 9)]
```
##### After tons of trainings, I realize it is not an easy task to obtain the shortest route in every training - that means most of the trainings are failed - especially in the case that the win_targets has more targets. For example, the result of training 8 targets is like this(part of result):
```
...
Epoch: 167/14999 | Loss: 0.0299 | Episodes: 407 | Win count: 63 | Win rate: 0.422 | time: 2.44 hours
Epoch: 168/14999 | Loss: 0.0112 | Episodes: 650 | Win count: 63 | Win rate: 0.414 | time: 2.46 hours
Epoch: 169/14999 | Loss: 0.0147 | Episodes: 392 | Win count: 64 | Win rate: 0.422 | time: 2.47 hours
Epoch: 170/14999 | Loss: 0.0112 | Episodes: 668 | Win count: 65 | Win rate: 0.422 | time: 2.48 hours
Epoch: 171/14999 | Loss: 0.0101 | Episodes: 487 | Win count: 66 | Win rate: 0.430 | time: 2.50 hours
Epoch: 172/14999 | Loss: 0.0121 | Episodes: 362 | Win count: 67 | Win rate: 0.438 | time: 2.51 hours
Epoch: 173/14999 | Loss: 0.0101 | Episodes: 484 | Win count: 68 | Win rate: 0.445 | time: 2.52 hours
...
```
##### The only one which is successful contains 4 targets(win_targets = [(4, 4),(4, 9),(4, 14),(9, 4)]) 
```
...
Epoch: 223/14999 | Loss: 0.0228 | Episodes: 30 | Win count: 165 | Win rate: 0.906 | time: 64.02 minutes
Epoch: 224/14999 | Loss: 0.0160 | Episodes: 52 | Win count: 166 | Win rate: 0.906 | time: 64.09 minutes
Epoch: 225/14999 | Loss: 0.0702 | Episodes: 34 | Win count: 167 | Win rate: 0.914 | time: 64.14 minutes
Epoch: 226/14999 | Loss: 0.0175 | Episodes: 40 | Win count: 168 | Win rate: 0.922 | time: 64.19 minutes
Epoch: 227/14999 | Loss: 0.0271 | Episodes: 46 | Win count: 169 | Win rate: 0.930 | time: 64.25 minutes
Epoch: 228/14999 | Loss: 0.0194 | Episodes: 40 | Win count: 170 | Win rate: 0.938 | time: 64.30 minutes
...
Epoch: 460/14999 | Loss: 0.0236 | Episodes: 60 | Win count: 401 | Win rate: 1.000 | time: 1.48 hours
Reached 100% win rate at epoch: 460
n_epoch: 460, max_mem: 2048, data: 32, time: 1.48 hours
```
##### In my opinion, there are 3 reasons cause such bad results.
##### 1. The parameters in the algorithm are not optimal including the rewards, exploration rate, and discount factor. To adjust the parameters and to validate them costs lots of time, and the most intuitive way is always not the best solution. For example, the parameters of 4 targets are fine, but if the number of targets expanded to 8, the parameters are not just 1/2 of the original ones.
##### 2. Beacuse of the exploration rate, every time the same training and testing data may have different result. It increases the difficulty to verify our result. Only way to check whether the parameters generate ideal results is training continuously until we collect sufficient data.
##### 3. The algorithm is for a rat in a maze at beginning, and the number of default target is only one. If we apply it for multiple targets, there may be inadequate for some reason. Moreover, the default size is 7x7 grid. It is possible that the 16x16 grid is too huge for this algorithm.
