# MATH4250-A2

To edit gridworld, change constant field variables found in Grid object. By default, they appear as they are in example 2:

The reward for any action per cell:
_CELLS = [[-0.2, 5.0, -0.2, -0.2, 2.5],
          [-0.2, -0.2, -0.2, -0.2, -0.2],
          [-0.2, -0.2, -0.2, -0.2, 0.0],
          [-0.2, -0.2, -0.2, -0.2, -0.2],
          [0.0, -0.2, -0.2, -0.2, -0.2]]

The color displayed for each cell:
_COLOURS = [['white', 'blue', 'white', 'white', 'green'],
            ['white', 'white', 'white', 'white', 'white'],
            ['white', 'white', 'white', 'white', 'black'],
            ['white', 'white', 'white', 'white', 'white'],
            ['black', 'white', 'red', 'white', 'yellow']]

The cell the agent will be teleported to upon acting on each cell. Set up as a dictionary where the keys are the destinations and the values are the probabilities:
self._LINKS = [[{self[2][3]:1}, None, None, None, {self[2][3]:0.5, self[4][4]:0.5}],
                      [None, None, None, None, None],
                      [None, None, None, None, None],
                      [None, None, None, None, None],
                      [None, None, None, None, None]]

Whether each cell is terminal
self._TERMINAL = [[False, False, False, False, False],
                  [False, False, False, False, False],
                  [False, False, False, False, True],
                  [False, False, False, False, False],
                  [True, False, False, False, False]]

To display a grid as in fig1 and fig2, run:

grid = Grid()\
display = Display(grid)\
display.update_display()




To create a train a policy, you create a new policy object by\
policy = Policy()\
Then, you can run any of\
policy.solve_system()\
policy.policy_iteration()\
policy.value_iteration()\
For each method. Once the value function is estimated, run\
print(policy)\
To display the value function or\
print(policy.display_actions)\
To display the actions. To use Monte Carlo methods, create a new Monte Carlo object by\
MC = MonteCarlo()\
Then, run any of\
MC.exploring_starts(epsilon=0.0)\
MC.behavior_policy()\
To display actions, again:
print(MC._policy.display_actions())


