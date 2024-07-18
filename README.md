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
display.update_display()\


