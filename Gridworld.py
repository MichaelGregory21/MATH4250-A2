import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

'''
A cell is a space which can be traversed by an agent
'''
class Cell:
    '''
    Constructor for objects of type Cell
    @param int x The x coordinate associated with this cell
    @param int y The y coordinate associated with this cell
    @param str colour The colour of this cell
    @param dict links A map which associates the destination the agent will be transported to with a probability
    @param lambda reward The reward acquired when this cell is traversed in a state
    @param boolean terminal True if this cell is a terminal state. False, otherwise
    '''
    def __init__(self, x, y, colour=None, links=None, reward=None, terminal=False):
        self._x = x
        self._y = y
        self._colour = colour
        self._links = links
        self._reward = reward
        self._terminal = terminal

    '''
    Accessor for the colour of this cell
    @return str The colour of this cell
    '''
    def getColour(self):
        return self._colour

    '''
    Accessor for the destination linked to this cell
    @return Cell The destination linked to this cell
    '''
    def getLink(self):
        if self._links is not None: return np.random.choice(list(self._links.keys()), p=list(self._links.values()))

    '''
    Accessor for the probability that the given cell is the destination linked to this cell
    @param Cell cell The cell you want to check the probability of
    @return float The probability of the given cell
    '''
    def getProbabilityLinked(self, cell):
        return self._links.get(cell, 0)

    '''
    Accessor for the reward acquired when this cell is traversed
    @return int The reward acquired when this cell is traversed
    '''
    def getReward(self, action):
        return self._reward(action)

    '''
    Accessor for the x coordinate of this cell
    @return int The x coordinate of this cell
    '''
    def getX(self):
        return self._x

    '''
    Accessor for the y coordinate of this cell
    @return int The y coordinate of this cell
    '''
    def getY(self):
        return self._y
    
    '''
    Is this a terminal cell
    @return boolean True if this cell is terminal. False, otherwise
    '''
    def isTerminal(self):
        return self._terminal

    '''
    Does this cell have a link
    @return boolean True if this cell has a link. False, otherwise
    '''
    def hasLinks(self):
        return self._links is not None

    '''
    Mutator for the colour of this cell
    @param str colour The colour you want to set for this cell
    '''
    def setColour(self, colour):
        self._colour = colour

    '''
    Mutator for the link associated with this cell
    @param dict links The links you want to associate with this cell
    @param float probability The probability that the given cell will be choosen
    '''
    def setLinks(self, links):
        self._links = links

    '''
    Mutator for the reward associated with this cell
    @param lambda reward The reward you want to associate with this cell
    '''
    def setReward(self, reward):
        self._reward = reward

    '''
    Mutator to the terminal parameter of this cell
    @param boolean terminal Set to True if you want this cell to be terminal. False, otherwise
    '''
    def setTerminal(self, terminal):
        self._terminal = terminal
    
'''
A grid is a constant sized array of cells with constant rewards
'''
class Grid:
    _WIDTH = 5
    _HEIGHT = 5  
    
    _CELLS = [[-0.2, 5.0, -0.2, -0.2, 2.5],
              [-0.2, -0.2, -0.2, -0.2, -0.2],
              [-0.2, -0.2, -0.2, -0.2, 0.0],
              [-0.2, -0.2, -0.2, -0.2, -0.2],
              [0.0, -0.2, -0.2, -0.2, -0.2]]

    _COLOURS = [['white', 'blue', 'white', 'white', 'green'],
                ['white', 'white', 'white', 'white', 'white'],
                ['white', 'white', 'white', 'white', 'black'],
                ['white', 'white', 'white', 'white', 'white'],
                ['black', 'white', 'red', 'white', 'yellow']]

    '''
    Constructor for objects of type Grid
    '''
    def __init__(self):
        self._cells = [[Cell(x=x, y=y) for y in range(self._HEIGHT)] for x in range(self._WIDTH)]
        self._LINKS = [[{self[2][3]:1}, None, None, None, {self[2][3]:0.5, self[4][4]:0.5}],
                      [None, None, None, None, None],
                      [None, None, None, None, None],
                      [None, None, None, None, None],
                      [None, None, None, None, None]]

        self._TERMINAL = [[False, False, False, False, False],
                          [False, False, False, False, False],
                          [False, False, False, False, True],
                          [False, False, False, False, False],
                          [True, False, False, False, False]]
        for cell in self:
            cell.setReward(lambda action, cell=cell: -0.5 if action(cell) is None else self._CELLS[action(cell).getY()][action(cell).getX()])
            cell.setColour(self._COLOURS[cell.getY()][cell.getX()])
            cell.setLinks(self._LINKS[cell.getY()][cell.getX()])

    
    '''
    Returns the cell that you arrive at after taking the given action in the given cell
    @param Cell cell The cell you take the given action in
    @param Action action The action you take
    @return Cell The cell you arrive at
    '''
    def destination(self, cell, action):
        return action(cell)

    '''
    Accessor for row of cells
    @param int index The index of the row you want to access
    @return list The row you want to access
    '''
    def __getitem__(self, index):
        return self._cells[index]

    '''
    Accessor for generator of all cells in this grid
    @return generator Generator of cells in this grid
    '''
    def __iter__(self):
        for row in self._cells:
            for element in row:
                yield element

    '''
    Accessor for the number of cells in this grid
    @return int The number of cells in this grid
    '''
    def __len__(self):
        return self._HEIGHT * self._WIDTH

    '''
    Accessor for string representing this grid
    @return str String representing this grid
    '''
    def __str__(self):
        return '\n'.join(['[ ' + ' '.join(map(lambda x: "{:.1f}".format(x).ljust(4), row)) + ' ]' for row in self._CELLS])

'''
An action is a choice an agent can make
'''
class Action:
    '''
    Constructor for objects of type Action
    @param lambda func The result of undergoing this action
    @param str name The name of this action
    '''
    def __init__(self, func=lambda state: state, name=None):
        self._func = func
        self._name = name

    '''
    Accessor for the result of undergoing this action
    @param Cell state The state that this action is taken in
    '''
    def __call__(self, state):
        return self._func(state)

    '''
    Accessor for the name of this action
    @return str The name of this action
    '''
    def __str__(self):
        return self._name


'''
A policy is a set of rules that an agent must follow
'''
class Policy:
    _GAMMA = 0.95

    '''
    Constructor for objects of type Policy
    '''
    def __init__(self, policy=None, actions=None, states=None):
        if actions == None: self._ACTIONS = [Action(func=lambda cell, dx=dx, dy=dy: self._STATES[cell.getX() + dx][cell.getY() + dy] 
                                if isinstance(cell, Cell) and 0 <= cell.getX() + dx < self._STATES._WIDTH and 0 <= cell.getY() + dy < self._STATES._HEIGHT
                                else None, name=name) for (dx,dy,name) in [(0,-1,'north'),(1,0,'east'),(0,1,'south'),(-1,0,'west')]]
        else: self._ACTIONS = actions
        if states == None: self._STATES = Grid()
        else: self._STATES = states
        self._REWARDS = {cell.getReward(action) for action in self._ACTIONS for cell in self._STATES}
        if policy == None: policy={(state, action): 1 / 4 for state in self._STATES for action in self._ACTIONS}
        self._policy = policy
        self._value = {state: 0 for state in self._STATES}


    '''
    Mutator for probability of given action in the given state
    @param Action action The action you want to set the probability of
    @param Cell state The cell you want to set the policy for
    @param float The new probability you want to assign to the given action in the given state
    '''
    def set_probability(self, action, state, prob):
        self._policy[(state, action)] = prob

    '''
    Mutator for the value of this policy
    @param dict value The value of each state according to this policy
    '''
    def setValue(self, value):
        for cell in self._STATES:
            self._value[cell] = value[cell]

    '''
    Set the policy of this policy to the given policy
    @param dict policy The policy you want to give to this policy
    '''
    def setPolicy(self, policy):
        self._policy = policy

    '''
    Accessor for the probability that this policy will yield the given new_state and reward, starting in the given old_state and taking the given action
    @param Cell new_state The state you want to check the likelihood of ending up in
    @param float The reward you want to check the likelihood of recieving
    @param Cell old_state The state you start in
    @param Action action The action you take
    @return float The probability that this policy will yield the given result
    '''
    def get_probability(self, new_state, reward, old_state, action):
        if old_state.hasLinks():
            return old_state.getProbabilityLinked(new_state) * int(old_state.getReward(action) == reward)
        return int(action(old_state) == new_state) * int(old_state.getReward(action) == reward)

    
    '''
    Accessor for the value of this policy
    @param Cell state The state that you want to evaluate the policy at
    '''
    def getValue(self, state):
        return self._value[state]

    def getPolicy(self):
        return self._policy

    '''
    Evaluate this policy arithmetically. This function is in O(n^3) and is only feasible in small environments
    '''
    def solve_system(self):
        P = np.array([[sum([self(s,action) * sum([self.get_probability(s_prime, reward, s, action)
                                                  for reward in self._REWARDS])
                            for action in self._ACTIONS])
                       for s_prime in self._STATES]
                      for s in self._STATES])
        R = np.array([sum([self(s,action) * sum([self.get_probability(s_prime, reward, s, action) * reward
                            for s_prime in self._STATES
                            for reward in self._REWARDS])
                           for action in self._ACTIONS])
                      for s in self._STATES])
        v = np.linalg.inv(np.eye(len(self._STATES)) - self._GAMMA * P).dot(R)
        self.setValue({s: v_i for s, v_i in zip(self._STATES, v)})
                
            
    '''
    Evaluates and sets the value of the policy that this agent is currently following
    @param float theta A small threshold determining the accuracy of estimation. 0.01, by default
    '''
    def policy_iteration(self, theta=0.01):
        if not theta > 0: raise ValueError('theta must be greater than 0')
        # Initialize V to assign each cell in the grid to its own random number is it is not terminal. Otherwise, it is assigned 0.
        V = {state: (0 if state.isTerminal() else random.random()) for state in self._STATES}

        while True:
            delta = 0
            for state in self._STATES:
                v = V[state]
                V[state] = sum([self(state, action) *
                                sum([self.get_probability(new_state, reward, state, action) *
                                     (reward + self._GAMMA * V[new_state])
                                     for new_state in self._STATES
                                     for reward in self._REWARDS])
                                for action in self._ACTIONS])
                delta = max(delta, abs(v - V[state]))
            if delta < theta:
                policy_stable = True
                for state in self._STATES:
                    old_action = self(state)
                    best_action = max(self._ACTIONS, key=lambda action: sum([self.get_probability(new_state, reward, state, action) *
                                                                            (reward + self._GAMMA + self.getValue(new_state))
                                            for new_state in self._STATES
                                            for reward in self._REWARDS]))
                    assignment = lambda action: 1 if action == best_action else 0
                    for action in self._ACTIONS: self.set_probability(action, state, assignment(action))
                    if old_action != self(state):
                        policy_stable = False
                if policy_stable:
                    for state in self._STATES: self._value[state] = V[state]
                    break
                    
            

    '''
    Undergoes value iteration method of policy evaluation and improvement
    @param float theta A small threshold determining the accuracy of estimation. 0.01, by default
    '''
    def value_iteration(self, theta=0.01):
        if not theta > 0: raise ValueError('theta must be greater than 0')
        # Initialize V to assign each cell in the grid to its own random number is it is not terminal. Otherwise, it is assigned 0.
        V = {state: (0 if state.isTerminal() else random.random()) for state in self._STATES}
        while True:
            delta = 0
            for state in self._STATES:
                v = V[state]
                V[state] = max([sum([self.get_probability(new_state, reward, state, action) * (reward + self._GAMMA * V[new_state]) for new_state in self._STATES for reward in self._REWARDS]) for action in self._ACTIONS])
                delta = max(delta, abs(v - V[state]))
            if delta < theta:
                break
        for state in self._STATES:
            best_action = max(self._ACTIONS, key=lambda action: sum([self.get_probability(new_state, reward, state, action) * (reward + self._GAMMA * V[new_state]) for new_state in self._STATES for reward in self._REWARDS]))
            assignment = lambda action: 1 if action == best_action else 0
            for action in self._ACTIONS: self.set_probability(action, state, assignment(action))
            self._value[state] = V[state]

    '''
    Returns a text display of the action taken for each state
    @return str Text display of actions
    '''
    def display_actions(self):
        items = list(self._value.values())
        n = self._STATES._HEIGHT
        m = self._STATES._WIDTH
        matrix = [[self(self._STATES[i][j]) for i in range(m)] for j in range(n)]
        return '\n'.join(['[ ' + ' '.join(map(lambda x: str(x)[:10].ljust(10), row)) + ' ]' for row in matrix])

            

    '''
    Does one of two things depending on parameters:
    1. If action == None, returns behavior according to policy
    2. If action =/= None, returns probability that the given action is taken in the given state
    '''
    def __call__(self, state, action=None):
        if not action == None: return self._policy[(state, action)]
        probabilities = [self._policy.get((state, action), 0) for action in self._ACTIONS]
        return np.random.choice(self._ACTIONS, p=probabilities)

    '''
    Accessor for the string representing the value of the policy
    @return str The string representing the value of this policy
    '''
    def __str__(self):
        items = list(self._value.values())
        n = self._STATES._HEIGHT
        m = self._STATES._WIDTH
        matrix = [[self.getValue(self._STATES[i][j]) for i in range(m)] for j in range(n)]
        return '\n'.join(['[ ' + ' '.join(map(lambda x: str(x)[:10].ljust(10), row)) + ' ]' for row in matrix])

    
'''
An agent is an actor which exists in a state
'''
class Agent:

    '''
    Constructor for objects of type Agent
    @param Cell state The state that this agent exists in
    '''
    def __init__(self, state):
        self._state = state

    '''
    Change this agents state to the given state
    @param Cell state The state you want this agent to move to
    '''
    def move(self, state):
        if state is not None:
            if self._state.hasLinks():
                self._state = self._state.getLink()
            else:
                self._state = state

    '''
    Accessor for this agents state
    @return Cell The state that this agent is in
    '''
    def getState(self):
        return self._state

class MonteCarlo:

    def __init__(self):
        self._states = Grid()
        self._actions = [Action(func=lambda cell, dx=dx, dy=dy: self._states[cell.getX() + dx][cell.getY() + dy] 
                                if isinstance(cell, Cell) and 0 <= cell.getX() + dx < self._states._WIDTH and 0 <= cell.getY() + dy < self._states._HEIGHT
                                else None, name=name) for (dx,dy,name) in [(0,-1,'north'),(1,0,'east'),(0,1,'south'),(-1,0,'west')]]
        self._policy = Policy(states = self._states, actions = self._actions)

    def exploring_starts(self, T=100, epsilon=0, num_episodes=200, permute=False):

        # Initialize variables
        for state in self._states:
            random_numbers = [random.random() for _ in range(len(self._actions))]
            random_numbers = [i/sum(random_numbers) for i in random_numbers]
            for action in self._actions:
                self._policy.set_probability(action,state,random_numbers[self._actions.index(action)])

        Q = {(state, action): 0 for state in self._states for action in self._actions}

        Returns = {(state, action): [] for state in self._states for action in self._actions}

        j=0
        while j<num_episodes:
            j+=1

            S0 = np.random.choice(self._states)
            A0 = np.random.choice(self._actions)
            
            episodes = [(S0, A0, S0.getReward(A0))]

            # Populate epsiodes
            for t in range(T):
                if permute and random.random() < 0.1: self._states[1][0], self._states[4][0] = self._states[4][0], self._states[1][0]
                if episodes[t][1](episodes[t][0]) is None: S = episodes[t][0]
                else: S = episodes[t][1](episodes[t][0])
                A = self._policy(S)
                episodes.append((S,A,S.getReward(A)))
                if S.isTerminal(): break

            G=0

            for t in reversed(range(len(episodes))):
                St,At,Rt1 = episodes[t]
                G = self._policy._GAMMA * G + Rt1

                if not any(s_a == (St,At) for s_a in episodes[:t]):
                    Returns[(St,At)].append(G)
                    Q[(St,At)] = np.mean(Returns[St,At])
                    best_action = max(self._actions, key=lambda a: Q[(St,a)])
                    for action in self._actions:
                        if action == best_action:
                            self._policy._policy[(St,action)] = 1 - epsilon + (epsilon / len(self._actions))
                        else:
                            self._policy._policy[(St,action)] = epsilon / len(self._actions)

    def behavior_policy(self, T=100, num_episodes=1000):
        Q = {(state, action): random.random() for state in self._states for action in self._actions}
        C = {(state, action): 0 for state in self._states for action in self._actions}
        for state in self._states:
            best_action = max(self._actions, key=lambda action: Q[(state,action)])
            for action in self._actions:
                if action == best_action:
                    self._policy.set_probability(action,state,1)
                else:
                    self._policy.set_probability(action,state,0)

        j=0
        while j<num_episodes:
            j+=1

            # Initialize b-policy
            b = Policy(states=self._states,actions=self._actions)
            for state in self._states:
                for action in self._actions:
                    b.set_probability(action,state,1 / len(self._actions))


            # Populate epsiodes
            S0 = np.random.choice(self._states)
            A0 = np.random.choice(self._actions)
            episodes = [(S0, A0, S0.getReward(A0))]
            for t in range(T):
                if episodes[t][1](episodes[t][0]) is None: S = episodes[t][0]
                else: S = episodes[t][1](episodes[t][0])
                A = b(S)
                episodes.append((S,A,S.getReward(A)))
                if S.isTerminal(): break

            G = 0
            W = 1

            for t in reversed(range(len(episodes))):
                St,At,Rt1 = episodes[t]
                G = b._GAMMA * G + Rt1

                C[(St,At)] += W
                Q[(St,At)] += (W/C[(St,At)]) * (G-Q[(St,At)])

                best_action = max(self._actions, key=lambda action: Q[(St,action)])
                for action in self._actions:
                    if action == best_action:
                        self._policy.set_probability(action,St,1)
                    else:
                        self._policy.set_probability(action,St,0)
                if At != self._policy(St): break
                W *= 1 / (b(St,At))

                

'''
A display is a GUI that simulates the behavior of an agent in a grid
'''
class Display:

    '''
    Constructor for objects of type Display
    @param Grid grid The grid you want to display
    @param Agent agent The agent you want to display
    '''
    def __init__(self, grid, agent=None):
        self._grid = grid
        self._agent = agent
        self._fig, self._ax = plt.subplots()


    '''
    Refresh the display to reflect the current state of the grid and agent
    '''
    def update_display(self):
        self._ax.clear

        for i in range(self._grid._HEIGHT):
            for j in range(self._grid._WIDTH):
                cell = self._grid[j][i]
                self._ax.add_patch(patches.Rectangle((j,i), 1, 1, facecolor=cell.getColour()))
        if self._agent is not None: self._ax.add_patch(patches.Circle((self._agent.getState().getX() + 0.5, self._agent.getState().getY() + 0.5), radius=0.25, facecolor='black'))


        self._ax.set_xlim([0, self._grid._WIDTH])
        self._ax.set_ylim([0, self._grid._HEIGHT])
        self._ax.set_aspect('equal')
        self._ax.grid(which='major', axis='both', linestyle='-', color='black', linewidth=1)
        plt.gca().invert_yaxis()

        plt.draw()
        plt.pause(0.001)


MC = MonteCarlo()
MC.exploring_starts(permute=False)
print(MC._policy.display_actions())

print()

