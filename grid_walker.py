import numpy as np
from tkinter import *
import tkinter
import time
from collections import deque


class GridWalker():
    '''
    Defines an enviroment to be used in RL
    
    '''
    def __init__(self, model_channels):

        self.grid = np.zeros((5,5))
        self.grid_deque = deque(maxlen = 10)
        self.steps = 0
        self.render_flag = True
        self.reset()
        self.model_channels = model_channels

    def toggle_render(self, event):
        '''
        Press R in the app window to toggle rendering
        '''
        if event.char == 'r':
            print('Toggled render option')
            self.render_flag = not self.render_flag

    def setup_gui(self):
        '''
        Initiates the app window
        '''
        self.window_h = (1+self.grid.shape[0])*20
        self.window_w = (1+self.grid.shape[1])*20
        
        self.gui = Tk()
        self.c = Canvas(self.gui ,width=self.window_w ,height=self.window_h)
        self.c.pack()
        self.gui.bind("<Key>", self.toggle_render)
        self.gui.title("Grid Walker (5x5))")
        self.build_tiles()

    def render(self, sleep_time = 0.0005):
        '''
        Renders the window at each iteration
        '''
        if self.render_flag == True:
            try:
                self.c.delete('all')
                self.build_tiles()
                self.gui.update()

            except:
                self.setup_gui()
        else:
            # In order to keep evaluating the toggle_render callback, 
            # we have to still update the gui (altought not as frequently)
            if np.random.randint(0,10) > 8:
                self.gui.update()


    
    def build_tiles(self):
        '''
        Builds the canvas objects as defined in self.grid
        '''
        for i in range(0,self.grid.shape[0]):
            for j in range(0,self.grid.shape[1]):
                center = [(i+1)*20, (j+1)*20]
                x1 = center[0]-9
                x2 = center[0]+9
                y1 = center[1]-9
                y2 = center[1]+9

                if self.grid[i,j] == 1:
                    color = 'red'
                elif self.grid[i,j] == -1:
                    color = 'black'
                else:
                    color = 'pink'
                self.c.create_rectangle(x1,y1,x2,y2,fill=color)

    def reset(self):
        '''
        Reset game, with random barriers in the grid

        '''
        self.grid = np.zeros((5,5))
        self.position = np.array([0,0])
        self.steps = 0

        blocks = np.random.randint(0,5,(2,2))
        for b in blocks:
            self.grid[b[0], b[1]] = -1 #coluna, linha
        
        # Remove blocks that block the entry/exit of the grid
        self.grid[0,0] = 1
        self.grid[1,0] = 0
        self.grid[0,1] = 0
        
        self.grid[4,4] = 1
        self.grid[4,3] = 0
        self.grid[3,4] = 0
        # Fill deque at startup
        for i in range(0,10):
            self.grid_deque.append(self.grid)

    def get_state(self):
        '''
        Returns the game state at current iteration
        Returns :
            - current grid
            - plus (self.model_channels - 2) of the past grids (time aspect, memory)
            - player_state grid, defined by only the 3x3 squares near the player
        '''
        player_state = np.zeros((5,5))
        custom_grid = np.zeros((5+1,5+1))-1
        custom_grid[0:5,0:5] = self.grid

        for i in range(-1,2):
            for j in range(-1,2):
                try:
                    player_state[i+1,j+1] = custom_grid[self.position[0]+i, self.position[1]+j]
                except:
                    player_state[i+1,j+1] = -1

                    
        state = np.array(self.grid_deque)[-self.model_channels:]
        state[0,:,:] = player_state
        return state
        


    def action(self, x, y):
        '''
        Apply movement [x,y] to the tile
        If the movement ends in, or passes through in any way,
        a [-1] tyle, it is considered invalid 
        '''
        assert x in [0,-1,1], 'Invalid move for x'
        assert y in [0,-1,1], 'Invalid move for y'
        pos0 = self.position
        posA = pos0 + [x,0]
        posB = pos0 + [0,y]
        posC = pos0 + [x,y]
        
        legal_move = True
        done = 0

        for pos in [posA, posB, posC]:
            if pos[0] < 0 or pos[0] > self.grid.shape[0]-1:
                legal_move = False
                reward = -10
                done = 0
                continue

            if pos[1] < 0 or pos[1] > self.grid.shape[1]-1:
                legal_move = False
                reward = -10
                done = 0
                continue

            tile_type = self.grid[pos[0], pos[1]]
            if tile_type == -1:
                legal_move = False
                reward = -10
                done = 0
                continue

        if legal_move == True:
            if np.all(posC == self.position):
                reward = -10
                done = 0
            else:
                self.grid[self.position[0], self.position[1]] = 0
                self.grid[posC[0],posC[1]] = 1
                self.position = posC

                reward = 0 # + (1/(4-self.position[0]+3)-1/12)*8 + (1/(4-self.position[1]+3)-1/12)*8 
                done = 0
        
        self.grid_deque.append(self.grid)
        
        if self.position[0] == (self.grid.shape[0]-1) and self.position[1] == (self.grid.shape[1]-1):
            reward = 100
            done = 1

        if self.steps > 25:
            reward = 0
            done = 1
        self.steps += 1
        return [self.get_state(), reward, done, {}] 
