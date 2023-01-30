import numpy as np
import yaml
import matplotlib.pyplot as plt
import json
import numba
from numba import jit
import time

with open('simple-sim/config.yml', 'r') as file:
    config = yaml.safe_load(file)

class OccupancyGrid:
    def __init__(self, world_file, apriori=False):
        self.width = 0
        self.height = 0
        self.__world_map = np.zeros((1, 1), dtype=np.int8)
        self.__occupancy_grid = np.zeros((1, 1), dtype=np.int8)
        self.__load_map(world_file, apriori)
        self.__discovered_tiles = 0

        # warmup the raycast function
        self.__wall_obstruction(0, 0, 1, 1)
        plt.ion()

        self.plot_is_shown = False

    def __load_map(self, world_file, apriori):
        
        with open(world_file, 'r') as file:
            world_data = json.load(file)
        
        self.width = world_data['width']
        self.height = world_data['height']

        self.__world_map = np.zeros((self.height, self.width), dtype=np.int8)
        self.__occupancy_grid = np.zeros_like(self.__world_map, dtype=np.int8)
        self.__occupancy_grid.fill(-1) # mark the whole thing as unknown

        if not apriori:
            for index, value in enumerate(world_data['layers'][0]['data']):
                if value == config['wall_number']:
                    self.__world_map[index // self.width, index % self.width] = 100
        else:
            for index, value in enumerate(world_data['layers'][0]['data']):
                if value == config['wall_number']:
                    self.__world_map[index // self.width, index % self.width] = 100
                elif value == config['explored_wall_number']:
                    self.__world_map[index // self.width, index % self.width] = 100
                    self.__occupancy_grid[index // self.width, index % self.width] = 100
                elif value == config['explored_free_number']:
                    self.__occupancy_grid[index // self.width, index % self.width] = 0

    def reset_occupancy_grid(self):
        self.__occupancy_grid.fill(-1) # reset our 'known' world to be all unknown
        self.__discovered_tiles = 0
    
    def is_in_obstacle(self, x, y):
        return self.__world_map[y, x] == 100

    def __getitem__(self, position):
        return self.__occupancy_grid[position[0], position[1]] # y, x
    
    def enable_interactive(self):
        plt.ion()
    
    def disable_interactive(self):
        plt.ioff()

    def explore(self, y, x):
        """
        Discovers an area around the robot within the configured radius
        """
        for i in range(-config['view_distance_radius'], config['view_distance_radius']):
            for j in range(-config['view_distance_radius'], config['view_distance_radius']):
                if 0 <= y + i < self.height and 0 <= x + j < self.width and config['view_distance_radius'] ** 2 > i ** 2 + j ** 2: # if we're in the view distance radius, and in bounds, with no wall obstruction
                    if self.__occupancy_grid[y + i, x + j] != -1:
                        continue
                    cast = self.__wall_obstruction(x, y, x + j, y + i)
                    if cast is None:
                        if self.__occupancy_grid[y + i, x + j] == -1 and self.__world_map[y + i, x + j] != 100:
                            self.__discovered_tiles += 1
                        self.__occupancy_grid[y + i, x + j] = self.__world_map[y + i, x + j]
                    else:
                        self.__occupancy_grid[cast[1], cast[0]] = 100
    
    def coverage(self):
        """
        Returns the % coverage of the map
        """
        return float(self.__discovered_tiles) / ((self.width * self.height) - np.count_nonzero(self.__world_map == 100))
    
    def as_flat(self):
        """
        Returns the current occupancy grid as a flat representation, for the planners that need it that way
        """
        return self.__occupancy_grid.flatten()
    
    # @jit(nopython=True)
    def __cast_inner(self, x1, y1, x2, y2, backwards=False):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        pk = 2 * dy - dx

        for i in range(0, dx + 1):
            if backwards and 0 <= x1 < self.height and 0 <= y1 < self.width and self.__world_map[x1, y1] == 100:
                return (y1, x1)
            elif not backwards and 0 <= x1 < self.width and 0 <= y1 < self.height and self.__world_map[y1, x1] == 100:
                return (x1, y1)

            if x1 < x2:
                x1 = x1 + 1
            else:
                x1 = x1 - 1
            
            if pk < 0:
                pk = pk + 2 * dy
            
            else:
                if y1 < y2:
                    y1 += 1
                else:
                    y1 -= 1
                
                pk = pk + 2 * dy - 2 * dx
        return None

    # @jit(nopython=True)
    def __wall_obstruction(self, x1, y1, x2, y2):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if dx < dy:
            return self.__cast_inner(y1, x1, y2, x2, backwards=True)
        else:
            return self.__cast_inner(x1, y1, x2, y2)
    
    def show_final_map(self, path_traveled=None):
        plt.ioff()
        fig, self.ax = plt.subplots(1, 3, figsize=(18, 12))
        modified_original_map = (self.__world_map + 1)
        unknown = np.ma.masked_array(modified_original_map, modified_original_map == 0)
        known = np.ma.masked_array(modified_original_map, modified_original_map == 1)
        obstacle = np.ma.masked_array(modified_original_map, modified_original_map == 101)
        img1 = self.ax[0].imshow(unknown, cmap='Blues')
        img2 = self.ax[0].imshow(known, cmap="Greens")
        img3 = self.ax[0].imshow(obstacle, cmap="Greens")
        self.ax[0].set_title("World Map")

        modified_original_map = (self.__occupancy_grid + 1)
        unknown = np.ma.masked_array(modified_original_map, modified_original_map == 0)
        known = np.ma.masked_array(modified_original_map, modified_original_map == 1)
        obstacle = np.ma.masked_array(modified_original_map, modified_original_map == 101)
        img1 = self.ax[1].imshow(unknown, cmap='Blues')
        img2 = self.ax[1].imshow(known, cmap="Greys")
        img3 = self.ax[1].imshow(obstacle, cmap="Greens")
        self.ax[1].set_title("Current Occupancy Grid")

        if path_traveled is not None:
            modified_original_map = (path_traveled)
            unknown = np.ma.masked_array(modified_original_map, modified_original_map == 0)
            known = np.ma.masked_array(modified_original_map, modified_original_map == 1)
            obstacle = np.ma.masked_array(modified_original_map, modified_original_map == 100)
            img1 = self.ax[2].imshow(unknown, cmap='Blues')
            img2 = self.ax[2].imshow(known, cmap="Greys")
            img3 = self.ax[2].imshow(obstacle, cmap="Greens")
            self.ax[2].set_title("Path Travelled")

        plt.show()

    def show_world_map(self, path_traveled=None, current_frontier=None):
        if not self.plot_is_shown:
            fig, self.ax = plt.subplots(2, 2, figsize=(12, 12))
        
        modified_original_map = (self.__world_map + 1)
        unknown = np.ma.masked_array(modified_original_map, modified_original_map == 0)
        known = np.ma.masked_array(modified_original_map, modified_original_map == 1)
        obstacle = np.ma.masked_array(modified_original_map, modified_original_map == 101)
        img1 = self.ax[0][0].imshow(unknown, cmap='Blues')
        img2 = self.ax[0][0].imshow(known, cmap="Greens")
        img3 = self.ax[0][0].imshow(obstacle, cmap="Greens")
        self.ax[0][0].set_title("World Map")

        modified_original_map = (self.__occupancy_grid + 1)
        unknown = np.ma.masked_array(modified_original_map, modified_original_map == 0)
        known = np.ma.masked_array(modified_original_map, modified_original_map == 1)
        obstacle = np.ma.masked_array(modified_original_map, modified_original_map == 101)
        img1 = self.ax[0][1].imshow(unknown, cmap='Blues')
        img2 = self.ax[0][1].imshow(known, cmap="Greys")
        img3 = self.ax[0][1].imshow(obstacle, cmap="Greens")
        self.ax[0][1].set_title("Current Occupancy Grid")

        if path_traveled is not None:
            modified_original_map = (path_traveled)
            unknown = np.ma.masked_array(modified_original_map, modified_original_map == 0)
            known = np.ma.masked_array(modified_original_map, modified_original_map == 1)
            obstacle = np.ma.masked_array(modified_original_map, modified_original_map == 100)
            img1 = self.ax[1][0].imshow(unknown, cmap='Blues')
            img2 = self.ax[1][0].imshow(known, cmap="Greys")
            img3 = self.ax[1][0].imshow(obstacle, cmap="Greens")
            self.ax[1][0].set_title("Path Travelled")

        modified_original_map = (current_frontier)
        unknown = np.ma.masked_array(modified_original_map, modified_original_map == 3)
        known = np.ma.masked_array(modified_original_map, modified_original_map == 1)
        obstacle = np.ma.masked_array(modified_original_map, modified_original_map == 100)
        img1 = self.ax[1][1].imshow(unknown, cmap='Blues')
        img2 = self.ax[1][1].imshow(known, cmap="Greys")
        img3 = self.ax[1][1].imshow(obstacle, cmap="Greens")
        self.ax[1][1].set_title("Current Goal Frontier")

        if not self.plot_is_shown:
            plt.show()
            self.plot_is_shown = True

        plt.draw()
        plt.pause(0.01)