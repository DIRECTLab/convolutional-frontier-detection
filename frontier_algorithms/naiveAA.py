import numpy as np
from sklearn.cluster import DBSCAN
import yaml
import random

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

random.seed(7)

class NaiveActiveArea:
    def __init__(self):
        self.invalid_frontiers = set()
        self.is_frontier = dict()
        self.was_frontier = set()
        self.algorithm_name = "NaiveActiveArea"

    def mark_frontier_invalid(self, invalid_frontier):
        """
        Marks a frontier invalid, useful for ones that have been found to be in a wall, or impossible to get to
        """
        self.invalid_frontiers.add(f"{invalid_frontier[0]},{invalid_frontier[1]}")

    def __get_bounded_area(self, positions):
        return np.array([[position - (config['view_distance_radius'] + 1), position + (config['view_distance_radius'] + 1)] for position in positions])

    def __call__(self, occupancy_grid, width, height, robot_positions):
        return self.identify_frontiers(occupancy_grid, width, height, robot_positions)

    def __is_frontier(self, frontier_map, x, y):
        if frontier_map[y, x] != 0:
            return False
        for y_offset in range(-1, 2):
            for x_offset in range(-1, 2):
                if not y_offset == x_offset == 0 and 0 <= y + y_offset < frontier_map.shape[0] and 0 <= x + x_offset < frontier_map.shape[1] and frontier_map[y + y_offset, x + x_offset] == -1:
                    return True
        return False

    def identify_frontiers(self, occupancy_grid, width, height, robot_positions):
        
        frontier_map = np.reshape(occupancy_grid, (height, width))

        frontiers = []

        areas_to_search = self.__get_bounded_area(robot_positions)

        for area in areas_to_search:
            for x in range(max(area[0,0], 0), min(area[1,0], width-1)):
                for y in range(max(area[1,0], 0), min(area[1,1], height-1)):
                    current = f"{x},{y}"
                    if frontier_map[y,x] == 0 and not current in self.invalid_frontiers:
                        if self.__is_frontier(frontier_map, x, y):
                            self.is_frontier[current] = [y, x]
                        elif current in self.is_frontier:
                            self.was_frontier.add(current)
                            del self.is_frontier[current]

        frontiers = [frontier for frontier in self.is_frontier.values()]

        frontiers = np.array(frontiers)

        if len(frontiers) == 0:
            return frontiers

        dbscan_cluster = DBSCAN(eps=1, min_samples=3)
        cluster_labels = dbscan_cluster.fit_predict(frontiers)
        
        average_frontiers = {}

        labelled_frontiers = {}

        # for each of the the labels find the average?
        for frontier_index, label in enumerate(cluster_labels):
            frontier = frontiers[frontier_index]
            if label not in average_frontiers:
                average_frontiers[label] = { 'count': 0, 'value': np.zeros_like(frontier) }
                labelled_frontiers[label] = []
            average_frontiers[label]['count'] += 1
            average_frontiers[label]['value'] += frontier
            labelled_frontiers[label].append(frontier)

        
        # Order the frontiers by polar coordinates
        temp_frontiers = []
        for label in np.unique(cluster_labels):
            average_frontier = average_frontiers[label]['value'] / average_frontiers[label]['count']
            polar_frontiers = []
            for frontier in labelled_frontiers[label]:
                translated_frontier = frontier - average_frontier
                phi = np.arctan2(translated_frontier[1], translated_frontier[0])
                polar_frontiers.append({'phi': phi, 'frontier': frontier})
            polar_frontiers = sorted(polar_frontiers, key=lambda d: d['phi'])
            temp_frontiers.append(polar_frontiers[len(polar_frontiers)//2]['frontier'])

        frontiers = []
        for frontier in temp_frontiers:
            if not f"{int(frontier[0])},{int(frontier[1])}" in self.invalid_frontiers:
                frontiers.append(frontier)

        frontiers = np.array(frontiers, dtype=np.int32)

        return frontiers