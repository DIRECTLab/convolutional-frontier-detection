import numpy as np
from sklearn.cluster import DBSCAN
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
class SimpleFrontierDetector:
    def __init__(self):
        self.invalid_frontiers = set()
        self.algorithm_name = "Simple"

    def mark_frontier_invalid(self, invalid_frontier):
        """
        Marks a frontier invalid, useful for ones that have been found to be in a wall, or impossible to get to
        """
        self.invalid_frontiers.add(f"{invalid_frontier[0]},{invalid_frontier[1]}")

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

        for y in range(height):
            for x in range(width):
                if self.__is_frontier(frontier_map, x, y):
                    frontiers.append([y, x])


        frontiers = np.array(frontiers)
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
            
            number_frontiers = len(polar_frontiers) // config['max_frontiers_in_section']

            for i in range(1, number_frontiers):
                temp_frontiers.append(polar_frontiers[(len(polar_frontiers) // number_frontiers) * i ])

        frontiers = []
        for frontier in temp_frontiers:
            if not f"{int(frontier['frontier'][0])},{int(frontier['frontier'][1])}" in self.invalid_frontiers:
                frontiers.append(frontier['frontier'])

        frontiers = np.array(frontiers, dtype=np.int32)

        return frontiers