import numpy as np
from sklearn.cluster import DBSCAN
import yaml
import random
from collections import deque

with open('simple-sim/config.yml', 'r') as file:
    config = yaml.safe_load(file)

random.seed(7)

class ExpandingWavefront:
    def __init__(self):
        self.invalid_frontiers = set()
        self.visited_frontiers = set()
        self.previous_frontiers = dict()

    def mark_frontier_invalid(self, invalid_frontier):
        """
        Marks a frontier invalid, useful for ones that have been found to be in a wall, or impossible to get to
        """
        self.invalid_frontiers.add(f"{invalid_frontier[0]},{invalid_frontier[1]}")

    def __call__(self, occupancy_grid, width, height, robot_positions):
        return self.identify_frontiers(occupancy_grid, width, height, robot_positions)

    def __mark_unvisited(self, O_t):
        """
        Marks the cells in O_t as unvisited
        [input]
        O_t: Using notation from paper, the set of frontiers from the last run, as well as the the new area covered
        """
        for frontier in O_t.keys():
            if frontier in self.visited_frontiers:
                self.visited_frontiers.remove(frontier)

    def __is_frontier(self, frontier_map, x, y):
        if frontier_map[y, x] != 0:
            return False
        for y_offset in range(-1, 2):
            for x_offset in range(-1, 2):
                if not y_offset == x_offset == 0 and 0 <= y + y_offset < frontier_map.shape[0] and 0 <= x + x_offset < frontier_map.shape[1] and frontier_map[y + y_offset, x + x_offset] == -1:
                    return True
        return False

    def __bresenham_circle(self, robot_positions):
        """
        Generates the points that form a circle around the given position
        """
        circle_radius = config["view_distance_radius"] - 1

        x = 0
        y = circle_radius
        d = 3 - (2 * circle_radius)

        results = []

        def eight_way_symmetric(x, y):
            for robot_position in robot_positions:
                results.append([x + robot_position[0], y + robot_position[1]])
                results.append([x + robot_position[0], -y + robot_position[1]])
                results.append([-x + robot_position[0], y + robot_position[1]])
                results.append([-x + robot_position[0], -y + robot_position[1]])
                results.append([y + robot_position[0], x + robot_position[1]])
                results.append([y + robot_position[0], -x + robot_position[1]])
                results.append([-y + robot_position[0], x + robot_position[1]])
                results.append([-y + robot_position[0], -x + robot_position[1]])

        eight_way_symmetric(x, y)

        while x <= y:
            if d < 0:
                d += (4 * x) + 6
            else:
                d += (4 * x) - (4 * y) + 10
                y -= 1
            x += 1

            eight_way_symmetric(x, y)

        return np.array(results)

    def identify_frontiers(self, occupancy_grid, width, height, robot_positions):
        
        frontier_map = np.reshape(occupancy_grid, (height, width))

        frontiers = []

        potential_new_areas = self.__bresenham_circle(robot_positions)

        O_t = dict()

        for area in potential_new_areas:
            label = f"{area[0]},{area[1]}"
            O_t[label] = area

        if len(self.previous_frontiers) == 0:
            for robot_pos in robot_positions:
                label = f"{robot_pos[0]},{robot_pos[1]}"
                O_t[label] = robot_pos

        O_t.update(self.previous_frontiers)
        self.__mark_unvisited(O_t)

        queue = deque()

        queue.extend(O_t.values())

        self.previous_frontiers.clear()

        while queue:
            current_position = queue.pop()
            label = f"{current_position[0]},{current_position[1]}"
            self.visited_frontiers.add(label)
            if 0 <= current_position[0] < frontier_map.shape[0] and 0 <= current_position[1] < frontier_map.shape[1] and (frontier_map[current_position[1], current_position[0]] == 0): # if free space
                if self.__is_frontier(frontier_map, current_position[0], current_position[1]):
                    self.previous_frontiers[label] = current_position
                    frontiers.append(np.array([current_position[1], current_position[0]]))
                
                for y_off in range(-1, 2):
                    for x_off in range(-1, 2):
                        new_pos = current_position + np.array([y_off, x_off])
                        new_label = f"{new_pos[0]},{new_pos[1]}"
                        if not new_label in self.visited_frontiers:
                            queue.append(new_pos)

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
                temp_frontiers.append(polar_frontiers[len(polar_frontiers) * i // number_frontiers]['frontier'])

        frontiers = []
        for frontier in temp_frontiers:
            if not f"{int(frontier[0])},{int(frontier[1])}" in self.invalid_frontiers:
                frontiers.append(frontier)
        # for frontier_section in temp_frontiers:
        #     for frontier in frontier_section:
        #         if not f"{int(frontier[0])},{int(frontier[1])}" in self.invalid_frontiers:
        #             frontiers.append(frontier)

        frontiers = np.array(frontiers, dtype=np.int32)

        return frontiers