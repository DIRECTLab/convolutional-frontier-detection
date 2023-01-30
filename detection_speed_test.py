from occupancy_grid import OccupancyGrid
from frontier_algorithms.simple_search import SimpleFrontierDetector
from frontier_algorithms.ours import FrontierDetector
from frontier_algorithms.naiveAA import NaiveActiveArea
from frontier_algorithms.expanding_wavefront import ExpandingWavefront
import numpy as np
import time

def valid_point(frontier, occupancy_grid):
    return 0 <= frontier[0] < occupancy_grid.shape[0] and 0 <= frontier[1] < occupancy_grid.shape[1]

if __name__ == '__main__':
    occupancy_grid = OccupancyGrid('simple-sim/maps/star.tmj', True)
    # occupancy_grid.show_final_map()

    simple_detection = SimpleFrontierDetector()
    our_detection = FrontierDetector()
    naive_aa = NaiveActiveArea()
    ewfd = ExpandingWavefront()

    current_frontier = np.zeros((occupancy_grid.height, occupancy_grid.width))
    # frontiers = ewfd.identify_frontiers(occupancy_grid.as_flat(), occupancy_grid.width, occupancy_grid.height, np.array([[500, 500]]))


    start = time.time()
    frontiers = simple_detection.identify_frontiers(occupancy_grid.as_flat(), occupancy_grid.width, occupancy_grid.height, np.array([[500, 500]]))
    # frontiers = our_detection.identify_frontiers(occupancy_grid.as_flat(), occupancy_grid.width, occupancy_grid.height, np.array([[500, 500]]))
    # frontiers = naive_aa.identify_frontiers(occupancy_grid.as_flat(), occupancy_grid.width, occupancy_grid.height, np.array([[500, 500]]))
    # frontiers = ewfd.identify_frontiers(occupancy_grid.as_flat(), occupancy_grid.width, occupancy_grid.height, np.array([[500, 500]]))
    elapsed = time.time() - start

    print(f"Took {elapsed} seconds")

    current_frontier.fill(0)
    for frontier in frontiers:
        for y_off in range(-2, 3):
            for x_off in range(-2, 3):
                current_point = frontier + np.array([y_off, x_off])
                if valid_point(current_point, current_frontier):
                    current_frontier[current_point[0], current_point[1]] = 1

    occupancy_grid.disable_interactive()
    occupancy_grid.show_world_map(current_frontier=current_frontier)

