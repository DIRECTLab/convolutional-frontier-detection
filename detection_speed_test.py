from occupancy_grid import OccupancyGrid
from frontier_algorithms.simple_search import SimpleFrontierDetector
from frontier_algorithms.convolutional import ConvolutionalFrontierDetector
from frontier_algorithms.naiveAA import NaiveActiveArea
from frontier_algorithms.expanding_wavefront import ExpandingWavefront
import numpy as np
import time
from tqdm import tqdm
import pandas as pd

def valid_point(frontier, occupancy_grid):
    return 0 <= frontier[0] < occupancy_grid.shape[0] and 0 <= frontier[1] < occupancy_grid.shape[1]

if __name__ == '__main__':
    world_files = [("star.tmj", np.array([[100, 100], [70, 500], [275, 180]])), ("large-field-large-explored.tmj", np.array([[450, 200], [820, 420], [600, 850]])), ("large-field-medium-explored.tmj", np.array([[450, 250], [250, 450], [675, 600]])), ("medium-field-large-explored.tmj", np.array([[80, 100], [400, 100], [270, 425]])), ("medium-field-medium-explored.tmj", np.array([[300, 100], [140, 340], [400, 320]]))]
    test_iterations = 40
    results = []

    for world_file, robot_positions in world_files:
        print(f"Loading world {world_file}...\n")
        occupancy_grid = OccupancyGrid(f'maps/{world_file}', True)

        detection_algorithms = [SimpleFrontierDetector(), ConvolutionalFrontierDetector(8), ConvolutionalFrontierDetector(16), ConvolutionalFrontierDetector(32), ConvolutionalFrontierDetector(64), ConvolutionalFrontierDetector(128), ExpandingWavefront(), NaiveActiveArea()]

        current_frontier = np.zeros((occupancy_grid.height, occupancy_grid.width)) # used for display at the end

        for current_detection_algorithm in detection_algorithms:
            print(f"\n=== {current_detection_algorithm.algorithm_name} ===")
            
            current_result_file = f"results/{world_file}-{current_detection_algorithm.algorithm_name}.csv"
            current_results = []
            # Warm up the run, EWFD and NaiveAA both require prior data, so to be fair in timing, we allow them to have their first run here
            current_detection_algorithm.identify_frontiers(occupancy_grid.as_flat(), occupancy_grid.width, occupancy_grid.height, robot_positions)
            
            timings = []
            for i in tqdm(range(test_iterations)):
                current = time.time()
                frontiers = current_detection_algorithm.identify_frontiers(occupancy_grid.as_flat(), occupancy_grid.width, occupancy_grid.height, robot_positions)
                current_elapsed = time.time() - current
                timings.append(current_elapsed)
                current_results.append({'time': current_elapsed})
            elapsed = sum(timings)

            df = pd.DataFrame(current_results)
            df.to_csv(current_result_file, index=False)
            
            print(f"Took {elapsed} seconds")
            results.append({"algorithm": current_detection_algorithm.algorithm_name, "average time": elapsed / float(test_iterations), "total time": elapsed, "best time": min(timings), "worst time": max(timings), "number iterations": test_iterations, "world file": world_file, })

    df = pd.DataFrame(results)

    df.to_csv(f'results/results.csv', index=False)

    # current_frontier.fill(0)
    # for frontier in frontiers:
    #     for y_off in range(-2, 3):
    #         for x_off in range(-2, 3):
    #             current_point = frontier + np.array([y_off, x_off])
    #             if valid_point(current_point, current_frontier):
    #                 current_frontier[current_point[0], current_point[1]] = 1

    # occupancy_grid.disable_interactive()
    # occupancy_grid.show_world_map(current_frontier=current_frontier)

