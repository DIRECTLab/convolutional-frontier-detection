import numpy as np
import pandas as pd
from frontier_algorithms.convolutional import ConvolutionalFrontierDetector
from frontier_algorithms.simple_search import SimpleFrontierDetector
from occupancy_grid import OccupancyGrid
from ConcaveHull import ConcaveHull
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


if __name__ == '__main__':
    ground_truth_detector = SimpleFrontierDetector()

    frontier_detectors = [ConvolutionalFrontierDetector(8)]
    results = []
    world_files = [("star", np.array([[100, 100], [70, 500], [275, 180]]))]
    
    for world_file, robot_positions in world_files:
        occupancy_grid = OccupancyGrid(f"maps/{world_file}.tmj", True)
        frontiers_ground_truth = ground_truth_detector.identify_frontiers(occupancy_grid.as_flat(),
                                                                          occupancy_grid.width,
                                                                          occupancy_grid.height,
                                                                          robot_positions,
                                                                          False)
    
        ch_ground = ConcaveHull()
        ch_ground.loadpoints(frontiers_ground_truth)
        ch_ground.calculatehull()
        boundary_points_ground_truth = np.vstack(ch_ground.boundary.exterior.coords.xy).T
        
        for frontier_detector in frontier_detectors:
            print(f"{frontier_detector.algorithm_name}...")
            frontiers_test = frontier_detector.identify_frontiers(occupancy_grid.as_flat(),
                                                                  occupancy_grid.width,
                                                                  occupancy_grid.height,
                                                                  robot_positions)
            if len(frontiers_test) < 3:
                print(f"Failed, only found {len(frontiers_test)} frontiers")
                exit()

            ch_test = ConcaveHull()
            ch_test.loadpoints(frontiers_test)
            ch_test.calculatehull()
            boundary_points_test = np.vstack(ch_test.boundary.exterior.xy).T
            distance, _ = fastdtw(boundary_points_ground_truth, boundary_points_test, dist=euclidean)

            results.append({ 'world file': world_file, 'detector': frontier_detector.algorithm_name, 'distance': distance / len(frontiers_ground_truth) })
            occupancy_grid.show_final_map()
    result_df = pd.DataFrame(results)
    result_df.to_csv(f'results/accuracy.csv', index=False)