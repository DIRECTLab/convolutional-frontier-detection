import numpy as np
import pandas as pd
from frontier_algorithms.convolutional import ConvolutionalFrontierDetector
from frontier_algorithms.simple_search import SimpleFrontierDetector
from frontier_algorithms.expanding_wavefront import ExpandingWavefront
from frontier_algorithms.naiveAA import NaiveActiveArea
from occupancy_grid import OccupancyGrid
from ConcaveHull import ConcaveHull
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


if __name__ == '__main__':
    world_files = [("star", np.array([[100, 100], [70, 500], [275, 180]])),
                   ("large-field-large-explored", np.array([[160, 250], [100, 780], [610, 890]])),
                   ("large-field-medium-explored", np.array([[460, 200], [220, 380], [710, 675]])),
                   ("medium-field-large-explored", np.array([[80, 70], [420, 90], [50, 430]])),
                   ("medium-field-medium-explored", np.array([[340, 80], [440, 340], [150, 370]]))]

    ground_truth_detector = SimpleFrontierDetector()

    frontier_detectors = [
                          ConvolutionalFrontierDetector(8),
                          ConvolutionalFrontierDetector(16),
                          ConvolutionalFrontierDetector(32),
                          ConvolutionalFrontierDetector(64),
                          ConvolutionalFrontierDetector(128),
                          SimpleFrontierDetector(),
                          ExpandingWavefront(),
                          NaiveActiveArea(),
                          ]
    
    results = []

    for world_file, robot_positions in world_files:

        # Reset expanding wavefront and naive active area
        frontier_detectors[6] = ExpandingWavefront()
        frontier_detectors[7] = NaiveActiveArea()

        print(f"===== Running {world_file} =====")
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
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(f'results/accuracy.csv', index=False)