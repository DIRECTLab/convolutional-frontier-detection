import yaml
import numpy as np
from occupancy_grid import OccupancyGrid
from a_star import AStarPlanner
from frontier_algorithms.convolutional import ConvolutionalFrontierDetector
from frontier_algorithms.simple_search import SimpleFrontierDetector
from frontier_algorithms.naiveAA import NaiveActiveArea
from frontier_algorithms.expanding_wavefront import ExpandingWavefront
import time
from cbba.frontier_assignment import frontier_assignment

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

if __name__ == '__main__':
    occupancy_grid = OccupancyGrid(config['maps'][0])

    frontier_detector = ConvolutionalFrontierDetector()
    # frontier_detector = SimpleFrontierDetector()
    # frontier_detector = NaiveActiveArea()
    # frontier_detector = ExpandingWavefront()

    a_star_planner = AStarPlanner(occupancy_grid)

    robot_positions = np.array([[15, occupancy_grid.height - 2], [30, occupancy_grid.height - 2]], dtype=np.int32)
    robot_goals = [[] for _ in range(robot_positions.shape[0])]
    
    steps = 0
    frontier_replan_counter = 0
    a_star_counter = 0
    movement_steps = 0

    path_travelled = np.zeros((occupancy_grid.height, occupancy_grid.width))

    current_frontier = np.zeros((occupancy_grid.height, occupancy_grid.width))

    path_x, path_y = [], []

    frontiers = []

    last_positions = robot_positions

    force_replan = True

    time_spent_detection = 0
    time_spent_assignment = 0
    time_spent_astar = 0

    start_time = time.time()
    total_loop_time = 0
    while occupancy_grid.coverage() < config['required_coverage']:
        current_loop_time = time.time()
        if steps % 100 == 0:
            print(f"Current Coverage: {round(occupancy_grid.coverage() * 100, 2)}%")
            if steps != 0:
                print(f"Average loop time: {total_loop_time / steps}")
        
        for robot in robot_positions:
            occupancy_grid.explore(robot[1], robot[0])

        if force_replan or frontier_replan_counter > 20: # or frontier_replan_counter % config['frontier_replan_steps'] == 0:
            force_replan = False
            frontier_replan_counter = 0
            a_star_counter = 0

            start = time.time()
            frontiers = frontier_detector(occupancy_grid.as_flat(), occupancy_grid.width, occupancy_grid.height, robot_positions)
            stop = time.time() - start
            time_spent_detection += stop
            print(f"Frontier Detection took: {round(stop, 4)} s")
            
            start = time.time()
            assignments = frontier_assignment(robot_positions=robot_positions, frontier_positions=frontiers)
            stop = time.time() - start
            time_spent_assignment += stop
            print(f"Frontier Assignment took: {round(stop, 4)} s")

            for robot_id in range(len(assignments)):
                robot_goals[robot_id].clear()
                for frontier_id in assignments[robot_id]:
                    robot_goals[robot_id].append(frontiers[frontier_id])

        if a_star_counter % config['a_star_replan_steps'] == 0:
            path_x.clear()
            path_y.clear()

            start = time.time()
            for index, robot in enumerate(robot_positions):
                if len(robot_goals[index]) == 0:
                    force_replan = True
                    path_x.append([])
                    path_y.append([])
                else:
                    current_goal = robot_goals[index][0]
                    print(f"Selected {current_goal} as current goal")
                    solved_x, solved_y = a_star_planner.planning(current_goal[0], current_goal[1], robot[0], robot[1], occupancy_grid)
                    path_x.append(solved_x)
                    path_y.append(solved_y)
                    print(f"Takes {len(solved_x)} steps")
                    if len(solved_x) == 0:
                        frontier_detector.mark_frontier_invalid(current_goal)
                        force_replan = True
            end = time.time() - start
            time_spent_astar += end
            print(f"A* took {round(end, 4)} s")
            if config['render']:
                current_frontier.fill(0)
                for robot in robot_positions:
                    current_frontier[robot[1], robot[0]] = 3
                for frontier in frontiers:
                    if frontier[0] < current_frontier.shape[0] and frontier[1] < current_frontier.shape[1]:
                        current_frontier[frontier[0], frontier[1]] = 1
                        current_frontier[frontier[0] - 1, frontier[1]] = 1
                        current_frontier[frontier[0], frontier[1] - 1] = 1
                        current_frontier[frontier[0] - 1, frontier[1] - 1] = 1
                for assigned_frontiers in robot_goals:
                    for xoff in range(-1, 2):
                        for yoff in range(-1, 2):
                            if len(assigned_frontiers) > 0 and 0 <= assigned_frontiers[0][0] + yoff < occupancy_grid.height and 0 <= assigned_frontiers[0][1] + xoff < occupancy_grid.width:
                                current_frontier[assigned_frontiers[0][0], assigned_frontiers[0][1]] = 3
                occupancy_grid.show_world_map(path_travelled, current_frontier)

        a_star_counter += 1
        frontier_replan_counter += 1

        last_positions = robot_positions
        # using A*, make a step towards the current goal, replan if there is an obstacle in our path now
        for i in range(len(robot_positions)):
            if len(path_x[i]) > 0 and len(path_y[i]) > 0:
                x, y = path_x[i].pop(0), path_y[i].pop(0)
                if not occupancy_grid.is_in_obstacle(x, y):
                    movement_steps += 1
                    robot_positions[i, 0] = x
                    robot_positions[i, 1] = y
                path_travelled[y, x] = 100
            steps += 1

        total_loop_time += time.time() - current_loop_time

    end_time = time.time() - start_time
    print(f"Took {movement_steps} steps to explore {occupancy_grid.coverage() * 100}% of the map\n\tTook {end_time} seconds")
    print(f"Time Breakdown\n\tDetection: {round(time_spent_detection, 4)} s\n\tAssignment: {round(time_spent_assignment, 4)} s\n\tA*: {round(time_spent_astar, 4)} s")

    occupancy_grid.show_final_map(path_travelled)