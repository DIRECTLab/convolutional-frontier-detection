from .CBBA import CBBA_agent
import numpy as np

"""
This code was adapted from https://github.com/keep9oing/consensus-based-bundle-algorithm/
"""

def frontier_assignment(robot_positions, frontier_positions):
    """
    Returns the frontiers for each robot, and the order they should travel them
    [input]
    robot_positions: (n, 2) numpy array of the robot positions, where n is the number of robots
    frontier_positions: (n, 2) numpy array of the frontier positions, where n in the number of frontiers
    [output]
    (n, m) A list of frontiers for each robot, given as a list of lists, with the innermost lists being the indices of the goals
    each robot should go to
    """

    task_num = frontier_positions.shape[0]
    robot_num = robot_positions.shape[0]

    robot_list = [CBBA_agent(robot_positions[i], id=i, vel=1, task_num=task_num, agent_num=robot_num, L_t=task_num) for i in range(robot_num)]

    # Initialize communication network
    G = np.ones((robot_num, robot_num)) # Fully connected network, each robot can talk to each other

    t = 0 # Iteration Number
    max_iterations = 100

    while True:
        converged_list = []

        # Phase 1: Auction Process
        for robot_id, robot in enumerate(robot_list):
            robot.build_bundle(frontier_positions)

        # Communication of phase 1
        message_pool = [robot.send_message() for robot in robot_list]

        for robot_id, robot in enumerate(robot_list):
            # Receive winning bidlist from neighbors
            g = G[robot_id]

            connected, = np.where(g == 1)
            connected = list(connected)
            connected.remove(robot_id)

            if len(connected) > 0:
                Y = { neighbor_id: message_pool[neighbor_id] for neighbor_id in connected }
            else:
                Y = None
            
            robot.receive_message(Y)
        
        # Phase 2: Consensus Process
        for robot_id, robot in enumerate(robot_list):
            if Y is not None:
                converged = robot.update_task()
                converged_list.append(converged)
            
        t += 1

        if sum(converged_list) == robot_num or t > max_iterations:
            break
    
    final_paths = [robot.p for robot in robot_list]

    return final_paths

if __name__ == '__main__':
    frontiers = np.random.uniform(low=0, high=1, size=(20, 2))
    robots = np.random.uniform(low=0, high=1, size=(4, 2))
    paths = frontier_assignment(robots, frontiers)

    for robot_id, path in enumerate(paths):
        print(f"Robot {robot_id}: {path}")
