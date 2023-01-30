# Adapted from Atsushi Sakai(@Atsushi_twi) A Star Planner
import math

class AStarPlanner:
    def __init__(self, occupancy_grid):
        self.width = occupancy_grid.width
        self.height = occupancy_grid.height
        self.motion = self.get_motion_model()
    
    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index
            
    # @jit(nopython=True)
    def planning(self, curr_x, curr_y, goal_x, goal_y, occupancy_grid):
        """
        Performs a path plan
        input:
            curr_x: start x position
            curr_y: start y position
            goal_x: goal x position
            goal_y: goal y position
        output:
            x positions, y positions
        """
        start_node = self.Node(curr_x, curr_y, 0.0, -1)
        goal_node = self.Node(goal_x, goal_y, 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                break

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))

            current = open_set[c_id]
            if (current.x == goal_node.x and current.y == goal_node.y): # or math.hypot(current.x - goal_node.x, current.y - goal_node.y) < 3: # instead we should check distance, so it can handle going
                goal_node.parent_index = current.parent_index
                # goal_node.x = current.x
                # goal_node.y = current.y
                goal_node.cost = current.cost
                break
            
            del open_set[c_id]

            closed_set[c_id] = current

            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)

                n_id = self.calc_grid_index(node)

                if not self.verify_node(node, occupancy_grid):
                    continue

                if n_id in closed_set:
                    continue
                
                if n_id not in open_set:
                    open_set[n_id] = node # new node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node
        
        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        rx, ry = [goal_node.x], [goal_node.y]
        parent_index = goal_node.parent_index
        if goal_node.parent_index == -1:
            print(f"Failed finding a path ({goal_node.x}, {goal_node.y})")
            return [], []
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(n.x)
            ry.append(n.y)
            parent_index = n.parent_index
        
        return rx, ry

    def verify_node(self, node, occupancy_grid):
        return 0 <= node.x < occupancy_grid.width and 0 <= node.y < occupancy_grid.height and occupancy_grid[node.y, node.x] != 100
    
    def calc_grid_index(self, node):
        return (node.y) * self.width + node.x
    
    # @jit(nopython=True)
    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d
    
    @staticmethod
    def get_motion_model():
        """
        Defines the cost of moving directions
        """
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion