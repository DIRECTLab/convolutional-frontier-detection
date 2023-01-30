import numpy as np

class FrontierDetector:
    def __init__(self):
        self.min_distance_to_goal = 0.5
        self.new_area_weight = 1
        self.proximity_weight = 1

        self.invalid_frontiers = set()

    def mark_frontier_invalid(self, invalid_frontier):
        """
        Marks a frontier invalid, useful for ones that have been found to be in a wall, or impossible to get to
        """
        self.invalid_frontiers.add(f"{invalid_frontier[0]},{invalid_frontier[1]}")

    def __get_normal_value(self, occupancy_probability):
        return np.abs(occupancy_probability * 2 - 99)

    def __get_reward_magnitude(self, original_magnitude, min_magnitude, max_magnitude, frontier_value, min_frontier_value, max_frontier_value):
        # no difference between min and max means they're all the same, so weights don't matter
        if (min_magnitude == max_magnitude):
            return 1
        x = (original_magnitude - min_magnitude) / (max_magnitude - min_magnitude)
        frontier_normalized_value = 1
        if not (min_frontier_value == max_frontier_value):
            frontier_normalized_value = (frontier_value - min_frontier_value) / (max_frontier_value - min_frontier_value)
        return self.proximity_weight * (-np.tanh(4*x - 2) + 1) + frontier_normalized_value * self.new_area_weight

    def __get_magnitude_normal_vector(self, vector):
        magnitude = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
        return (magnitude, vector / magnitude)
    
    def __get_relative_vector(self, relative_to, original_vector):
        return np.array([original_vector[1] - relative_to[1], original_vector[0] - relative_to[0]])

    # Taken from https://github.com/macbuse/macbuse.github.io/blob/master/PROG/convolve.py
    def __convolve2D(self, frontier_map, kernel):
        # Cross Correlation
        kernel = np.flipud(np.fliplr(kernel))

        pad_x = kernel.shape[0] - frontier_map.shape[0] % kernel.shape[0]
        pad_y = kernel.shape[1] - frontier_map.shape[1] % kernel.shape[1]
        frontier_map = np.pad(frontier_map, [(0, pad_x), (0, pad_y)], "edge")
        # Gather Shapes of Kernel + map + Padding
        x_ker, y_ker = kernel.shape
        x_size, y_size = frontier_map.shape[0:2]
        

        # Initialize Output Convolution
        x_out = int(((x_size - x_ker + 2) / kernel.shape[0]) + 1)
        y_out = int(((y_size - y_ker + 2) / kernel.shape[1]) + 1)
        output = np.zeros((x_out, y_out))


        # Iterate through map
        loop_count = 0
        for y in range(round(y_size / y_ker)):
            for x in range(round(x_size / x_ker)):
                output[x, y] = (kernel * frontier_map[x * x_ker: (x + 1) * x_ker, y * y_ker: (y + 1) * y_ker]).sum() / (x_ker * y_ker)
                loop_count +=1
        return output

    def __call__(self, occupancy_grid, width, height, robot_positions):
        return self.identify_frontiers(occupancy_grid, width, height, robot_positions)

    def identify_frontiers(self, occupancy_grid, width, height, robot_positions):
        frontier_map = np.reshape(occupancy_grid, (height, width))

        frontier_map[frontier_map == -1] = 50

        frontier_map = self.__get_normal_value(frontier_map)
        kernel = np.ones((16, 16)) # old was 16 16
        frontier_map = self.__convolve2D(frontier_map, kernel)

        frontiers = np.argwhere((frontier_map > 1))


        temp_frontiers = []
        frontier_values = []
        value = (99 * (0.7 * kernel.shape[0] ** 2) + (0.3 * kernel.shape[0] ** 2)) / (kernel.shape[0] * kernel.shape[1])

        for frontier in frontiers:
            if (frontier_map[frontier[0], frontier[1]]) < value and not f"{frontier[0] * kernel.shape[0] + kernel.shape[0] // 2},{frontier[1] * kernel.shape[0] + kernel.shape[0] // 2}" in self.invalid_frontiers:
                temp_frontiers.append(frontier)
                frontier_values.append(frontier_map[frontier[0], frontier[1]])

        frontiers = np.array(temp_frontiers)
        frontiers *= kernel.shape[0]

        frontiers += kernel.shape[0] // 2

        return frontiers