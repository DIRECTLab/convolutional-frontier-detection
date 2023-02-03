import rclpy
from rclpy.node import Node
from nav_msg.msg import OccupancyGrid, MapMetadata
from geometry_msgs.msg import Pose, PoseArray, TransformStamped
import numpy as np

class ConvolutionalFrontierDetector(Node):
    def __init__(self):
        super().__init__("convolutional_frontier_detector")
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.occupancy_callback,
            10
        )

        self.publisher = self.create_publisher(PoseArray, 'frontiers', 10)
        timer_period = 5.0
        self.timer = self.create_timer(timer_period, self.publish_frontiers)

        self.map = None # occupancy grid
        self.info = None # occupancy grid info

        self.invalid_frontiers = set()
        self.window_size = 32
    
    def occupancy_callback(self, occupancy_map):
        self.map = occupancy_map.data
        self.info = occupancy_map.info

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

    def __get_normal_value(self, occupancy_probability):
        return np.abs(occupancy_probability * 2 - 99)

    def detect_frontiers(self):
        frontier_map = np.reshape(self.map, (self.info.height, self.info.width))
        frontier_map[frontier_map == -1] = 50

        frontier_map = self.__get_normal_value(frontier_map)

        kernel = np.ones((self.window_size, self.window_size))

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
    
    def publish_frontiers(self):
        if self.map == None or self.info == None:
            return # no map updates yet
        
        msg = PoseArray()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = '/map'

        frontiers = self.detect_frontiers()

        poses = []

        resolution = round(self.info.resolution, 3)

        for frontier in frontiers:
            pose = Pose()

            m_x = frontier[1]
            m_y = self.info.height - frontier[0]

            x = m_x * resolution + self.info.origin.position.x
            y = m_y * resolution + self.info.origin.position.y
            z = self.info.origin.position.z

            pose.position.x = x
            pose.position.y = y
            pose.position.z = z

            poses.append(pose)
        
        msg.poses = poses

        self.publisher.publish(msg)
        self.get_logger().info(f'Published {len(poses)} frontiers')


def main(args=None):
    rclpy.init(args=args)
    frontier_detector = ConvolutionalFrontierDetector()

    rclpy.spin(frontier_detector)
    frontier_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
