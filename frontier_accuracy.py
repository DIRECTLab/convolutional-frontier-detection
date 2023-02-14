import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from collections import deque


"""
Parts taken from https://plotly.com/python/v3/alpha-shapes/#generating-alpha-shape-of-a-set--of-2d-points
"""

def sq_norm(v):
    return np.linalg.norm(v) ** 2

def circumcircle(points, simplex):
    A = [points[simplex[k]] for k in range(3)]
    M = [[1.0] * 4]
    M += [[sq_norm(A[k]), A[k][0], A[k][1], 1.0] for k in range(3)]
    M = np.asarray(M, dtype=np.float32)
    S = np.array([0.5 * np.linalg.det(M[1:, [0, 2, 3]]), -0.5 * np.linalg.det(M[1:, [0, 1, 3]])])
    a = np.linalg.det(M[1:, 1:])
    b = np.linalg.det(M[1:, [0, 1, 2]])
    return S/a, np.sqrt(b / a + sq_norm(S) / a ** 2) # center, radius

def fill_triangle(occupancy_map, x1, y1, x2, y2, x3, y3):
    def get_area(x, y):
        """
        Gets the area of a polygon given numpy arrays of x and y points
        """
        s1 = np.sum(x * np.roll(y, -1))
        s2 = np.sum(y * np.roll(x, -1))
        area = .5 * np.absolute(s1 - s2)
        return area
    def in_triangle(point_x, point_y, x1, y1, x2, y2, x3, y3):
        total_area = get_area(np.array([x1, x2, x3]), np.array([y1, y2, y3]))
        a1 = get_area(np.array([point_x, x2, x3]), np.array([point_y, y2, y3]))
        a2 = get_area(np.array([x1, point_x, x3]), np.array([y1, point_y, y3]))
        a3 = get_area(np.array([x1, x2, point_x]), np.array([y1, y2, point_y]))

        return total_area == a1 + a2 + a3

    start_x = min([x1, x2, x3])
    start_y = min([y1, y2, y3])
    end_x = max([x1, x2, x3])
    end_y = max([y1, y2, y3])

    for x in range(start_x, end_x+1):
        for y in range(start_y, end_y+1):
            if in_triangle(x, y, x1, y1, x2, y2, x3, y3):
                occupancy_map[y, x] = 1

def flood_fill(occupancy_map, start_x, start_y):
    queue = deque()
    queue.append((start_x, start_y))

    while queue:
        x, y = queue.popleft()
        occupancy_map[y, x] = 1

        if x - 1 >= 0 and occupancy_map[y, x-1] == 0:
            queue.append((x-1, y))
            occupancy_map[y, x-1] = 1
        if x + 1 < occupancy_map.shape[1] and occupancy_map[y, x+1] == 0:
            queue.append((x+1, y))
            occupancy_map[y, x+1] = 1
        if y - 1 >= 0 and occupancy_map[y-1, x] == 0:
            queue.append((x, y-1))
            occupancy_map[y-1, x] = 1
        if y + 1 < occupancy_map.shape[1] and occupancy_map[y+1, x] == 0:
            queue.append((x, y+1))
            occupancy_map[y+1, x] = 1

def get_alpha_complex(alpha, points, simplexes):
    """
    alpha is the parameter for the alpha shape
    points are given in data points
    simplexes is the list of indices in the array of points that define 2-simplexes in the Delaunay triangulation
    """
    return filter(lambda simplex: circumcircle(points, simplex)[1] < alpha, simplexes)

def fill_grid(occupancy_map, frontiers, start_x, start_y, alpha=100):
    x, y = get_connection_points(frontiers, alpha=alpha)

    for i in range(1, len(x)):
        if x[i] is None or x[i-1] is None:
            continue
        fill_line(occupancy_map, x[i-1], y[i-1], x[i], y[i])

    triangles_x = [x[i:i+3] for i in range(0, len(x), 5)]
    triangles_y = [y[i:i+3] for i in range(0, len(y), 5)]

    for x_values, y_values in zip(triangles_x, triangles_y):
        fill_triangle(occupancy_map, x_values[0], y_values[0], x_values[1], y_values[1], x_values[2], y_values[2])

    # flood_fill(occupancy_map, start_x, start_y)
    return occupancy_map

def accuracy_metric(ground_truth, approximation, world_size=(1000, 1000)):
    """
    ground_truth: the frontiers of the ground truth (positions)
    approximation: the frontiers of the approximation (positions)
    """
    ground_truth_map = fill_grid(np.zeros(world_size), ground_truth)
    approximation_map = fill_grid(np.zeros(world_size), approximation)
    return np.sum(np.abs(ground_truth_map - approximation_map))

def get_points(points, complex_s):
    """
    points: the given data points
    complex_s: list of indices in the array of points defining 2-simplexes (triangles) in the simplicial complex
    """
    x = []
    y = []
    for s in complex_s:
        x += [points[s[k]][0] for k in [0, 1, 2, 0]] + [None]
        y += [points[s[k]][1] for k in [0, 1, 2, 0]] + [None]

    return x, y

def get_connection_points(frontiers, alpha=10):
    tri = Delaunay(frontiers)
    alpha_complex = get_alpha_complex(alpha, frontiers, tri.simplices)
    x, y = get_points(frontiers, alpha_complex)
    return x, y

def fill_line(occupancy_grid, x1, y1, x2, y2):
    """
    Fills the line between first point and second point
    """
    def handle_pixels(occupancy_grid, x1, y1, x2, y2, dx, dy, decide):
        """
        Handles the actual line drawing, taken from https://www.geeksforgeeks.org/bresenhams-line-generation-algorithm/
        """
        pk = 2 * dy - dx
        if not decide:
            occupancy_grid[y1, x1] = 1
        else:
            occupancy_grid[x1, y1] = 1

        for i in range(dx + 1):
            if x1 < x2:
                x1 += 1
            else:
                x1 -= 1
            
            if pk < 0:
                if not decide:
                    pk += 2 * dy
                    occupancy_grid[y1, x1] = 1
                else:
                    pk += 2 * dy
                    occupancy_grid[x1, y1] = 1
            else:
                if y1 < y2:
                    y1 += 1
                else:
                    y1 -= 1
                if not decide:
                    occupancy_grid[y1, x1] = 1
                else:
                    occupancy_grid[x1, y1] = 1
                pk += 2 * dy - 2 * dx

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx > dy:
        handle_pixels(occupancy_grid, x1, y1, x2, y2, dx, dy, False)
    else:
        handle_pixels(occupancy_grid, y1, x1, y2, x2, dy, dx, True)

if __name__ == '__main__':
    current_map = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])

    frontiers = np.argwhere(current_map)

    tri = Delaunay(frontiers)

    alpha_complex = get_alpha_complex(10, frontiers, tri.simplices)

    x_points, y_points = get_points(frontiers, alpha_complex)

    plt.plot(x_points, y_points, 'bo-')
    plt.show()
