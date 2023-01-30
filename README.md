# Convolutional Frontier Detection

## Running the code

Two versions of running/testing the algorithms can be found. The first only runs detection, just to compare the speeds of the algorithms to each other. This can be found in [detection_speed_test.py](./detection_speed_test.py).

The second version of running the algorithms can be found in [map_exploration.py](./map_exploration.py). This one runs a full exploration process, from detection, goal assignment, and path planning/traversal. The code tends to run fairly slow due to the A* algorithm though, which is the reasoning for providing the first version.

The file [config.yml](./config.yml) contains a few parameters that can be tuned, specifically for the map exploration version, such as the view distance of the robot, along with steps before replanning and such.

## Creating Maps

To create the maps, we utilize the free [Tiled Map Editor](https://www.mapeditor.org/). The software provides an easy way to produce tilemap based worlds, which works very well for the occupancy-grid based exploration. After exporting to JSON format, you will want to take note of which number represents 'wall' and which represents 'open'. Additionally, we specially mark 'explored-wall' and 'explored-open' for the detection speed test. These values should be updated in [config.yml](./config.yml) after taking note. This ensures the occupancy grid loads in the map correctly.

Here is an example of what the editor looks like: ![An screenshot of the Tiled map editor](./images/tiled-editor.png)