from setuptools import setup

package_name = 'convolutional_frontier_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='tganderson0@gmail.com',
    description='ROS2 Package for Convolutional Frontier Detection',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'convolutional_frontier_detection = convolutional_frontier_detection.convolutional_frontier_detection:main'
        ],
    },
)
