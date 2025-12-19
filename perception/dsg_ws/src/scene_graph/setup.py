from setuptools import setup
import os
from glob import glob

package_name = 'scene_graph'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'prompts'), glob(package_name + '/prompts/*'))
    ],
    install_requires=['numpy==2.0.1','setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='A ROS2 package with multiple dependencies',
    license='License Declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'room_detection = scene_graph.room_detection:main',
            'scene_tree_generation = scene_graph.scene_tree_generation:main',
            'voting_metrics = scene_graph.voting_metrics:main',
        ],
    },
)
