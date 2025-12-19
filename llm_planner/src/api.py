import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav2_msgs.msg import BehaviorTreeLog
from std_msgs.msg import Bool
import tf2_ros
from builtin_interfaces.msg import Time
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import time
import threading
import subprocess

class PosePublisher(Node):
    def __init__(self):
        super().__init__('goal_pub')
        self.publisher = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer, self)
        self.current_pose = None
        self.target_pose = None
        self.running = True
        self.thread = threading.Thread(target=self.get_current_pose)
        self.thread.start()
        self.current_pose = None
        self.goal_success = False

    def go_to(self, x, y):
        print("Going to location: ",x,y)
        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = 'map'
        self.target_pose.pose.position.x = x
        self.target_pose.pose.position.y = y
        self.target_pose.pose.position.z = 0.0
        self.target_pose.pose.orientation.w = 1.0  
        command = (
            f"ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped '{{"
            f"header: {{frame_id: \"{self.target_pose.header.frame_id}\"}}, "
            f"pose: {{position: {{x: {self.target_pose.pose.position.x}, "
            f"y: {self.target_pose.pose.position.y}, "
            f"z: {self.target_pose.pose.position.z}}}, "
            f"orientation: {{x: 0.0, y: 0.0, z: 0.0, w: {self.target_pose.pose.orientation.w}}}}}}}'"
        )
        subprocess.run(command, shell=True)
        print("Goal Published")
        
    def get_current_pose(self):
        while self.running:
            try:
                self.current_pose = self.tfBuffer.lookup_transform('map', 'base_link', rclpy.time.Time(), rclpy.time.Duration(seconds=0.1)).transform
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                pass
            time.sleep(0.01)

    
    def rotate_around(self, degrees=60):
        while not self.current_pose:
            continue
        current_pose = self.current_pose
        # print(current_pose)
        print("Current orientation: ",current_pose.rotation.x, current_pose.rotation.y, current_pose.rotation.z, current_pose.rotation.w)
        # Create a rotation object from the current quaternion
        current_orientation = R.from_quat([current_pose.rotation.x, current_pose.rotation.y, current_pose.rotation.z, current_pose.rotation.w])
        # Rotate about the Z-axis by the specified degrees
        rotation = R.from_euler('z', degrees, degrees=True)
        # Compute the new rotation
        new_orientation = current_orientation * rotation
        # Update the pose with the new orientation
        self.target_pose = PoseStamped()
        self.target_pose.header.stamp = Time(sec=0, nanosec=0)
        self.target_pose.header.frame_id = 'map'
        self.target_pose.pose.position.x = current_pose.translation.x
        self.target_pose.pose.position.y = current_pose.translation.y
        self.target_pose.pose.position.z = current_pose.translation.z
        new_quat = new_orientation.as_quat()
        self.target_pose.pose.orientation = Quaternion(x=new_quat[0], y=new_quat[1], z=new_quat[2], w=new_quat[3])
        command = (
            f"ros2 topic pub --once /goal_pose geometry_msgs/PoseStamped '{{"
            f"header: {{frame_id: \"{self.target_pose.header.frame_id}\"}}, "
            f"pose: {{position: {{x: {self.target_pose.pose.position.x}, "
            f"y: {self.target_pose.pose.position.y}, "
            f"z: {self.target_pose.pose.position.z}}}, "
            f"orientation: {{x: {self.target_pose.pose.orientation.x}, "
            f"y: {self.target_pose.pose.orientation.y}, "
            f"z: {self.target_pose.pose.orientation.z}, "
            f"w: {self.target_pose.pose.orientation.w}}}}}}}'"
        )
        subprocess.run(command, shell=True)
        print("Goal Published")

    def is_position_reached(self):
        current_pose = self.current_pose
        if not current_pose:
            return False
        current_position = np.array([current_pose.translation.x, current_pose.translation.y])
        target_position = np.array([self.target_pose.pose.position.x, self.target_pose.pose.position.y])
        position_difference = np.linalg.norm(current_position - target_position)
        print("Error in position: ",position_difference)
        return position_difference < 1.00

    def is_orientation_reached(self):
        current_pose = self.current_pose
        if not current_pose:
            return False
        current_orientation = np.array([current_pose.rotation.x, current_pose.rotation.y, current_pose.rotation.z, current_pose.rotation.w])
        current_theta = R.from_quat(current_orientation).as_euler('zyx',degrees=False)[0]
        target_orientation = np.array([self.target_pose.pose.orientation.x, self.target_pose.pose.orientation.y, self.target_pose.pose.orientation.z, self.target_pose.pose.orientation.w])
        target_theta =  R.from_quat(target_orientation).as_euler('zyx',degrees=False)[0]
        orientation_difference = abs(current_theta - target_theta)
        return orientation_difference < 0.3


    def close(self):
        self.running = False
        self.thread.join()  
        self.destroy_node()

class GoalStatus(Node):
    def __init__(self):
        super().__init__('goal_status')
        self.subscriber = self.create_subscription(BehaviorTreeLog, '/behavior_tree_log', self.check_goal_status, 10)
        self.subscriber
        self.goal_success = False

    def check_goal_status(self,msg):
        i=10
        for log in msg.event_log:
            if log.node_name == "ComputePathToPose":
                i = msg.event_log.index(log)
                print("previous status: ",msg.event_log[i].previous_status)
                print("current status: ",msg.event_log[i].current_status)
        if i==10:
            return
        if msg.event_log[i].previous_status == "SUCCESS" and msg.event_log[i].current_status == "IDLE":
            self.goal_success = True


class ExploreLocally(Node):
    def __init__(self):
        super().__init__('goal_status')
        self.publisher = self.create_publisher(Twist, '/go2/cmd_vel', 10)
        self.goal_success = False

    def explore_locally(self,rot=math.pi/3):
        omega = 0.4
        t = math.ceil(rot/omega)
        vel = Twist()
        vel.angular.z = omega
        start_time = time.time()
        while time.time()-start_time < t:
            self.publisher.publish(vel)
            time.sleep(0.1)


class ExploreGlobally(Node):
    def __init__(self):
        super().__init__('run_explore_globally')
        self.publisher = self.create_publisher(Bool, '/explore/resume', 1)

    def start_explore(self):
        msg = Bool()
        msg.data = True
        self.publisher.publish(msg)
        self.get_logger().info('Start explore globally')
    
    def end_explore(self):
        msg = Bool()
        msg.data = False
        self.publisher.publish(msg)
        self.get_logger().info('Ending explore globally')


class GoalPub():
    def __init__(self):
        rclpy.init()
        self.goto = PosePublisher()
        self.rotate = ExploreLocally()
        self.explore_globally = ExploreGlobally()
        self.executor = rclpy.executors.MultiThreadedExecutor()
        self.executor.add_node(self.goto)
        self.executor.add_node(self.rotate)
        self.executor.add_node(self.explore_globally)

    def execute_goto(self,x=0.0,y=0.0):
        self.goto.go_to(x,y)
        start=time.time()
        while not self.goto.is_position_reached() or not self.goto.is_orientation_reached():
            if time.time()-start > 20:
                key = input('Confirm if nav2 execution is over')
                if key == 'y':
                    break
        print("Goal is reached!")
        return True

    
    def execute_explore_locally(self,theta=math.pi/6,rotations=6):
        for i in range(rotations):
            print(f"rotation {i+1}")
            self.rotate.explore_locally(theta)
            time.sleep(2)
            
        print("Goal is reached!")
        return True

    def execute_explore_globally(self,time_period=60):
        self.explore_globally.start_explore()
        start = time.time()
        while (time.time()-start)<time_period:
            continue
        self.explore_globally.end_explore()
        self.execute_explore_locally(theta=math.pi/2, rotations=4)
        return True

if __name__ == '__main__':
    g = GoalPub()
    while True:
        g.execute_explore_globally(7)
        time.sleep(2)