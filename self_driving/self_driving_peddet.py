#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden, ru2saig
import os
import cv2
import math
import rospy
import signal
import threading
import numpy as np
import lane_detect
import hiwonder_sdk.pid as pid
import hiwonder_sdk.misc as misc
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
import geometry_msgs.msg as geo_msg
import hiwonder_sdk.common as common
from hiwonder_interfaces.msg import ObjectsInfo
from hiwonder_servo_msgs.msg import MultiRawIdPosDur
from hiwonder_servo_controllers.bus_servo_control import set_servos

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

class SelfDrivingNode:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)
        self.name = name
        self.pid = pid.PID(0.01, 0.0, 0.0)

        self.normal_speed = 0.15
        self.bounding_boxes = []

        self.colors = common.Colors()
        signal.signal(signal.SIGINT, self.shutdown)
        self.machine_type = os.environ.get('MACHINE_TYPE')
        self.lane_detect = lane_detect.LaneDetector("yellow")
        self.mecanum_pub = rospy.Publisher('/hiwonder_controller/cmd_vel', geo_msg.Twist, queue_size=1)  # 底盘控制
        # self.result_publisher = rospy.Publisher(self.name + '/image_result', Image, queue_size=1)  # 图像处理结果发布
        self.joints_pub = rospy.Publisher('servo_controllers/port_id_1/multi_id_pos_dur', MultiRawIdPosDur, queue_size=1)
        depth_camera = rospy.get_param('/depth_camera/camera_name', 'depth_cam')  # 获取参数
        rospy.Subscriber('/%s/rgb/image_raw' % depth_camera, Image, self.image_callback)  # 摄像头订阅
        rospy.Subscriber('/yolov5/object_detect', ObjectsInfo, self.get_object_callback)
        print(ObjectsInfo)
        if not rospy.get_param('~only_line_follow', False):
            while not rospy.is_shutdown():
                try:
                    if rospy.get_param('/yolov5/init_finish'):
                        break
                except:
                    rospy.sleep(0.1)
            rospy.ServiceProxy('/yolov5/start', Trigger)()
        while not rospy.is_shutdown():
            try:
                if rospy.get_param('/hiwonder_servo_manager/init_finish') and rospy.get_param('/joint_states_publisher/init_finish'):
                    break
            except:
                rospy.sleep(0.1)
        
        set_servos(self.joints_pub, 1000, ((6, 490),))  # 初始姿态
        rospy.sleep(1)
        self.mecanum_pub.publish(geo_msg.Twist())
        self.is_running = True
        self.cur_state = "go"
        self.wait_time = 1
        self.last_detection = rospy.get_time()

        self.run_state_machine()

    def shutdown(self, signum, frame):  # ctrl+c关闭处理
        self.is_running = False
        rospy.loginfo('shutdown')

    def image_callback(self, ros_image):
        rgb_image = np.ndarray(shape=(ros_image.height, ros_image.width, 3), dtype=np.uint8, buffer=ros_image.data) # 原始 RGB 画面
        if len(self.bounding_boxes): # draw the boxes here
            for bb, label in self.bounding_boxes:
                plot_one_box(bb, rgb_image, color=(0, 255, 0), label=label)
        self.image = rgb_image

    def run_state_machine(self):
        while self.is_running:
            if self.image is not None:
                twist = geo_msg.Twist()
              
                if self.cur_state == "go":
                    if rospy.get_time() - self.last_detection > self.wait_time: # wait for a bit, before starting again
                        twist.linear.x = self.normal_speed
                else:
                    twist.linear.x = 0

                cv2.imshow("result", cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.is_running = False

                self.mecanum_pub.publish(twist)
            
            else:
                rospy.sleep(0.01)

    def get_object_callback(self, msg):
        objects_info = msg.objects
        self.bounding_boxes = []
        peoples = list(filter(lambda obj: (obj.score >= 0.8 and obj.class_name == "person"), objects_info)) # filter out only people

        if len(peoples):
            self.cur_state = "stop"
            for peep in peoples:
                self.bounding_boxes.append((peep.box, f"Person: {peep.score:.2f}"))
            self.last_detection = rospy.get_time()
        else:
            self.cur_state = "go"



if __name__ == "__main__":
    SelfDrivingNode('self_driving')
