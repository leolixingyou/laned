import cv2
import time
import copy

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from cv_bridge import CvBridge, CvBridgeError

import numpy as np   
import pickle  
from combined_thresh import combined_thresh,mag_thresh
from perspective_transform import perspective_transform
from Line import Line
from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset,angel




window_size = 10 
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False  
left_curve, right_curve = 0., 0.  
left_lane_inds, right_lane_inds = None, None  


def callback(msg):
    global calibration
    global cur_img
    global get_new_img_msg
    
    np_arr = np.fromstring(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 403))#, interpolation=cv2.INTER_AREA)
    cur_img['img'] = img #calibration.undistort(img)
    cur_img['header'] = msg.header
    get_new_img_msg = True

def get_bbox_arry_msg(bboxes, label, header):
    bbox_array_msg = BoundingBoxArray()
    bbox_array_msg.header = header

    for i in bboxes:
        tl = (i[0], i[1])
        w = i[2] - i[0]
        h = i[3] - i[1]
        id = i[4]
        label = i[-1]

        bbox_msg = BoundingBox()
        bbox_msg.pose.position.x = tl[0]
        bbox_msg.pose.position.y = tl[1]
        bbox_msg.pose.position.z = 0.0
        bbox_msg.dimensions.x = w
        bbox_msg.dimensions.y = h
        bbox_msg.dimensions.z = 0.0
        bbox_msg.value = id
        bbox_msg.label = label

        bbox_array_msg.boxes.append(bbox_msg)   
    
    return bbox_array_msg


def annotate_image(img_in):


    
    global mtx, dist, left_line, right_line, detected
    global left_curve, right_curve, left_lane_inds, right_lane_inds

    img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img_in)
    
    binary_warped, binary_unwarped, m, m_inv = perspective_transform(img,2)


    ret = None

    if not detected:
        try:
            ret = line_fit(binary_warped)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

            left_fit = left_line.add_fit(left_fit)
            right_fit = right_line.add_fit(right_fit)
            left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
            detected = True  
        except:
            ret = None



    else:  
        try:
            left_fit = left_line.get_fit()
            right_fit = right_line.get_fit()
            ret = tune_fit(binary_warped, left_fit, right_fit)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']
        except:
            print('skip')

        if ret is not None:
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']
            left_fit = left_line.add_fit(left_fit)
            right_fit = right_line.add_fit(right_fit)
            left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

        else:
            detected =  False
    try:

        vehicle_offset = calc_vehicle_offset(img_in, left_fit, right_fit)
        theta = angel(img_in, left_fit, right_fit, left_curve, right_curve)
        result= final_viz(img_in, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset,theta)
    except:
        result = img_in
    # return result,binary_warped
    return result











if __name__ == "__main__":
    import sys
    import cv2

    try:

        ############################## Start ##############################
        cur_img = {'img':None, 'header':None}
        get_new_img_msg = False

        rospy.init_node('object_detection')
        #rospy.Subscriber('/gmsl_camera/port_0/cam_0/image_raw/compressed', CompressedImage, callback)
        rospy.Subscriber('/vds_node_localhost_2218/image_raw/compressed', CompressedImage, callback)
        pub_od = rospy.Publisher('/od_result', Image, queue_size=1, latch=True)
        
        pub_bbox = rospy.Publisher("/od_bbox", BoundingBoxArray, queue_size=10)
        bridge = CvBridge()

        ############################## End ##############################

       
        while not rospy.is_shutdown():
            if get_new_img_msg:
                    img = cv2.resize(cur_img['img'], (1280, 720))#im
                    img = cv2.pyrDown(img)
                    result = annotate_image(img)

  
                if pub_od.get_num_connections() > 0:
                    msg = None
                    try:
                        msg = bridge.cv2_to_imgmsg(result, "bgr8")
                        msg.header = cur_img['header']
                    except CvBridgeError as e:
                        print(e)
                    pub_od.publish(msg)

                get_new_img_msg = False

    except rospy.ROSInterruptException:
        rospy.logfatal("{object_detection} is dead.")