# 导入所需模块
import numpy as np  # 数组与矩阵运算
import cv2  # 常用图像处理函数库
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # 绘图库
import pickle  # Python序列化的一个工具，通俗点说就是将Python对象转成一串字节，并将其保存到一个文件中
from combined_thresh import combined_thresh,mag_thresh
from perspective_transform import perspective_transform
from Line import Line
from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset, angel,viz1,viz2,cross_walk
from moviepy.editor import VideoFileClip  # 关于视频编辑的Python库
from moviepy.editor import *
import time





# 全局变量
with open('calibrate_camera.p', 'rb') as f:
    save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']
window_size = 10 # 平滑线条有多少帧
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False  # 标志位：是否检测到线
left_curve, right_curve = 0., 0.  # 左右车道的曲率半径
left_lane_inds, right_lane_inds = None, None  # 用于计算曲率


# MoviePy 视频调用函数
def annotate_image(img_in):
    """
    用车道线标记标注输入图像
    返回带标注的图像
    """

    
    global mtx, dist, left_line, right_line, detected
    global left_curve, right_curve, left_lane_inds, right_lane_inds

    # 不失真、阈值、透视变换
    img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img_in)
    # 进行多项式拟合
    
    binary_warped, binary_unwarped, m, m_inv = perspective_transform(img,2)


    ret = None

    if not detected:
        # 曲线拟合
        start = time.time()
        try:
            ret = line_fit(binary_warped)
            left_fit = ret['left_fit']
            right_fit = ret['right_fit']
            nonzerox = ret['nonzerox']
            nonzeroy = ret['nonzeroy']
            left_lane_inds = ret['left_lane_inds']
            right_lane_inds = ret['right_lane_inds']

            # 求曲线拟合系数的平均值z
            left_fit = left_line.add_fit(left_fit)
            right_fit = right_line.add_fit(right_fit)
            # 计算曲率
            left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
            detected = True  # 始终检测到车道线

        except:
            ret = None



    else:  # 表示 detected == True
        # 快速曲线拟合

        try:
            left_fit = left_line.get_fit()
            right_fit = right_line.get_fit()
            ret = tune_fit(binary_warped, left_fit, right_fit)
            # 只有在当前帧中检测到行时才进行更新
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


    # 在原始图像上执行可视化
        result= final_viz(img_in, left_fit, right_fit, m_inv, left_curve, right_curve, vehicle_offset,theta)
    except:
        result = img_in


    # return result,binary_warped
    return result,binary_warped,binary_unwarped,ret


def annotate_video(input_file, output_file):
    """ 给定输入视频文件，将带标注的视频保存到输出文件 """

    input_file = cv2.VideoCapture(input_file)
    

    while True :
        start = time.time()

        ret, frame = input_file.read()
        frame = cv2.resize(frame,(1280,720))
        frame = cv2.pyrDown(frame)
        # frame = cv2.pyrDown(frame)
        # frame = cv2.pyrDown(frame)


        result,binary_warped,binary_unwarped,ret= annotate_image(frame)

        if not ret == None:
            # per = Roi_boundry_yellow(hls_bin)
            # cv2.namedWindow('per',0)
            # cv2.imshow('per',per)
            v  = viz2(binary_warped,ret)
            cv2.namedWindow('vi',0)
            cv2.imshow('vi',v)
            cv2.namedWindow('warp',0)
            cv2.imshow('warp',binary_warped)            
            cv2.namedWindow('unwarp',0)
            cv2.imshow('unwarp',binary_unwarped)
        else:
            continue
        
        cv2.imshow('result', result)
        end = time.time()

        a = end - start

        print(a)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

  ##  make video can change width 
    # video = VideoFileClip(input_file).fx(vfx.resize,width= 320)
    # # input_file = cv2.pyrDown(input_file)
    # video = VideoFileClip(input_file)
    # annotated_video = video.fl_image(annotate_image)
    # # annotated_video.preview(fps=20,audio=False)


    # annotated_video.write_videofile(output_file, audio=False)

if __name__ == '__main__':
    # 给视频添加标注
    
    annotate_video('a.mp4', '500.mp4')

   


    # 显示带标注的图像以进行调试
    # img_file = 'test_images/test2.jpg'
    # img = mpimg.imread(img_file)
    # result = annotate_image(img)
    # result = annotate_image(img)
    # result = annotate_image(img)
    # plt.imshow(result)
    # plt.show()
