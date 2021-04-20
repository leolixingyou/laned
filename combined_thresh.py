# 导入所需模块
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import time



def abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100):
    """
    获取图像、渐变方向和阈值的最小值与最大值
    """
    # 转换为灰度
    # cv2.cvtColor() 颜色空间转换函数

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 使用OpenCV的Sobel() 函数应用x或y渐变
    # 取绝对值
    if orient == 'x':
        # 利用Sobel方法可以进行sobel边缘检测
        # gray表示源图像，即进行边缘检测的图像
        # cv2.CV_64F表示64位浮点数即64 float
        # 第三和第四个参数分别是对x和y方向的导数【即dx,dy】
        # 对于图像来说就是差分，这里1表示对x求偏导【差分】，0表示不对y求导【差分】
        # 对x求导就是检测x方向上是否有边缘
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # 重新缩放回8位整数
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # 创建一个副本并应用阈值
    binary_output = np.zeros_like(scaled_sobel)
    # 这里我使用 >= and <= 阈值，但是 < and > 也可以
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1



    # 返回结果
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
    # 返回给定Sobel内核大小和阈值的梯度大小
    # 转换为灰度
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 取Sobel x 和 y 梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 计算梯度大小
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # 重新缩放到8位
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # 创建二值图像达到阈值为1，否则为0。
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1


    # 返回二值图像
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    返回给定的sobel内核大小和阈值渐变的方向
    """
    # 转换为灰度
    

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 计算x和y梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 取梯度方向的绝对值
    # 应用阈值，并创建一个二值图像结果

    
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    

    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1



    # 返回二值图像
    return binary_output


def hls_thresh(img, thresh=(0.7, 2)):
    """
    利用S通道将RGB转换为HLS，将阈值转换为二值图像。
    """

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h = hls[:, :, 0]
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    binary_output = np.zeros_like(s)
    binary_output[(s > thresh[0]) & (s <= thresh[1]) ] = 1
    # binary_output[(s > thresh[0]) & (s <= thresh[1]) | ((h>0.083) &(h<=0.15)) ] = 1
    # binary_output[(l > 240) | ((h > 0.083) & (h<=0.15))] = 1
    binary_output.astype(bool)
    return binary_output


# def yellow(img):
#     hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     h = hls[:, :, 0]
#     l = hls[:, :, 1]
#     s = hls[:, :, 2]
#     binary_output = np.zeros_like(s)
#     # binary_output[(s > thresh[0]) & (s <= thresh[1]) ] = 1
#     # binary_output[(s > thresh[0]) & (s <= thresh[1]) | ((h>0.083) &(h<=0.15)) ] = 1
#     binary_output[(h >0.083) &(h<=0.15) ] = 1
#     binary_output.astype(bool)
#     return binary_output
def combined_thresh(img):

    abs_bin = abs_sobel_thresh(img, orient='x', thresh_min=50, thresh_max=255)
    mag_bin = mag_thresh(img, sobel_kernel=3, mag_thresh=(50, 255))
    dir_bin = dir_threshold(img, sobel_kernel=5, thresh=(0.8, 1))
    hls_bin = hls_thresh(img, thresh=(0.7, 1))
    combined = np.zeros_like(dir_bin)
    combined[ ((abs_bin==1)|(mag_bin == 1) & (dir_bin == 1)) | (hls_bin == 1)] = 1
   

    # yellow = np.zeros_like(hls_bin)
    # yellow [((mag_bin==1)!=(abs_bin)&(hls_bin))|(hls_bin)] =1
    # cv2.namedWindow('result2t', 0)
    # cv2.imshow('result2', abs_bin)
    # # cv2.namedWindow('result3', 0)

    # # cv2.imshow('result4', dir_bin)
    # # cv2.namedWindow('result5', 0)
    # # cv2.imshow('result5', hls_bin*255)

    return combined, abs_bin, mag_bin, dir_bin, hls_bin 

if __name__ == '__main__':
    # img_file = 'test_images/straight_lines1.jpg'
    img_file = 'test_images/result_screenshot_04.02.2021.png'

    with open('calibrate_camera.p', 'rb') as f:
        save_dict = pickle.load(f)
    mtx = save_dict['mtx']
    dist = save_dict['dist']


    img = mpimg.imread(img_file)
    combined, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)

    # mag = abs_sobel_thresh(mag_bin)


    # img = cv2.undistort(img, mtx, dist, None, mtx)
    # img = cv2.imread(img_file)
    # abs_bin = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)
    # print(hls_bin.shape)
    # print(abs_bin.shape)
    # print(mag_bin.shape)
    # print(dir_bin.shape)
    # print(combined.shape)
    # cv2.imshow('result2', abs_bin)
    # cv2.waitKey(0)
    # cv2.destroyALLWindows()

    plt.subplot(2, 3, 1)
    plt.imshow(abs_bin, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 3, 2)
    plt.imshow(mag_bin, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 3, 3)
    plt.imshow(dir_bin, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 3, 4)
    plt.imshow(hls_bin, cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 3, 5)
    plt.imshow(img,cmap='gray', vmin=0, vmax=1)
    plt.subplot(2, 3, 6)
    plt.imshow(combined, cmap='gray', vmin=0, vmax=1)
    
    plt.tight_layout()
    # plt.imshow(mag_bin)
    
    # plt.imshow(mag)
    plt.show()


    # img = cv2.imread(img_file)
    # hough = hough(img)
    # cv2.imshow('result2', hough)
    # cv2.waitKey(0)

