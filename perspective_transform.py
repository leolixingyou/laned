import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh,mag_thresh


def perspective_transform(img,mode):
    """
    执行透视变换：将倾斜视角拍摄到的道路图像转换成鸟瞰图，即将摄像机的视角转换到和道路平行。
    """
    img_size = (img.shape[1], img.shape[0])
    # 手动提取用于执行透视变换的顶点
    #      # y= 360/close

    if mode == 1:

        src = np.float32(
            [[img.shape[1]*0.196, img.shape[0]],#125.5
             [img.shape[1]*0.793, img.shape[0]],#508
             [img.shape[1]*0.420, img.shape[0]*0.648],#269
             [img.shape[1]*0.578, img.shape[0]*0.648]])#370
        dst = np.float32(
            [[img.shape[1]*0.25, img.shape[0]],
             [img.shape[1]*0.75, img.shape[0]],
             [img.shape[1]*0.25, 0],
             [img.shape[1]*0.75, 0]])
    # y = 360// long
    if mode == 2:

        src = np.float32(
            [[img.shape[1]*0.208, img.shape[0]],#125.5
             [img.shape[1]*0.803, img.shape[0]],#508
             [img.shape[1]*0.475, img.shape[0]*0.578],#269
             [img.shape[1]*0.539, img.shape[0]*0.578]])
        dst = np.float32(
            [[img.shape[1]*0.25, img.shape[0]],
             [img.shape[1]*0.75, img.shape[0]],
             [img.shape[1]*0.25, 0],
             [img.shape[1]*0.75, 0]])
    if mode == 3:


        src = np.float32(
            [[0, img.shape[0]*0.71],#125.5
             [img.shape[1], img.shape[0]*0.71],#508
             [img.shape[1]*0.414, img.shape[0]*0.557],#269
             [img.shape[1]*0.576, img.shape[0]*0.557]])
        dst = np.float32(
            [[img.shape[1]*0.25, img.shape[0]],
             [img.shape[1]*0.75, img.shape[0]],
             [img.shape[1]*0.25, 0],
             [img.shape[1]*0.75, 0]])
  

    # src源图像中待测矩形的四点坐标
    # dst目标图像中矩形的四点坐标
    # cv2.getPerspectiveTransform() 计算透视变换矩阵
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    # cv2.warpPerspective()进行透视变换
    # 参数：输入图像、输出图像、目标图像大小、cv2.INTER_LINEAR插值方法
    warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # 调试

    return warped, unwarped, m, m_inv


# if __name__ == '__main__':
#     img_file = 'test_images/004.png'

#     with open('calibrate_camera.p', 'rb') as f:
#         save_dict = pickle.load(f)
#     mtx = save_dict['mtx']
#     dist = save_dict['dist']

#     img = mpimg.imread(img_file)
#     # img = cv2.undistort(img, mtx, dist, None, mtx)

#     img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(img)
#     # img = mag_thresh(img)

#     warped, unwarped, m, m_inv = perspective_transform(img,mode =1 )
  


#     plt.imshow(img, cmap='gray', vmin=0, vmax=1)
#     plt.show()

#     plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
#     plt.show()

#     plt.imshow
# (unwarped, cmap='gray', vmin=0, vmax=1)
#     plt.show()


#video
if __name__ == '__main__':
    

    img_file = 'a.mp4'
    input_file = cv2.VideoCapture(img_file)
    

    while True :

        ret, frame = input_file.read()
        frame = cv2.resize(frame,(1280,720))
        frame = cv2.pyrDown(frame)
        # frame = cv2.pyrDown(frame)

        img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(frame)
        binary_warped, binary_unwarped, m, m_inv = perspective_transform(img,mode =3)
        binary_warped2, binary2_unwarped2, m, m_inv = perspective_transform(frame,mode =3)
        
        cv2.imshow('frame',binary_warped2)
        cv2.imshow('img',hls_bin)
        cv2.imshow('result', binary_warped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
