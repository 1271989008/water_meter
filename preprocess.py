import cv2
from PIL import Image
import numpy as np

def double_filter(img): #噪点预处理：双边滤波
    img=cv2.bilateralFilter(img,d=0, sigmaColor=100, sigmaSpace=15)
    return img

def gray_world(img):#灰度世界算法——对比度增强处理
    b=img[:,:,0]
    g=img[:,:,1]
    r=img[:,:,2]
    aveb=np.average(b)
    aveg=np.average(g)
    aver=np.average(r)
    ave=(aveb+aveg+aver)/3
    kb=ave/aveb
    kg=ave/aveg
    kr=ave/aver
    b=b*kb
    g=g*kg
    r=r*kr
    img=cv2.merge((b,g,r))
    return img

def contrast_enhance(img):#对比度增强处理
    img=img[:,:,0]
    img=cv2.equalizeHist(img)
    img=cv2.merge((img,img,img))
    return img

def OTSU(img):
    img = img[:,:,0]
    ret,img=cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    img = cv2.merge((img,img,img))
    return img

def canny(img): #canny边缘检测
    canny = cv2.Canny(img,50,150)
    # _, Thr_img = cv2.threshold(canny, 210, 255, cv2.THRESH_BINARY)  # 设定红色通道阈值210（阈值影响梯度运算效果）
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 定义矩形结构元素
    # gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)  # 梯度
    # cv2.imshow("gradient", gradient)
    # cv2.imshow('Canny', canny)
    cv2.imshow("canny", canny)
    contours, hier = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # for c in contours:
    #     # find bounding box coordinates
    #     # 先计算出一个简单的边界狂，也就是一个矩形啦
    #     # 就是将轮廓信息转换为(x,y)坐标，并加上矩形的高度和宽度
    #     x, y, w, h = cv2.boundingRect(c)
    #     # 画出该矩形
    #     # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    #     # find minimum area
    #     # 然后计算包围目标的最小矩形区域
    #     # 这里先计算出最小矩形区域，然后计算区域的顶点，此时顶点坐标是浮点型，但是像素坐标是整数
    #     # 需要将浮点型转换成矩形
    #     rect = cv2.minAreaRect(c)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     # draw contours
    #     # 画出最小矩形
    #     # drawContours()也会修改源图像
    #     # 第二个参数保存轮廓的数组，也就是保存着很多轮廓
    #     # 第三个参数是要绘制的轮廓数组的索引：-1是绘制所有的轮廓，否则只绘制[box]中指定的轮廓
    #     # 颜色和thickness(密度,就是粗细)放在最后两个参数
    #     cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

    # 绘制轮廓
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
    cv2.imshow("contours", img)
    return img

img=cv2.imread("00009.jpg")
# img=double_filter(img)
# img=contrast_enhance(img)
# img=OTSU(img)
img=canny(img)
cv2.imshow("00009",img)
cv2.waitKey(0)

