# Python和C++的逻辑如出一辙，可以说是一模一样，没有繁琐的Mat定义
# 小窗imshow的时候不要用中文，其他和C++差不多，就不写那么多注释了
import cv2
import numpy as np
def main():
    # 读取原图片
    ma = cv2.imread('/home/chunshouy/桌面/ma.png')
    # 转化为灰度图片
    ma_gray = cv2.cvtColor(ma, cv2.COLOR_BGR2GRAY)
    # 转化为黑白图
    _, ma_bw = cv2.threshold(ma_gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 膨胀 - 让文字区域变大
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    ma_bigger = cv2.dilate(ma_bw, kernel)

    # 找轮廓
    # contours：找到的轮廓列表每个轮廓是一个点的数组
    # _：层次信息（这里用不到，所以用下划线忽略）
    contours, _ = cv2.findContours(ma_bigger, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 在原图上画红色轮廓
    #先复制一下原图
    redoutline_ma = ma.copy()
    cv2.drawContours(redoutline_ma, contours, -1, (0, 0, 255), 3)

    # 保存图片
    cv2.imwrite('/home/chunshouy/桌面/outline_ma.jpg', redoutline_ma)

    # 显示结果
    cv2.imshow('redoutline', redoutline_ma)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()