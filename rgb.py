import cv2
import numpy as np

# 读取本地图片
ma = cv2.imread("/home/chunshouy/桌面/ma.png")

# 获取图片的行数和列数
rows, cols = ma.shape[:2]

"""
rows ------ 获取图片的像素行数
cols ------ 获取图片的像素列数
ma[a, b] ------ 访问(a,b)位置的像素值
Python中像素值是BGR顺序：[蓝色, 绿色, 红色]bgr
外循环：先用rows来得出图片的行数，从行数0开始进入内循环
内循环列数从0跑到最后一列，行数加1，开始遍历第二行的每个像素点的红绿蓝三色像素值
(0,0),(0,1),(0,2)......(0,cols-1)
(1,0),(1,1),(1,2)......(1,cols-1)
......
(rows-1,0).....(rows-1,cols-1)
"""
# 外循环：遍历每一行
for a in range(rows):
    # 内循环：遍历每一列
    for b in range(cols):
        # 访问各个点位的像素值
        pixel = ma[a, b]
        # 输出坐标和RGB值（tips;Python是BGR顺序，要反向输出）
        print(f"({a},{b}):{pixel[2]},{pixel[1]},{pixel[0]} ", end="")#设置每组数据留一个空格间隔
    
    print()  # 每行结束后换行