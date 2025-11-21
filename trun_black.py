import cv2
import numpy
#定义原图片路径
ma_path = "/home/chunshouy/桌面/ma.png"
#定义目标图片的存储路径
black_path = "/home/chunshouy/桌面/black_ma.jpg"
# 先读取原图片
ma = cv2.imread(ma_path)
#定义白色像素区间，找到白色的像素块
# numpy.array([200,200,200])白色块最低的BGR值（白雪公主）
# numopy.array([255,255,255])白色块最高BGR值（像某人的内心一样洁白）
white_mask = cv2.inRange(ma, numpy.array([200, 200, 200]), numpy.array([255, 255, 255]))
#复制原来的图片准备修改，python里面是copy,C++就是clone了
ma_black_background = ma.copy()
#让前面筛选的BGR>0的白色块改为0的黑色块
ma_black_background[white_mask > 0] = [0, 0, 0]
#保存处理后的图片
cv2.imwrite(black_path, ma_black_background)
#展示图片
#666,小窗名字我设成汉字就给我展示黑屏图片，隔壁C++咋就不会？
cv2.imshow("madewohaocai", ma)
cv2.imshow("black_version", ma_black_background)
#展示时间为无限
cv2.waitKey(0)