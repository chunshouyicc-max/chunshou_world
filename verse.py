import cv2
# 定义原图片路径
ma_path="/home/chunshouy/桌面/ma.png"
# 读取原图片并存储在ma里面
ma=cv2.imread(ma_path)

# 获得图像的长和宽
# ma.shape返回图片的维度信息（高度，宽度，通道数RGB），保存在shuju里面
# print(type(shuju))------list
shuju=ma.shape
# 我只需要前两个参数，赋值给height和widen,[:2]是作切片处理，取列表前两个数据
height,widen=shuju[:2]#读了前两个数据

# 找到中心(宽和高坐标的一半)
chang=height//2
kuan=widen//2
center=(kuan,chang)

#创建旋转矩阵（中心点，角度，缩放比例）
rotate_mat=cv2.getRotationMatrix2D(center,45,0.75)

# 开始旋转
# cv2.warpAffine()
# 传参：原图片，旋转矩阵，输出尺寸，调试边框
ma_verse=cv2.warpAffine(ma,rotate_mat,(ma.shape[1], ma.shape[0]),borderValue=(255, 255, 255))

# 保存图片
cv2.imwrite('/home/chunshouy/桌面/ma_verse.jpg',ma_verse)

# 展示图片
cv2.imshow('ma',ma)#原图
cv2.imshow('ma_verse',ma_verse)#旋转图片

cv2.waitKey(0)