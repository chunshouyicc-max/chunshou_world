import cv2
# 读取图片
# 定义原图片链接
path = '/home/chunshouy/桌面/ma.png'
image = cv2.imread(path)

# 转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 保存灰度图
# 定义灰度图链接
gray_path="/home/chunshow/桌面/ma_gray.jpg"
cv2.imwrite(gray_path, gray_image)

# 显示图片
cv2.imshow("madewhaocai", image)
cv2.imshow("madewohaocai_huidu", gray_image)

#定义图片保留时间为无限，按任意键退出
cv2.waitKey(0)
