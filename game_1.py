# import torch
# print(f'PyTorch版本: {torch.__version__}, CUDA可用: {torch.cuda.is_available()}')
# import ultralytics
# print(f'Ultralytics版本: {ultralytics.__version__}')

# test_yolo.py - 快速测试脚本
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# 在COCO8数据集上训练模型你设计的次数
train_results = model.train(
    data="coco8.yaml",  # 数据集配置文件的路径
    epochs=200,  # 设置次数
    imgsz=640,  # 训练图像的尺寸
    device="CUDA",  # 运行设备 (e.g., 'cpu', 0, [0,1,2,3])
)

## 在验证集上评估模型性能
metrics = model.val()

# 对目标图片进行检测
results = model("/home/chunshouy/桌面/baobao.jpg")  # 图片链接
results[0].show()  # 显示结果
# 将模型导出为ONNX格式以便部署
path = model.export(format="onnx")  # 返回导出模型的路径