# -*- coding: utf-8 -*-
"""
基于ResNet的手写数字识别系统 - 超强版
使用ResNet架构和迁移学习技术
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torchvision import datasets, transforms, models
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置参数 - 超强版
# ==========================================

DEFAULT_IMAGE_PATH = "/home/chunshouy/桌面/1.jpg"
MODEL_WEIGHTS_PATH = "resnet_mnist_best.pth"

# 🔥 超强训练参数
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.1  # 高初始学习率
DROPOUT_RATE = 0.5

# 🔥 超强数据增强
ROTATION_RANGE = 45    # 更大旋转
TRANSLATE_RANGE = 0.25 # 更大平移
SCALE_RANGE = 0.4      # 更大缩放
SHEAR_RANGE = 20       # 剪切变换

# 设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 正在使用的设备: {device}")
if device.type == 'cuda':
    print(f"💻 GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"🔧 PyTorch 版本: {torch.__version__}")

# ==========================================
# 2. 自定义ResNet模型（专门为MNIST优化）
# ==========================================

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out

class CustomResNet(nn.Module):
    """自定义ResNet，专门为28x28的MNIST图像优化"""
    def __init__(self, block, layers, num_classes=10, dropout_rate=0.5):
        super(CustomResNet, self).__init__()
        
        # 初始卷积层（适配28x28小图像）
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        # 自适应池化和Dropout
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类器
        self.fc = nn.Linear(256, num_classes)
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 池化和分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

def create_resnet_model(dropout_rate=0.5):
    """创建ResNet-18风格模型"""
    return CustomResNet(ResidualBlock, [2, 2, 2], dropout_rate=dropout_rate)

# ==========================================
# 3. 超强数据增强和加载
# ==========================================

def get_ultra_augmentation():
    """获取超强数据增强"""
    return transforms.Compose([
        # 几何变换
        transforms.RandomAffine(
            degrees=ROTATION_RANGE,
            translate=(TRANSLATE_RANGE, TRANSLATE_RANGE),
            scale=(1-SCALE_RANGE, 1+SCALE_RANGE),
            shear=SHEAR_RANGE
        ),
        # 弹性变换
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        # 颜色变换
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        # 随机遮挡
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        # 标准化
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # 添加高斯噪声
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05)
    ])

def get_data_loaders_ultra():
    """获取超强数据加载器"""
    print("📥 加载MNIST数据集（超强增强版）...")
    
    # 训练集使用超强增强
    train_transform = get_ultra_augmentation()
    
    # 测试集转换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据集
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 分割训练验证集
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    print(f"📊 数据集统计（超强增强）:")
    print(f"  训练集: {len(train_dataset):,} 张图片")
    print(f"  验证集: {len(val_dataset):,} 张图片")
    print(f"  测试集: {len(test_dataset):,} 张图片")
    
    return train_loader, val_loader, test_loader

# ==========================================
# 4. 超强训练策略
# ==========================================

def train_ultra_model():
    """超强训练函数"""
    print("🔥 开始超强训练（ResNet + 超强增强）...")
    start_time = time.time()
    
    # 获取数据
    train_loader, val_loader, test_loader = get_data_loaders_ultra()
    
    # 创建模型
    model = create_resnet_model(dropout_rate=DROPOUT_RATE).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧮 模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 🔥 使用SGD + 大动量 + 权重衰减
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=5e-4,  # 更强的权重衰减
        nesterov=True
    )
    
    # 🔥 OneCycleLR学习率调度（最先进的调度策略）
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # 🔥 使用标签平滑的交叉熵损失
    class LabelSmoothCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1):
            super().__init__()
            self.smoothing = smoothing
            
        def forward(self, pred, target):
            confidence = 1. - self.smoothing
            logprobs = F.log_softmax(pred, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
    
    criterion = LabelSmoothCrossEntropy(smoothing=0.1)
    
    # 训练历史
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_acc': [], 'lr_history': []
    }
    
    # 早停和模型保存
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    best_model_state = None
    
    print(f"🎯 开始超强训练，共{EPOCHS}个epoch...")
    print("=" * 80)
    
    for epoch in range(EPOCHS):
        # ===== 训练阶段 =====
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # 每个batch更新学习率
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1:03d}/{EPOCHS} | "
                      f"Batch {batch_idx:04d}/{len(train_loader):04d} | "
                      f"Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
        
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        
        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # ===== 记录历史 =====
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['lr_history'].append(optimizer.param_groups[0]['lr'])
        
        # ===== 测试阶段（每2个epoch） =====
        if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
            test_acc = evaluate_model(model, test_loader)
            history['test_acc'].append(test_acc)
            test_display = f"测试: {test_acc:.2f}%"
        else:
            test_display = ""
        
        # 打印结果
        print(f"✅ Epoch {epoch+1:03d}/{EPOCHS} 完成")
        print(f"  训练: 损失={avg_train_loss:.4f}, 准确率={train_acc:.2f}%")
        print(f"  验证: 损失={avg_val_loss:.4f}, 准确率={val_acc:.2f}%")
        if test_display:
            print(f"  {test_display}")
        print("-" * 80)
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
            print(f"💾 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"🛑 早停触发，验证准确率连续{patience}个epoch未提升")
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 最终测试
    final_test_acc = evaluate_model(model, test_loader)
    
    # 训练时间
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\n🎯 最终测试准确率: {final_test_acc:.2f}%")
    print(f"🎯 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"⏱️  总训练时间: {training_time:.2f}秒 ({training_time/60:.1f}分钟)")
    print(f"📈 训练轮数: {epoch+1}/{EPOCHS}")
    
    # 绘制详细训练曲线
    plot_ultra_training_curve(history, final_test_acc)
    
    return model, final_test_acc

def evaluate_model(model, data_loader):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total

# ==========================================
# 5. 专家级图片预处理
# ==========================================

def expert_preprocess(image_path):
    """
    专家级图片预处理
    """
    try:
        # 打开图片
        img = Image.open(image_path).convert('L')
        print(f"📄 原始图片: {os.path.basename(image_path)}, 尺寸: {img.size}")
        
        # 保存原始图片
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(img, cmap='gray')
        plt.title('原始图片')
        plt.axis('off')
        
        # 1. 自适应直方图均衡化（提高对比度）
        img_array = np.array(img)
        
        # 使用CLAHE（对比度受限的自适应直方图均衡化）
        try:
            import cv2
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_array = clahe.apply(img_array)
        except:
            # 备用：普通直方图均衡化
            from skimage import exposure
            img_array = exposure.equalize_hist(img_array) * 255
        
        img = Image.fromarray(img_array.astype(np.uint8))
        
        plt.subplot(1, 4, 2)
        plt.imshow(img, cmap='gray')
        plt.title('增强对比度')
        plt.axis('off')
        
        # 2. 自适应二值化
        from skimage.filters import threshold_local
        try:
            block_size = 35
            binary_adaptive = img_array > threshold_local(img_array, block_size, offset=10)
            img = Image.fromarray((binary_adaptive * 255).astype(np.uint8))
        except:
            # 备用：Otsu阈值
            from skimage.filters import threshold_otsu
            thresh = threshold_otsu(img_array)
            binary = img_array > thresh
            img = Image.fromarray((binary * 255).astype(np.uint8))
        
        # 3. 形态学操作（去噪和连接）
        try:
            import cv2
            kernel = np.ones((2,2), np.uint8)
            img_array = np.array(img)
            img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
            img_array = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
            img = Image.fromarray(img_array)
        except:
            pass
        
        # 4. 找到数字区域（带智能边距）
        non_zero = np.where(np.array(img) < 250)
        if len(non_zero[0]) > 0:
            min_y, max_y = np.min(non_zero[0]), np.max(non_zero[0])
            min_x, max_x = np.min(non_zero[1]), np.max(non_zero[1])
            
            # 计算智能边距（基于数字大小）
            height = max_y - min_y
            width = max_x - min_x
            margin_ratio = 0.2  # 20%的边距
            
            margin_y = int(height * margin_ratio)
            margin_x = int(width * margin_ratio)
            
            min_y = max(0, min_y - margin_y)
            max_y = min(img.height, max_y + margin_y)
            min_x = max(0, min_x - margin_x)
            max_x = min(img.width, max_x + margin_x)
            
            img = img.crop((min_x, min_y, max_x, max_y))
        
        plt.subplot(1, 4, 3)
        plt.imshow(img, cmap='gray')
        plt.title('数字区域提取')
        plt.axis('off')
        
        # 5. 调整大小（保持纵横比，填充到28x28）
        width, height = img.size
        target_size = 24  # 先缩放到24，然后填充到28
        
        # 计算缩放比例
        scale = target_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 创建28x28画布
        canvas = Image.new('L', (28, 28), color=0)  # 黑底
        
        # 居中放置
        left = (28 - new_width) // 2
        top = (28 - new_height) // 2
        canvas.paste(img, (left, top))
        
        # 6. 智能颜色反转
        np_canvas = np.array(canvas)
        hist, bins = np.histogram(np_canvas, bins=256, range=(0, 255))
        
        # 判断是否需要反转（基于直方图分析）
        dark_pixels = np.sum(hist[:128])  # 暗像素
        bright_pixels = np.sum(hist[128:])  # 亮像素
        
        if bright_pixels > dark_pixels * 1.5:  # 如果亮像素明显多于暗像素
            canvas = Image.eval(canvas, lambda x: 255 - x)
            print(f"  智能颜色反转（暗像素: {dark_pixels}, 亮像素: {bright_pixels}）")
        
        # 7. 高斯模糊去噪
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.7))
        
        plt.subplot(1, 4, 4)
        plt.imshow(canvas, cmap='gray')
        plt.title('最终预处理')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('expert_preprocess.png', dpi=120, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 专家级预处理完成，结果已保存为 'expert_preprocess.png'")
        
        # 转换为模型输入
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        tensor = transform(canvas).unsqueeze(0)
        return tensor
        
    except Exception as e:
        print(f"❌ 专家级预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# ==========================================
# 6. 绘制超强训练曲线
# ==========================================

def plot_ultra_training_curve(history, test_acc):
    """绘制超强训练曲线"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 损失曲线
        axes[0, 0].plot(history['train_loss'], 'b-', linewidth=2, label='训练损失')
        axes[0, 0].plot(history['val_loss'], 'r-', linewidth=2, label='验证损失')
        axes[0, 0].set_title('训练和验证损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率曲线
        axes[0, 1].plot(history['train_acc'], 'g-', linewidth=2, label='训练准确率')
        axes[0, 1].plot(history['val_acc'], 'orange', linewidth=2, label='验证准确率')
        if history['test_acc']:
            test_x = [2*i for i in range(len(history['test_acc']))]
            axes[0, 1].plot(test_x, history['test_acc'], 'r--', linewidth=2, 
                           marker='o', label='测试准确率')
        axes[0, 1].axhline(y=test_acc, color='purple', linestyle=':', 
                          linewidth=2, label=f'最终测试 ({test_acc:.2f}%)')
        axes[0, 1].set_title('准确率曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率曲线
        axes[1, 0].plot(history['lr_history'], 'purple', linewidth=2)
        axes[1, 0].set_title('学习率变化 (OneCycleLR)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # 准确率分布
        axes[1, 1].hist(history['val_acc'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].axvline(x=test_acc, color='red', linestyle='--', linewidth=2, 
                          label=f'测试准确率: {test_acc:.2f}%')
        axes[1, 1].set_title('验证准确率分布')
        axes[1, 1].set_xlabel('Accuracy (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ultra_training_curve.png', dpi=120, bbox_inches='tight')
        plt.close()
        print("📈 超强训练曲线已保存为 'ultra_training_curve.png'")
        
    except Exception as e:
        print(f"⚠️ 无法绘制训练曲线: {e}")

# ==========================================
# 7. 主程序
# ==========================================

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ResNet手写数字识别系统')
    parser.add_argument('--image', type=str, default=DEFAULT_IMAGE_PATH,
                       help='手写数字图片路径')
    parser.add_argument('--train', action='store_true',
                       help='强制重新训练模型')
    parser.add_argument('--test', action='store_true',
                       help='只测试模型，不识别图片')
    parser.add_argument('--quick', action='store_true',
                       help='快速训练模式（15个epoch）')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 80)
    print("🔥 ResNet手写数字识别系统 - 超强版")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.benchmark = True
    
    # 调整训练参数（快速模式）
    global EPOCHS
    if args.quick:
        EPOCHS = 15
        print("⚡ 快速训练模式: 15个epoch")
    
    # 训练或加载模型
    need_train = args.train or not os.path.exists(MODEL_WEIGHTS_PATH)
    
    if need_train:
        print("📂 开始超强训练...")
        try:
            model, test_acc = train_ultra_model()
            print(f"✅ 超强训练完成，测试准确率: {test_acc:.2f}%")
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print(f"📂 加载预训练模型: {MODEL_WEIGHTS_PATH}")
        model = create_resnet_model().to(device)
        try:
            model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
            model.eval()
            print("✅ 模型加载成功")
            
            # 测试模型性能
            if args.test:
                print("\n🔍 测试MNIST数据集性能...")
                _, _, test_loader = get_data_loaders_ultra()
                test_acc = evaluate_model(model, test_loader)
                print(f"📊 模型测试准确率: {test_acc:.2f}%")
                return
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("⚠️ 将重新训练模型...")
            model, test_acc = train_ultra_model()
    
    # 识别图片
    if os.path.exists(args.image):
        print(f"\n🔍 开始识别: {os.path.basename(args.image)}")
        
        # 预处理
        input_tensor = expert_preprocess(args.image)
        if input_tensor is None:
            return
        
        # 预测
        input_tensor = input_tensor.to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0, predicted_class].item()
            
            # 获取所有概率
            probs = probabilities.squeeze().cpu().numpy()
            sorted_indices = np.argsort(probs)[::-1]
        
        # 显示结果
        print("\n" + "=" * 60)
        print("🎯 ResNet识别结果")
        print("=" * 60)
        print(f"📁 图片: {os.path.basename(args.image)}")
        print(f"🔢 预测数字: {predicted_class}")
        print(f"🏆 置信度: {confidence*100:.1f}%")
        
        if confidence > 0.9:
            print("✅ 状态: 非常可靠")
        elif confidence > 0.7:
            print("✅ 状态: 可靠")
        elif confidence > 0.5:
            print("⚠️  状态: 一般")
        else:
            print("❓ 状态: 不确定")
        
        print("\n📊 概率分布:")
        for i in range(3):  # 显示前3个
            idx = sorted_indices[i]
            prob = probs[idx] * 100
            bar = "█" * int(prob / 4)
            rank = ["🥇", "🥈", "🥉"][i]
            print(f"  {rank} 数字 {idx}: {prob:5.1f}% {bar}")
        
        print("\n🔍 详细概率:")
        for i in range(10):
            prob = probs[i] * 100
            if prob > 1:
                mark = " ←" if i == predicted_class else ""
                print(f"  数字 {i}: {prob:5.1f}%{mark}")
        
        print("=" * 60)
        
    else:
        print(f"❌ 图片不存在: {args.image}")
        print(f"💡 使用方法: python {__file__} --image 你的图片路径")

# ==========================================
# 8. 程序入口
# ==========================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()