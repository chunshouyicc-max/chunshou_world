import cv2
import numpy as np
import subprocess
from ultralytics import YOLO

# 文件路径
CHICKEN_IMG = "/home/chunshouy/桌面/头.jpg"
INPUT_VIDEO = "/home/chunshouy/桌面/哥哥.mp4"
OUTPUT_VIDEO = "/home/chunshouy/桌面/哥哥打头.mp4"
TEMP_VIDEO = "/tmp/temp_video.mp4"

def main():
    print("开始处理: 篮球变鸡")
    
    # 1. 加载鸡图片
    print("加载鸡图片...")
    chicken = cv2.imread(CHICKEN_IMG, cv2.IMREAD_UNCHANGED)
    if chicken.shape[2] == 3:
        b, g, r = cv2.split(chicken)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        chicken = cv2.merge([b, g, r, alpha])
    
    # 2. 加载模型
    print("加载YOLO模型...")
    model = YOLO('yolo11n.pt')
    
    # 3. 打开视频
    print("处理视频...")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 4. 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(TEMP_VIDEO, fourcc, fps, (w, h))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测并替换篮球
        results = model(frame, classes=[32], conf=0.2)
        
        for result in results:
            if result.boxes:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # 替换为鸡
                    ball_w = x2 - x1
                    ball_h = y2 - y1
                    
                    if ball_w > 10 and ball_h > 10:
                        chicken_resized = cv2.resize(chicken, (ball_w, ball_h))
                        
                        y1, y2 = max(0, y1), min(h, y2)
                        x1, x2 = max(0, x1), min(w, x2)
                        
                        if chicken_resized.shape[2] == 4:
                            alpha = chicken_resized[:, :, 3] / 255.0
                            for c in range(3):
                                frame[y1:y2, x1:x2, c] = (
                                    frame[y1:y2, x1:x2, c] * (1 - alpha) +
                                    chicken_resized[:, :, c] * alpha
                                )
        
        # 写入帧
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"已处理 {frame_count} 帧")
    
    # 清理
    cap.release()
    out.release()
    print(f"视频处理完成: {frame_count}帧")
    
    # 5. 添加音频
    print("添加音频...")
    try:
        subprocess.run([
            'ffmpeg', '-i', TEMP_VIDEO, '-i', INPUT_VIDEO,
            '-c:v', 'copy', '-c:a', 'aac',
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest', '-y', OUTPUT_VIDEO
        ], check=True)
        
        # 删除临时文件
        import os
        if os.path.exists(TEMP_VIDEO):
            os.remove(TEMP_VIDEO)
        
        print(f"✅ 完成！视频已保存: {OUTPUT_VIDEO}")
        
    except:
        print("音频合并失败，保存无声视频")
        import os
        os.rename(TEMP_VIDEO, OUTPUT_VIDEO)

if __name__ == "__main__":
    main()