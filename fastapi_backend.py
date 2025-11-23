from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import cv2
import uvicorn
from fastapi import FastAPI
import numpy as np
import base64
import os

app = FastAPI(title="Image Processing System", version="1.0")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建上传目录
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def numpy_to_base64(image):
    """Convert numpy image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode()

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    contents = await file.read()
    
    with open(file_path, "wb") as f:
        f.write(contents)
    
    return {"message": "Image uploaded successfully", "status": "success", "file_path": file_path}

@app.post("/process/grayscale")
async def process_grayscale(file_path: str = Form(...)):
    """Grayscale processing"""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        # 读取图片
        ma = cv2.imread(file_path)
        if ma is None:
            raise HTTPException(status_code=400, detail="Cannot read image")
        
        # 转换为灰度图
        gray_image = cv2.cvtColor(ma, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        result_data = numpy_to_base64(gray_bgr)
        return {"result": result_data, "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process/black_background")
async def process_black_background(file_path: str = Form(...)):
    """Black background conversion"""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        ma = cv2.imread(file_path)
        if ma is None:
            raise HTTPException(status_code=400, detail="Cannot read image")
        
        white_mask = cv2.inRange(ma, np.array([200, 200, 200]), np.array([255, 255, 255]))
        ma_black_background = ma.copy()
        ma_black_background[white_mask > 0] = [0, 0, 0]
        
        result_data = numpy_to_base64(ma_black_background)
        return {"result": result_data, "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process/red_outline")
async def process_red_outline(file_path: str = Form(...)):
    """Red outline detection"""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        ma = cv2.imread(file_path)
        if ma is None:
            raise HTTPException(status_code=400, detail="Cannot read image")
        
        ma_gray = cv2.cvtColor(ma, cv2.COLOR_BGR2GRAY)
        _, ma_bw = cv2.threshold(ma_gray, 127, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        ma_bigger = cv2.dilate(ma_bw, kernel)

        contours, _ = cv2.findContours(ma_bigger, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        redoutline_ma = ma.copy()
        cv2.drawContours(redoutline_ma, contours, -1, (0, 0, 255), 3)

        result_data = numpy_to_base64(redoutline_ma)
        return {"result": result_data, "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process/rotate")
async def process_rotate(file_path: str = Form(...)):
    """Image rotation"""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        ma = cv2.imread(file_path)
        if ma is None:
            raise HTTPException(status_code=400, detail="Cannot read image")
        
        height, width = ma.shape[:2]
        center = (width // 2, height // 2)
        rotate_mat = cv2.getRotationMatrix2D(center, 45, 0.75)
        ma_verse = cv2.warpAffine(ma, rotate_mat, (width, height), borderValue=(255, 255, 255))

        result_data = numpy_to_base64(ma_verse)
        return {"result": result_data, "status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/process/rgb_output")
async def process_rgb_output(file_path: str = Form(...)):
    """RGB pixel output"""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        ma = cv2.imread(file_path)
        if ma is None:
            raise HTTPException(status_code=400, detail="Cannot read image")
        
        rows, cols = ma.shape[:2]

        rgb_data = []
        for a in range(min(5, rows)):
            row_data = []
            for b in range(min(5, cols)):
                pixel = ma[a, b]
                row_data.append(f"({a},{b}): {pixel[2]}, {pixel[1]}, {pixel[0]}")
            rgb_data.append(row_data)
        
        original_image_data = numpy_to_base64(ma)
        
        return {
            "result": {
                "preview": rgb_data,
                "image": original_image_data,
                "total_pixels": f"{rows} x {cols}"
            },
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/")
async def main():
    """Main page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Processing System</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial; margin: 40px; background: #f0f2f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .content { display: grid; grid-template-columns: 300px 1fr; gap: 20px; }
            .sidebar { background: #f8f9fa; padding: 20px; border-radius: 8px; }
            .main-content { padding: 20px; }
            .card { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            input, button { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px; }
            button { background: #007bff; color: white; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            .operations { display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 15px; }
            .operation-btn { background: #28a745; padding: 12px; }
            .operation-btn:hover { background: #218838; }
            .image-preview { text-align: center; margin: 20px 0; }
            .image-preview img { max-width: 100%; max-height: 300px; border: 1px solid #ddd; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Image Processing System</h1>
                <p>FastAPI Backend + Frontend</p>
            </div>
            
            <div class="content">
                <div class="sidebar">
                    <div class="card">
                        <h3>📁 Upload Image</h3>
                        <input type="file" id="imageUpload" accept="image/*">
                        <button onclick="uploadImage()">Upload Image</button>
                        <div id="uploadStatus"></div>
                    </div>
                    
                    <div class="card">
                        <h3>🛠️ Processing</h3>
                        <div class="operations">
                            <button class="operation-btn" onclick="processImage('grayscale')">Grayscale</button>
                            <button class="operation-btn" onclick="processImage('black_background')">Black Background</button>
                            <button class="operation-btn" onclick="processImage('red_outline')">Red Outline</button>
                            <button class="operation-btn" onclick="processImage('rotate')">Rotate 45°</button>
                            <button class="operation-btn" onclick="processImage('rgb_output')">RGB Output</button>
                        </div>
                    </div>
                </div>
                
                <div class="main-content">
                    <div class="card">
                        <h3>👀 Preview</h3>
                        <div id="status">Please upload an image</div>
                        <div class="image-preview">
                            <div id="originalImage">
                                <p>Original image will appear here</p>
                            </div>
                            <div id="resultImage">
                                <p>Processing result will appear here</p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card" id="rgbData" style="display:none;">
                        <h3>📊 RGB Data</h3>
                        <div id="rgbContent"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let currentImagePath = '';
            
            async function uploadImage() {
                const fileInput = document.getElementById('imageUpload');
                if (!fileInput.files[0]) {
                    alert('Please select an image file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    const response = await fetch('/upload/image', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    if (data.status === 'success') {
                        currentImagePath = data.file_path;
                        document.getElementById('uploadStatus').innerHTML = '<div style="color: green;">✓ Upload successful</div>';
                        document.getElementById('status').textContent = '✓ Image ready for processing';
                        
                        // Show original image
                        showOriginalImage();
                    }
                } catch (error) {
                    alert('Upload failed: ' + error);
                }
            }
            
            async function showOriginalImage() {
                const formData = new FormData();
                formData.append('file_path', currentImagePath);
                
                try {
                    const response = await fetch('/process/grayscale', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    if (data.status === 'success') {
                        document.getElementById('originalImage').innerHTML = 
                            '<h4>Original Image</h4><img src="data:image/jpeg;base64,' + data.result + '">';
                    }
                } catch (error) {
                    console.error('Error showing original image:', error);
                }
            }
            
            async function processImage(operation) {
                if (!currentImagePath) {
                    alert('Please upload an image first');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file_path', currentImagePath);
                
                try {
                    document.getElementById('status').textContent = 'Processing...';
                    
                    const response = await fetch('/process/' + operation, {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        document.getElementById('status').textContent = '✓ Processing completed';
                        
                        if (operation === 'rgb_output') {
                            document.getElementById('rgbData').style.display = 'block';
                            document.getElementById('rgbContent').innerHTML = 
                                '<p><strong>Total pixels:</strong> ' + data.result.total_pixels + '</p>' +
                                '<div style="background: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px;">' +
                                data.result.preview.map(row => row.join(' | ')).join('<br>') +
                                '</div>';
                            
                            document.getElementById('resultImage').innerHTML = 
                                '<h4>RGB Data Displayed</h4>';
                        } else {
                            document.getElementById('rgbData').style.display = 'none';
                            document.getElementById('resultImage').innerHTML = 
                                '<h4>Processed Result</h4><img src="data:image/jpeg;base64,' + data.result + '">';
                        }
                    } else {
                        alert('Processing failed: ' + data.detail);
                    }
                } catch (error) {
                    alert('Processing failed: ' + error);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run("fastapi_backend:app", host="0.0.0.0", port=8000, reload=True)