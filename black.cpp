#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main() {
    // 读取图片
    //定义源图片的链接
    string ma_path = "/home/chunshouy/桌面/ma.png";
    Mat ma = imread(ma_path);
      
    // 创建掩码：找到所有白色像素
    Mat white_mask;
    //这个有点看不懂参数设置，只看了几节基础课，大概是色块参数知识，
    //inRange(源图片，颜色范围，颜色范围，输出的掩码）
    //颜色下限（浅色开始）颜色上限（纯白色）
    inRange(ma, Scalar(200, 200, 200), Scalar(255, 255, 255), white_mask);
    
    // 将白色区域变为黑色

    //先把ma.png复制一份，不然等会儿没有修改后原图没有了
    Mat ma_black_background = ma.clone();
    //掩码的设置Scalar(0,0,0)就算特别黑
    ma_black_background.setTo(Scalar(0, 0, 0), white_mask);
    
    // 保存结果
    //imwrite(图片保存的链接，Mat定义值)
    string black_path = "/home/chunshoouy/桌面/black_ma.jpg";
    imwrite(black_path, ma_black_background);//imwrite第二个参数
    
    // 显示结果
    //imshow第二个参数是Mat后面的名字
    // imshow(图片小窗口标题, Mat定义值)
    imshow("妈的我好菜", ma);
    imshow("妈的我好菜（黑色背景）", ma_black_background);
    
    waitKey(0);
    return 0;
}