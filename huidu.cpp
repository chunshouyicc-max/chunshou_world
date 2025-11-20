
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

// imread(路径, 模式) - 读取图像，模式可以不写，默认是彩色的
// cvtColor(输入, 输出, 转换类型) - 颜色空间转换
// imwrite(路径, 图像) - 保存图像
// imshow(标题, 图像) - 显示图像

int main() {
    // 读取图片
    string ma_path = "/home/chunshouy/桌面/ma.png";//上传
    Mat image = imread(ma_path);
    
    // 将图片转换为灰度图
    // cvtColor(输入, 输出, 转换类型) - 颜色空间转换
    //前面两个参数都是Mat定义后面的
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);//改成灰色
    
    // 保存灰度图
    //// imwrite(路径, 图像) - 保存图像
    string gray_path = "/home/chunshouy/桌面/ma_gray.jpg";
    imwrite(gray_path, gray_image);
    
    // 显示原图和灰度图
    // imshow(图片小窗口标题, Mat图像) - 显示图像

    imshow("妈的我好菜", image);
    imshow("妈的我好菜（灰度版）", gray_image);
    
    cout << "灰度图已保存到: " << gray_path << endl;
    
    // 等待按键后关闭窗口
    waitKey(0);
    
    return 0;
}