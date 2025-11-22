//灰度图--二值化--膨胀--描边
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
using namespace cv;

int main() {
    //定义原图片路径
    string ma_path = "/home/chunshouy/桌面/ma.png";
    //读取原图片
    Mat ma = imread(ma_path);

    //转化为灰度图片
    Mat ma_gray;
    cvtColor(ma, ma_gray, COLOR_BGR2GRAY);

    //转化为黑白图
    // threshold(输入图像，输出图像，阈值，最大值，THRESH_BINARY)阈值=128；大于128变白(255)，小于128变黑(0)
    Mat ma_bw;
    // 使用 THRESH_BINARY_INV 让文字变成白色，背景变成黑色
    threshold(ma_gray, ma_bw, 128, 255, THRESH_BINARY_INV);

    //膨胀
    Mat ma_bigger; //膨胀后的变量
    
    // getStructuringElement(): 创建结构元素，
    // MORPH_RECT是矩形结构，Size()规定出来方框大小
    // 使用更大的核尺寸来充分膨胀文字
    Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15));
    
    // dilate(原图，目标图，kernel)：膨胀函数，让白色区域变大，kernel是小矩阵
    dilate(ma_bw, ma_bigger, kernel);

    //找轮廓
    // vector<vector<Point>> contours：创建存储轮廓的向量
    // findContours(定义的膨胀后的变量)：查找轮廓函数
    // ma_bigger：输入的膨胀后图像
    // contours：输出的轮廓集合
    // RETR_EXTERNAL：只检测最外层轮廓
    // CHAIN_APPROX_SIMPLE：简化轮廓点
    // 这块好抽象
    vector<vector<Point>> contours;
    findContours(ma_bigger, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    //画红线
    Mat redoutline_ma; //定义红色边框的变量
    
    //复制原图
    redoutline_ma = ma.clone();
    
    //找到所有的轮廓像素
    //红色的BGR: Scalar(0, 0, 255)
    for (int i = 0; i < contours.size(); i++) {
        drawContours(redoutline_ma, contours, i, Scalar(0, 0, 255), 3);
    } //这里的3是线宽

    //保存图片
    string outLine_path = "/home/chunshouy/桌面/outline_ma.jpg";
    imwrite(outLine_path, redoutline_ma);
    
    //展示图片
    imshow("红色轮廓", redoutline_ma);
    imshow("原图",ma);
    waitKey(0);
    
    return 0;
}