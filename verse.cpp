#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

int main() {
    // 定义图片路径
    string ma_path = "/home/chunshouy/桌面/ma.png";//原图片路径
    string ma_verse_path = "/home/chunshouy/桌面/ma_verse.png";//目标图片存放路径
    
    // 读取原图片
    Mat ma = imread(ma_path);

    // 计算旋转中心点（图片中心）
    // 参数：x坐标, y坐标
    Point2f center(ma.cols / 2.0f, ma.rows / 2.0f);
    
    // 创建旋转矩-getRotationMatrix2D()
    // 参数：旋转中心, 旋转角度(正数向左), 缩放比例
    Mat rotate_mat = getRotationMatrix2D(center, 45.0, 0.75);//我缩小了一点，不然放不下超出去了
    
    // 执行图像旋转
    // 参数：原图, 目标图, 变换矩阵, 输出尺寸, 插值方法, 边框类型, 边框颜色
    //Scalar(255, 255, 255)白色
    Mat ma_verse;
    warpAffine(ma, ma_verse, rotate_mat, ma.size(), 
               INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));
    
    // 保存旋转后的图片
    imwrite(ma_verse_path, ma_verse);
   
    //显示图片
    imshow("妈的我好菜",ma);
    imshow("妈的我好菜(旋转45度)",ma_verse);
    //设置图片显示时间
    waitKey(0);
    return 0;
}