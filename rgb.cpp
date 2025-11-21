#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    Mat ma = imread("/home/chunshouy/桌面/ma.png");//读取本地图片
    int a =0;//定义一个a,待会儿用来循环遍历出像素的列
    int b =0;//同理，这是像素块的行
    /*
    ma.rows------获取图片的像素行数
    ma.cols------获取图片的像素列数
    */
   //Vec.3b 每个像素包含3个颜色通道（BGR），所以需要用包含3个值的数据结构来存储。
    // ma.at<Vec3b>(a, b)
    /*外循环：先用ma.rows来得出图片的行数，从行数1开始进入内循环，内循环列数从1跑到最后一列，行数加1,开始遍历第二行的每个像素点的红绿蓝三色像素值
    （1,1),(1,2),(1,3)......（1,ma.cols）
     (2,1),(2,2).....           .......
     (ma.rows,1).....(ma.rows,ma.cols)*/
    for( a=0;a < ma.rows; a++){ //  ma.rows------获取图片的像素行数
       // 内循环：用ma.cols得出列数，从第一列到最后一列
        for(b=0;b < ma.cols; b++){// ma.cols------获取图片的像素列数
            Vec3b pixel = ma.at<Vec3b>(a, b);
            //访问各个点位的像素值并输出
            cout << "(" << a << "," << b<< "):" 
                 << (int)pixel[2] << ","  // R  红色
                 << (int)pixel[1] << ","  // G  绿色
                 << (int)pixel[0] << " "; // B  蓝色
        }
        cout << endl;  // 每行多打印几组数据后再换行，节省行数
    }
    
    return 0;
}