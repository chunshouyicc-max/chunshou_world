//面向对象的基础练习
// #include <iostream>
// using namespace std;
// int main(){
//     int score=0;
//     cout<<"请输入一个整数："<<endl;
//     cin>>score;
//     cout<<"你的分数为："<<score<<endl;
//     system("pause");
//     return 0;
// }




// // 
// #include <iostream>
// using namespace std;

// //定义圆周率派的数值
// const double PI = 3.14;

// //定义圆类Circle:
// class Circle
// {
//     //访问权限:
// public:
//     //添加属性(半径)
//     int m_r;
//     //添加方法
//     double calculate_ZC()
//     {
//         return 2 * PI * m_r;
//     }
// };  // tips:这里必须加上分号!

// int main(){
//     //实例化对象
//     Circle yuan_1;
//     yuan_1.m_r = 10;
//     cout << "圆的周长为" << yuan_1.calculate_ZC() << endl;
//     system("pause");
//     return 0;
// }




/*
任务2：
设计一个学生类，属性有姓名和学号
可以给姓名和学号赋值，可以显示学生的姓名和学号
*/
// #include<iostream>
// #include<string>
// using namespace std;
// //创建一个类
// class Student
// {
// public://设置访问权限
//     // 设定类的属性和行为（方法）
//     string m_Name;//姓名
//     // tips:用到string时必须添加头文件#include<string>
//     int m_ID;//学号
//     //定义一个获取学生信息的方法
//     void get_info(){
//         cout<<"学生姓名："<<m_Name<<"ID:"<<m_ID<<endl;
//     }
//         //定义一个函数给实例化对象传姓名,可以代替下面的手动传参
//     void set_name(string name){
//         m_Name=name;
//     }
    
// };
// // 实例化对象
// int main(){
//     Student _01;
//     Student _02;
//     // 给实例化对象传参
//     // _01.m_Name="Chen";
//     _01.set_name("Chen");
//     _01.m_ID=5201314;
//     _02.m_Name="tjy";
//     _02.m_ID=7878278;
//     //调用方法
//     _01.get_info();
//     _02.get_info();
//     system("pause");
//     return 0;
// }
/*
类中的属性和行为统称为“成员”
属性  成员属性 成员变量
行为  成员函数 成员方法
*/




/*
封装的访问权限
公共权限  public      成员内外均可以访问
保护权限  protected   成员内可以访问，外不可以
私有权限  private     成员内可以访问，外不可以访问
*/
// #include<iostream>
// #include<string>
// using namespace std;
// class  Person
// {
// public:
//     //公共权限
//     string m_name;//我的名字
// protected:
//     //保护属性
//     string m_car;//我的车
// private:
//     //私有属性
//     int m_password;//我的密码
// public:
//     void func(){
//     m_name="Chen";
//     m_car="bicycle";
//     m_password=123456;
//     }
// }
// int main(){
//         Person p1;
//         p1.m_name="tjy";
//         p1.m_car="托有他";//报错了，protected不可以在外部访问
//         p1.m_password="54188";//还是报错了,private不可以在外部访问
//         system ("pause");
//         return 0;
// }
/*
报错细则：
cppgame_1.cpp: In function ‘int main()’:
cppgame_1.cpp:130:12: error: ‘std::string Person::m_car’ is protected within this context
  130 |         p1.m_car="托有他";//报错了，protected不可以在外部访问
      |            ^~~~~
cppgame_1.cpp:116:12: note: declared protected here
  116 |     string m_car;//我的车
      |            ^~~~~
cppgame_1.cpp:131:12: error: ‘int Person::m_password’ is private within this context
  131 |         p1.m_password="54188";//还是报错了,private不可以在外部访问
      |            ^~~~~~~~~~
cppgame_1.cpp:119:9: note: declared private here
  119 |     int m_password;//我的密码
      |         ^~~~~~~~~~
cppgame_1.cpp:131:23: error: invalid conversion from ‘const char*’ to ‘int’ [-fpermissive]
  131 |         p1.m_password="54188";//还是报错了,private不可以在外部访问
      |                       ^~~~~~~
      |                       |
      |                       const char*

*/


// struct 与 class 的区别
//默认访问权限的不同
/* struct  公有————public
   class   私有————private
   
(说人话就是在你定义一个类时，不写public,private,protected时候，自动定义了)   
*/



//成员属性设为私有
// #include<iostream>
// #include<string>
// using namespace std;
// class Person//设置一个人类
// {
// public:
//     void set_name(string name)//设置姓名
//     {
//         m_name=name;
//     }
//     //获取姓名
//     string get_name(){
//         return m_name;
//     }
    // 获取年龄
    /*保留疑问，这里在public下面设置的成员get_age()前面为司马不是void而是int，上面那个是string?是跟后面括号内内容的数据类型有关系吗？这和python中调用私有方法的内外均使用def定义函数有一定的差异。
    下面是资料的内容
        函数声明 	               含义	                  调用示例
    void set_name(string)	设置姓名，无返回值	      p.set_name("李四")
    string get_name()	    获取姓名，返回字符串	  string name = p.get_name()
    int get_age()	        获取年龄，返回整数	      int age = p.get_age()
    C++要求显式声明返回类型，这是它与Python在语法上的重要区别之一
    */
    //获取年龄
//     int get_age(){
//         return m_age;
//     }
//     //设置年龄（范围：0-150）
//     void set_age(int age){
//         if(age<0||age>150){
//             cout<<"年龄输入错误！！赋值失败"<<endl;\
//             return;
//         }      
//         m_age=age;
//     }


//     //获取私有属性idol同理
//     string get_idol(){
//         return m_idol;
//     }
//     //如果你想只写，让别人访问不了，就像下面这么写
//     void set_idol(string idol){
//         m_idol=idol;
//     }
// private:
//     string m_name;//姓名  可读可写
//     int m_age=18;//年龄  给定初始值  只读
//     string m_idol="陈奕迅";//偶像 只写 
// };
// int main(){
//     //实例化一个对象
//     Person p1;
//     //设置姓名
//     p1.set_name("Chen");
//     p1.set_age(160);//年龄输入错误！！赋值失败
//     //获取姓名
//     cout<<"姓名："<<p1.get_name()<<endl;//姓名：Chen
//     //设置年龄
//     cout<<"年龄："<<p1.get_age()<<endl;//年龄：18
//     //只读偶像
//     cout<<"偶像："<<p1.get_idol()<<endl;//偶像：陈奕迅
//     // 只写偶像是无法访问的
//     p1.set_idol("徐学坤");
//     // cout<<"偶像："<<p1.get.idol()<<endl;//只写外界无法访问，是错的
    
//     system("pause");
//     return 0;
// }


/*
封装案例，设计立方体类
要求：
    1、立方体类Cube
    2、求出表面积和体积
    3、分别用全局函数和成员函数判断两个立方体是否相等
tips:
    1、将属性m_L(长),m_W(宽),m_H(高),设置为私有属性private

    */
// #include<iostream>
// #include<string>
// using namespace std;
// class Cube
// {
// public:
//     //设置长度
//     void set_l(int length){
//         m_L=length;
//     }
//     //获取长度
//     int get_l(){
//         return m_L;
//     }
//     //设置宽度
//     void set_w(int width){
//         m_W=width;
//     }
//     //获取宽度
//     int get_w(){
//         return m_W;
//     }
//     //设置高度
//     void set_h(int height){
//         m_H=height;
//     }
//     //获取高度
//     int get_h(){
//         return m_H;
//     }
//     //获得体积
//     int get_volume(){
//         return (m_L*m_W*m_H);//爆红了不用管
//     }
//     //获得表面积
//     int get_square(){
//         return 2*(m_H*m_W+m_H*m_L+m_W*m_L);
//     }
//     //利用成员函数判断两个立方体是否相等
//     bool is_Sameclass(Cube &c){//传一个函数立方体，和已知的立方体作比较
//     if (m_L==c.get_l()&& m_W==c.get_w() &&m_H==c.get_h()){
//         return true;
//     }
//     return false;
//     }

// private:
//     int m_L=0;
//     int m_W=0;
//     int m_H=0;

// };
// //利用全局函数判断两个立方体是否相等
// bool is_Same(Cube &c1,Cube&c2){//传两个立方体
//     if (c1.get_l()==c2.get_l()&&c1.get_w()==c2.get_w()&&c1.get_h()==c2.get_h()){
//         return true;
//     }
//     return false;
// }


// int main(){
//     //实例化对象
//     Cube c1;
//     Cube c2;
//     c1.set_l(10);
//     c1.set_w(5);
//     c1.set_h(2);
//     c2.set_l(4);
//     c2.set_w(10);
//     c2.set_h(3);
//     cout<<"体积为："<<c1.get_volume()<<endl;//体积为：100
//     cout<<"表面积为："<<c1.get_square()<<endl;//体积为：100
//     //利用全局函数判断
//     bool result=is_Same(c1,c2);////全局判断下:c1和c2是不同的立方体
//     if(result){
//         cout<<"全局判断下:c1和c2是相同的立方体"<<endl;
//     }
//     else{
//         cout<<"全局判断下:c1和c2是不同的立方体"<<endl;
//     }
//     //利用全局函数判断
//     bool ret=c1.is_Sameclass(c2);////成员判断下:c1和c2是不同的立方体
//      if(ret){
//         cout<<"成员判断下:c1和c2是相同的立方体"<<endl;
//     }
//     else{
//         cout<<"成员判断下:c1和c2是不同的立方体"<<endl;
//     }

//     system("pause");
//     return 0;
// }




/*
构造函数
没有返回值，不用写void
函数名与类名相同
可以有参数
创建对象时，构造函数会自动调用，而且只调用一次
*/
/*析构函数  
进行清理的操作
没有返回值
函数名和类名相同，在名称前加~
不可以有参数，不可以发生重载
对象在销毁前会自动调用析构函数，而且只会调用一次

*/
// #include<iostream>
// #include<string>
// using namespace std;
// class Person
// {
// public:
//     Person(){
//         cout<<"Person 构造函数的调用"<<endl;
//     }
//     ~Person(){
//         cout<<"Person的析构函数调用"<<endl;
//     }

// };

// void test01(){
//     Person p;//在栈上的数据，test01执行完毕后，释放这个对象
// }//Person 构造函数的调用
//  //Person的析构函数调用

// int main(){       //返回：Person 构造函数的调用
//     test01();     //     Person的析构函数调用
//  //你不写的话系统自己给你写一个空实现的构造和析构函数
//     system("pause");
//     return 0;
// }



// //类对象作为类成员
// #include<iostream>
// #include<string>
// using namespace std;
// //手机类
// class Phone
// {
// public:

//     Phone(string pName){//这里用到了构造函数，类名和函数名是一样的
//         m_PName=pName;
//     }
//     string m_PName;//类属性：手机的品牌
// };
// //人类
// class Person
// {
// public:
//     //Phone m_Phone=pName  隐式转换法实例化对象
//     Person(string name,string pName):m_Name(name),m_Phone(pName)//初始化列表语法
//     {
//     }
//     //姓名
//     string m_Name;
//     //手机
//     Phone m_Phone;
// };

// //当其他类对象作为本类成员，构造时先构造类对象，在、再构造自身；析构的顺序与构造相反

// void test01(){
//     Person p("张三","苹果MAX");
//     cout<<p.m_Name<<"拿着"<<p.m_Phone.m_PName<<endl;
// }

// int main(){
//     test01();
//     system("pause");
//     return 0;
// }//输出：张三拿着苹果MAX



/*
    静态成员
1、静态成员变量：共享同一份数据，类内声明，类外初始化
2、静态成员函数：所有对象共享同一份函数，静态函数访问静态变量   
*/
// #include<iostream>
// #include<string>
// using namespace std;
// //静态成员变量
// class Person  
// {
// public:
//     //所有对象共享同一份数据
//     //编译阶段内存
//     //类内声明，类外初始化操作
//     static int m_A;

// private:
// // 静态成员也是有访问权限
//     static int m_B;


// };
// //拉到类的外面，类名::静态属性名称=赋值；
// int Person::m_A=100;
// int Person::m_B=200;//私有权限在类的外面访问不了
// void test01(){
//     Person p1;
//     cout<<p1.m_A<<endl;
//     Person P2;
//     P2.m_A=200;
//     cout<<p1.m_A<<endl;//200,说明m_A数据是共享
// }
// //静态成员变量不属于某个对象上，所有对象都共享同一份数据
// //静态变量访问的两种方式：
// //1、通过对象进行访问
// // Person p1
// // cout<<p1.m_A<<endl;
// //2、通过类名进行访问
// // cout<<Person::m_A<<endl;
// //cout<<Person::m_B<<endl;//会报错！！类外访问不了私有静态成员变量
// int main(){
//     test01();
//     system("pause");
//     return 0;
// }



//静态函数
/*
所有对象共享一个函数
静态成员函数只能访问静态成员变量
*/
// #include<iostream>
// #include<string>
// using namespace std;
// class Person
// {
// public:
//     static void func(){
//         cout<<"静态函数static void func()被调用"<<endl;
//     }
// };
// // 如何访问静态函数
// void test01(){
//     //1、通过对象访问
//     Person p1;
//     p1.func();
//     //2、通过类名访问
//     Person::func();

// }

// int main(){
//     test01();
//     system("pause");
//     return 0;
// }//输出：静态函数static void func()被调用
// //      静态函数static void func()被调用
   



// //继承
// #include<iostream>
// #include<string>
// using namespace std;
// //创立Java类
// class Java
// {
// public:
//     void header(){
//         cout<<"首页，公开课，登录，注册...（公共头部）"<<endl;
//     }
//     void footer(){
//         cout<<"帮助中心、交流合作、站内地图...（公共底部）"<<endl;
//     }
//     void left(){
//         cout<<"Java、Python、C++...（公共分类列表）"<<endl;
//     }
//     void content(){
//         cout<<"Java学科视频"<<endl;
//     }
// };

//     //创建Python类
// class Python{
// public:
//     void header(){
//         cout<<"首页，公开课，登录，注册...（公共头部）"<<endl;
//     }
//     void footer(){
//         cout<<"帮助中心、交流合作、站内地图...（公共底部）"<<endl;
//     }
//     void left(){
//         cout<<"Java、Python、C++...（公共分类列表）"<<endl;
//     }
//     void content(){
//         cout<<"Python学科视频"<<endl;
//     } 
// };

// class Cpp
// {
// public:
//     void header(){
//         cout<<"首页，公开课，登录，注册...（公共头部）"<<endl;
//     }
//     void footer(){
//         cout<<"帮助中心、交流合作、站内地图...（公共底部）"<<endl;
//     }
//     void left(){
//         cout<<"Java、Python、C++...（公共分类列表）"<<endl;
//     }
//     void content(){
//         cout<<"C++学科视频"<<endl;
//     } 
// };
// void test01(){
//     "Java的下载视频如下："<<endl;
//     Java ja;
//     ja.header();
//     ja.footer();
//     ja.left();
//     ja.content();

//     cout<<"--------------------------------------"<<endl;
//      "Python的下载视频如下："<<endl;
//     Python py;
//     py.header();
//     py.footer();
//     py.left();
//     py.content();
    
//     cout<<"--------------------------------------"<<endl;
//      "C++的下载视频如下："<<endl;
//     Cpp cpp;
//     cpp.header();
//     cpp.footer();
//     cpp.left();
//     cpp.content();
// }
// int main(){
//     test01();
//     system("pause");
//     return 0;
// }

//采用继承实现
//好处：减少重复的代码
/* 语法： class 子类 ：继承方式 父类
   子类 也叫做 派生类
   父类 也叫做 基类

*/

// #include<iostream>
// #include<string>
// using namespace std;
// class BasePage
// {
// public:
//     void header(){
//         cout<<"首页，公开课，登录，注册...（公共头部）"<<endl;
//     }
//     void footer(){
//         cout<<"帮助中心、交流合作、站内地图...（公共底部）"<<endl;
//     }
//     void left(){
//         cout<<"Java、Python、C++...（公共分类列表）"<<endl;
//     }
// };

// //Java页面
// class Java :public BasePage
// {
// public:
//     void content(){
//         cout<<"Java学习视频"<<endl;
//     }
// };

// //Python页面
// class Python :public BasePage
// {
// public:
//     void content(){
//         cout<<"Python学习视频"<<endl;
//     }
// };

// //C++页面
// class Cpp :public BasePage
// {
// public:
//     void content(){
//         cout<<"C++学习视频"<<endl;
//     }
// };
// void test01(){
//     cout<<"Java的下载视频如下："<<endl;
//     Java ja;
//     ja.header();
//     ja.footer();
//     ja.left();
//     ja.content();

//     cout<<"--------------------------------------"<<endl;
//     cout<< "Python的下载视频如下："<<endl;
//     Python py;
//     py.header();
//     py.footer();
//     py.left();
//     py.content();
    
//     cout<<"--------------------------------------"<<endl;
//      cout<<"C++的下载视频如下："<<endl;
//     Cpp cpp;
//     cpp.header();
//     cpp.footer();
//     cpp.left();
//     cpp.content();
// }
// int main(){
//     test01();
//     system("pause");
//     return 0;
// }
/*输出：
    Java的下载视频如下：
首页，公开课，登录，注册...（公共头部）
帮助中心、交流合作、站内地图...（公共底部）
Java、Python、C++...（公共分类列表）
Java学习视频
--------------------------------------
Python的下载视频如下：
首页，公开课，登录，注册...（公共头部）
帮助中心、交流合作、站内地图...（公共底部）
Java、Python、C++...（公共分类列表）
Python学习视频
--------------------------------------
C++的下载视频如下：
首页，公开课，登录，注册...（公共头部）
帮助中心、交流合作、站内地图...（公共底部）
Java、Python、C++...（公共分类列表）
C++学习视频
*/





/*
继承方式
语法：class 子类 : 继承方式 父类
eg:创建Son类以protected的方式继承Father类
class Son : protected Father
三种继承方式
1、public
2、protected
3、private

共有就正常，保护都保护，私有全私有，父类的私有不可访问
*/
// #include<iostream>
// #include<string>
// using namespace std;
// class Father//创立一个父类
// {
// public:
//     int m_A;//公有属性
// protected:
//     int m_B;//保护属性
// private:
//     int m_C;//私有属性，不可访问
// };
// class Son1 :public Father
// {
// public:
//     void func1(){
//         int m_A=10;
//         int m_B=15;
//         int m_C=20;
//     }

// };
// void test1(){
//     Son1 s1;
//     s1.m_A=122;
//     //s1.m_B=100;//不能访问父类的protected
//     s1.m_C=100;//父类的私有是他的小秘密，不给你访问 error: ‘int Father::m_C’ is private within this context
// }
// class Son2 : protected Father//protected继承
// {
// public:
//     void func2(){
//         int m_A=10;
//         int m_B=15;
//         int m_C=20;
//     }
// };
// void test2(){
//     Son2 s2;
//     //s1.m_A=122;//protected类外不可以访问
//     //s1.m_B=100;//protected类外不可以访问
//     //s1.m_C=100;//父类的私有是他的小秘密，不给你访问
// }
// class Son2 : private Father//protected继承
// {
// public:
//     void func3(){
//         int m_A=10;
//         int m_B=15;
//         int m_C=20;
//     }
// };
// void test3(){
//     Son2 s2;
//     //s1.m_A=122;//private可以访问
//     //s1.m_B=100;//private类外不可以访问
//     //s1.m_C=100;//父类的私有是他的小秘密，不给你访问
// }
// int main(){
//     test1();
//     test2();
//     test3();
//     system("pause");
//     return 0;
// }



//多态
#include<iostream>
using namespace std;
//动物类
class Animal
{
public://加了virtual就可以各自调用各的
    virtual void speak(){
        cout<<"动物在说话"<<endl;
    }
};
//猫类
class Cat :public Animal
{
public:
    void speak(){
        cout<<"小猫在说话"<<endl;
    }
};

//狗类
class Dog :public Animal
{
public:
    void speak(){
        cout<<"小狗在说话"<<endl;
    }
};
//执行说话函数
//如果想让猫说话，那么这个函数就不能提前绑定，需要晚绑定
void doSpeak(Animal &animal){
    animal.speak();
}

//测试阶段
void test01(){
    Cat cat;
    doSpeak(cat);//小猫在说话
    Dog dog;
    doSpeak(dog);//小狗在说话
}
int main(){
    test01();//没加virtual动物在说话
    system("pause");
    return 0;
}
/*多态满足的条件
1、有继承关系
2、子类要重写父类的虚函数*/

//动态多态的调用
//父类的指针或者引用 执行子类对象