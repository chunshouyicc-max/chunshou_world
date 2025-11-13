# 这是之前写过的例题，再来写一遍，温故而知新
# 封装一个银行卡类:Card
# 1.Card类中有实例属性:
#           用户名:name
#           年龄:age
#           密码为私有属性:__password,
#           均可以从实例化对象过程中传入
# 2.定义一个实例方法get_user(),可以获取这并且打印用户信息(name和age)
# 3.定义一个修改密码的私有方法,__set_password()
# 4.定义一个实例方法,可以调用修改私有属性的方法,函数名称为:setpassword(),
#   如果检测到该方法传入的用户名称正确,则修改, 否则,提示'用户名称错误'
class Card:
    def __init__(self,name,age,password):
        self.name=name
        self.age=age
        self.__password=password#创建私有属性
    def get_user():
        return f'用户姓名：{self.name}\n用户年龄：{self.age}'
    #定义一个私有方法
    def __set_password(self,new_password):
        self.password=new_password
        return f'密码已修改为:{self.password}'
    #定义一个公有函数间接访问私有函数
    def set_password(self,my_name,newpassword):
        #通过名字作出判断是否能调用修改密码的私有函数
        if my_name==self.name:
            return __set_password(newpassword)
        else:
            return ('用户名称错误！')
        
#实例化对象来测试
card_1=Card('Chen',19,131499)
print(card_1.get_user())
print(card_1.setpassword('Chen',246801))
print(card_1.setpassword('tjy',246801))









# class Card:
#     def __init__(self,name,age,password):
#         self.name=name
#         self.age=age
#         self.__password=password
#     def __get_user(self):#私有方法
#         print(f'用户的名字是{self.name},年龄是{self.age}岁')
#     def get_user(self):#定义方法间接在类的内部调用私有方法
#         self.__get_user()
#     def __set_password(self,newpassword):#定义可以修改名字的私有函数
#         self.__password=newpassword
#         print(f'修改成功！新密码为{newpassword}')
#     def setpassword(self,my_name,newpassword):
#         if my_name==self.name:
#           return self.__set_password(newpassword)
#         else:
#            return ('用户名称错误')
# #调试阶段：
# card_1=Card('Chen',19,131499)
# print(card_1.get_user())
# print(card_1.setpassword('Chen',246801))
# print(card_1.setpassword('tjy',246801))
class Card:
    def __init__(self, name, age, password):  # 初始化方法
        self.name = name
        self.age = age
        self.__password = password  # 私有属性
    
    def __get_user(self):  # 私有方法
        print(f"用户名字是{self.name}, 年龄是{self.age}岁")
    
    def get_user(self):  # 公有方法调用私有方法
        self.__get_user()
    
    def __set_password(self, newpassword):  # 私有方法修改密码
        self.__password = newpassword
        print(f"修改成功！新密码为{newpassword}")
    
    def setpassword(self, my_name, newpassword):  # 公有方法验证后修改密码
        if my_name == self.name:  # 验证用户名
            self.__set_password(newpassword)
            return "密码修改成功"
        else:
            return "用户名称错误"

# 测试代码
card_1 = Card('Chen', 19, 131489)
card_1.get_user()  # 调用get_user方法
print(card_1.setpassword('Chen', 246881))
print(card_1.setpassword('Uy', 246881))