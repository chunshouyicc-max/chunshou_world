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
    def get_user(self):
        return f"用户姓名：{self.name} 用户年龄：{self.age}"
    #定义一个私有方法
    def __set_password(self,new_password):
        self.__password=new_password
        return f'密码已修改为:{self.__password}'
    #定义一个公有函数间接访问私有函数
    def set_password(self,my_name,new_password):
        #通过名字作出判断是否能调用修改密码的私有函数
        if my_name==self.name:
            return self.__set_password(new_password)
        else:
            return ('用户名称错误！')
        
#实例化对象来测试
card_1=Card('Chen',19,131499)
print(card_1.get_user())
print(card_1.set_password('Chen',246801))
print(card_1.set_password('tjy',246801))