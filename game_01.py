class Father:
    def __init__(self,age,name):
       self.age=age
       self.name=name
    #在父类定义一个方法
    def say_hello(self):
      return f'Hi bro!my name is {self.name},I am {self.age}years old'
#定义子类，继承父类的所有
class Son(Father):
   pass#空实现，仅继承
#实例化一个对象
son_1=Son(19,'Tu')
print(son_1.say_hello())
