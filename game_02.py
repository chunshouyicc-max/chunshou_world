# #定义三个类，再各自定义say_hello()的方法
# class duck:
#     def say_hello(self):
#         return '嘎嘎嘎'
# class human:
#     def say_hello(self):
#         return 'hello'
# class dog:
#     def say_hello(self):
#         return '汪汪汪'
# #分别实例化对象，再各自调用say_hello()方法
# duck_1=duck()
# human_1=human()
# dog_1=dog()
# print(duck_1.say_hello())
# print(human_1.say_hello())
# print(dog_1.say_hello())


"""
类方法，静态方法，实例方法
"""
class Person:
    total_people = 0  # 类属性，记录总人数

    def __init__(self, name, age):
        self.name = name  # 实例属性
        self.age = age
        Person.total_people += 1  # 每创建一个实例，总人数+1

    # 1. 实例方法：操作实例属性
    def show_info(self):
        print(f"姓名: {self.name}, 年龄: {self.age}")

    # 2. 类方法：操作类属性（统计总人数）
    @classmethod
    def get_total_people(cls):
        return cls.total_people

    # 3. 静态方法：与实例和类无关的工具函数（验证年龄是否合法）
    @staticmethod
    def is_valid_age(age):
        return 0 < age <= 120


# 创建实例
person1 = Person("Alice", 30)
person2 = Person("Bob", 25)

# 调用实例方法
person1.show_info()  # 输出：姓名: Alice, 年龄: 30

# 调用类方法
print(f"总人数: {Person.get_total_people()}")  # 输出：总人数: 2

# 调用静态方法
print(Person.is_valid_age(150))  # 输出：False
print(Person.is_valid_age(25))  # 输出：True

"""
特性	                实例方法	            类方法	        静态方法
装饰器	        无（普通方法）	        @classmethod	    @staticmethod
第一个参数	    self（实例本身）	    cls（类本身）	        无（可自定义参数）
访问权限	        可访问实例属性和类属性	可访问类属性	        不能访问实例或类属性
调用方式	        必须通过实例调用	    可通过类或实例调用	    可通过类或实例调用
使用场景	        操作实例属性或行为	    操作类属性等      	与类无关的工具函数
"""
