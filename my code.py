def dummy(func):
    def wrapper(*args, **kwargs):
        print("A")
        return func(5,args[1], **kwargs)
    return wrapper
@dummy
def my_function(a,b):
    print(f"{a}, {b} ")
    
my_function(100, 200)