

class Student():
    def __init__(self,name,grades):
        
        self.name = name
        self.age= 23
        self.grades = grades
    
    def average(self):
       result = sum(self.grades)/len(self.grades)
       return result
    
    def mathema(self,index):
        
       mathematics = self.grades[index]
       
       return mathematics
    
    

student = Student("boby" , (2,3,4,5,6,4))

print(student.name)

maths = student.average()       

print(maths)