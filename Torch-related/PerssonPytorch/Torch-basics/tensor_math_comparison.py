# tensor math and comparison operations

import torch

x = torch.tensor([1,2,3])
y = torch.tensor(([9,8,7]))

# addition
z1 = torch.empty(3)
torch.add(x,y, out=z1) # first way

z2 = torch.add(x,y) # second way

z = x + y # third way (preferred)

#subtraction

z = x - y

# division

z = torch.true_divide(x,y) # element wise division, y can be j an integer aswell

# inplace operations (mutate in place and doesnt make copy)

t = torch.zeros(3)
t.add_(x) # whenever u see an operation with underscore, that op is done in place
t += x # does the same thing but t = t + x doesnt do it

# exponentiation

z = x.pow(2) # elementwise power of 2 so 1,4,9

z = x ** 2 # same thing (preferred)


# simple comparison

z = x > 0
Z = x < 0

# matrix multiplication

x1 = torch.rand((2,5))
x2 = torch.rand((5,3))
x3 = torch.mm(x1, x2) # 2x3 output
x3 = x1.mm(x2) # another way to do it (torch. or the tensor itself)

# matrix exponentiation (raise the entire matrix)

matrix_exp = torch.rand((5,5))
matrix_exp.matrix_power(3)

# element wise multiplication
z = x * y

# dot product
z = torch.dot(x, y)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2) # output is (batch,n,p) keep in mind this are dimensions

# example of broadcasting
x1 = torch.rand((5,5))
x2 = torch.rand((1,5))

z = x1 - x2 # this makes sense because the row of "1" is gonna be expanded equal to each other
# broadcasting is automatically expanding to match operation

z = x1 ** x2

#other useful tensor operations
sum_x = torch.sum(x, dim=0) #dimention 0 cuz its a vector, but can change
values, indices = torch.max(x, dim=0) # can also do x.max(dim=0) same for abs, min, sum, argmax,sort, etc
values, indices = torch.min(x, dim=0)
abs_x = torch.abs(x)
z = torch.argmax(x, dim=0) # same as torch.max but only returns index of which one is max
z = torch.argmin(x, dim=0)
mean_x = torch.mean(x.float(), dim=0) # python requires it to be in float form
z = torch.eq(x, y) # check which elements are equal (True otherwise False)
sorted_y, indices = torch.sort(y, dim=0, descending = False) # sorts elements in tensor

z = torch.clamp(x, min=0, max=10) # check all elements of x that are less than 0 then set to 0 if i put max=10, then all values > 10 will be set to 10

x = torch.tensor([1,0,1,1,1], dtype= torch.bool)
z = torch.any(x) # check if any values are true or 1
z = torch.all(x) # check if all values are true or 1

