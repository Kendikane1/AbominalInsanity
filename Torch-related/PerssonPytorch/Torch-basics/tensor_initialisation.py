# Ariz's pytorch learning journey following Aladdin Persson's curriculum
# run print(torch.__version__) to check the version

import torch
from mpmath import linspace

# to give an option if a user has cpu or gpu
device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32, device=device, requires_grad=True) # two rows, 3 columns / requires_grad is for autograd

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

#other common initialisation methods

#if dont have values readily available,
x = torch.empty(size = (3,3))

x = torch.zeros(3,3)
x = torch.rand(3,3) #values from a uniform dist rangin from 0-1
print(x)
x = torch.ones(3,3) # matrix with ones
x = torch.eye(5,5) #identity matrix
x = torch.arange(start=0, end=5, step=1)
print(x)
x = torch.linspace(start=0.1, end=1, steps=10) #evenly spaced tensor similar to arange where steps is no. of points in the range
x = torch.empty(size=(1,5)).normal_(mean=0, std=1) # uninitialised matrix with values across a normal dist.
x = torch.empty(size=(1,5)).uniform_(0,1)
x = torch.diag(torch.ones(3)) # 3x3 diagonal matrix (doesnt have to be ones, can be anything)

# How to initialise and convert tensors to other types (int,float,double)

tensor = torch.arange(4) # start is 0 by default

print(tensor.bool()) #.bool just converts it to T/F
print(tensor.short()) # int16
print(tensor.long()) # int64 and more used
print(tensor.half()) # float16
print(tensor.float()) # float32
print(tensor.double()) # float64

# array to tensor conversion and etc
import numpy as np
np_array = np.zeros((5,5))
tensor = torch.from_numpy(np_array) # to convert array to tensor
np_array_back = tensor.numpy() #to bring back the array