# tensor indexing

import torch

batch_size = 10
features = 25

x = torch.rand((batch_size, features))

# lets say we want to get features of first example

print(x[0]) # similar to x[0, :] meaning first row but all columns

# get first feature for all examples
print(x[:, 0]) # similar to x[:,0] meaning all rows but first column

# get first ten features
print(x[2, 0:10]) # 0:10 ---> [0,1,2,3...,9]

x[0,0] = 100 # set specific elements

# fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
print(x[indices]) #gonna pick out three elements of the tensor primarily 3rd, 6th and 9th

x = torch.rand((3,5))
rows = torch.tensor([1,0])
cols = torch.tensor([4,0])
print(x[rows,cols]) # first pick out second row and 5th column, then first row and first column so pick out 2 elements

#advanced indexing

x = torch.arange(10)
print(x[(x < 2) | (x > 8)])
print(x[x.remainder(2) == 0]) # print out elements with r =0 for mod 2

# Useful operations
print(torch.where(x > 5, x, x*2))
print(torch.tensor([0,0,1,2,2,3,4]).unique()) # get unique value of the data
print(x.ndimension()) # dimension of tensor
print(x.numel()) # count no. of elements in the tensor