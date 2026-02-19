## TENSOR RESHAPING

import torch

x = torch.arange(9)

x_3x3 = x.view(3,3)
print(x_3x3)
x_3x3 = x.reshape(3,3)
# these two are v similar but view acts on contiguous tensors meaning tensors are stored contiguously in memory where in memory there are pointers pointing to each element so reshape better

y = x_3x3.t() # transpose
print(y)

x1 = torch.rand((2,5))
x2 = torch.rand((2,5))
print(torch.cat((x1,x2), dim=0).shape) # .cat is concatenate, specify what dimension you want and etc
print(torch.cat((x1,x2), dim=1).shape)

z = x1.view(-1) # to flatten the entire thing
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
z = x.view(batch, -1) # keep first dimension but flatten the rest so becomes 64x10
print(z.shape)

z = x.permute(0,2,1) # a special case of transpose where lets say you want to keep the batch dimension but 2nd dimension is 5 and 3rd is 2 so swap the index
print(z.shape)

x = torch.arange(10) # [10] size

print(x.unsqueeze(0)) # adds a 1 at the front so becomes 1x10 or [1,10]
print(x.unsqueeze(1)) # 10x1

x = torch.arange(10).unsqueeze(0).unsqueeze(1) # 1x1x10

z = x.squeeze(1) # removes it to become 1x10
print(z.shape)




