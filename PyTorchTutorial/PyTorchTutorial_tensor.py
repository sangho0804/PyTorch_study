import torch
import numpy as np

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

## tensor clear

#debug 1.
print("debug 1.\n")
print(x_data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#debug 2.
print("debug 2.\n")
print(x_np)

x_ones = torch.ones_like(x_data) 
x_rand = torch.rand_like(x_data, dtype=torch.float) 

#debug 3.
print("debug 3.\n")
print(f"Ones Tensor: \n {x_ones} \n")
print(f"Random Tensor: \n {x_rand} \n")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

#debug 4.
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")


