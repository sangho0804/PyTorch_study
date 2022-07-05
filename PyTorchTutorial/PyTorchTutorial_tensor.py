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
print(f"Zeros Tensor: \n {zeros_tensor}\n")

#shape 은 텐서의 차원 (dimension) 을 나타내는 (tuple)로,
#아래 함수들에서는 출력 텐서의 차원을 결정 한다. ( 무슨말임 이게 )

tensor = torch.rand(3,4)

#debug 5.
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

#debug 6.
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

#debug 7.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

#debug 8.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

#debug 9.
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#debug 10.
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

#debug 11.
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

#debug 12.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

#debug 13.
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")











