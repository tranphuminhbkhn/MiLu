import torch
import torch.nn as nn
import time

x = torch.randn(1000, 30)
y = torch.randn(1000, 10)


class Network(nn.Module):
	def __init__(self):
		super().__init__()
		self.w1 = nn.Linear(30, 50)
		self.w2 = nn.Linear(50, 10)
	def forward(self, x):
		h = torch.relu(self.w1(x))
		o = self.w2(h)
		return o

nn = Network()

op = torch.optim.Adam(nn.parameters(), lr=0.05)
t1 = time.time()
for i in range(2000):
	o = nn(x)
	l = torch.mean((o - y) ** 2)
	print(i, float(l))
	l.backward()
	op.step()

t2 = time.time()

print(t2 - t1)


