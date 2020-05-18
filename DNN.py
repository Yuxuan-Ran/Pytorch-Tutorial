import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

train = datasets.MNIST('', train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST('', train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)

X_size = np.prod(train[0][0].shape)
y_range = 10

class Net(nn.Module):
	def __init__(self, _X_size = None, _y_size = None):
		super().__init__()
		self.fc1 = nn.Linear(_X_size, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 64)
		self.fc4 = nn.Linear(64, _y_size)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		return F.log_softmax(x, dim=1)

net = Net(X_size, y_range)

import torch.optim as optim

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

for epoch in range(1):
	for data in trainset:
		X, y = data
		net.zero_grad()
		output = net(X.view(-1, 784))
		loss = F.nll_loss(output, y)
		loss.backward()
		optimizer.step()
	print(loss)

correct = 0
total = 0

with torch.no_grad():
	for data in testset:
		X, y = data
		output = net(X.view(-1,784))
		for idx, i in enumerate(output):
			if torch.argmax(i) == y[idx]:
				correct += 1
			total += 1

print ("Accuracy: ", round(correct/total, 3))
