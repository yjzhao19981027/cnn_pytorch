import data_utils
from cnn import Net
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

#   get datasets
train_loader = data_utils.get_train_loader()

#   initiate net & optimizer
learning_rate = 0.01
net = Net()
optimizer = optim.SGD(net.parameters(), learning_rate, weight_decay=0.01)
loss_data = []
loss = 0

for epoch in range(5):
    for images, labels in train_loader:
        optimizer.zero_grad()
        #   forward
        out = net(images)
        entropy = nn.CrossEntropyLoss()

        loss = entropy(out, labels)
        # loss_data.append(loss.data)

        print(loss)
        #   backward
        loss.backward()

        #   update parameters
        #   use torch.optim
        optimizer.step()
        # break
    print(epoch)
    loss_data.append(loss.data)

PATH = './weights/cnn_weights.pth'
torch.save(net.state_dict(), PATH)

len = len(loss_data)
plt.plot(range(len), loss_data)
plt.show()
