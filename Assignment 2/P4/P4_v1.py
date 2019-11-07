import torch
import torchvision
from torchvision.datasets import CIFAR10
#%%
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
#%%
bs = 16

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#%%
class Net(nn.Module):
    def __init__(self):
        self.ks = 3
        self.pd = 1
        self.in_ch = 3
        
        super().__init__()
        
        self.conv1_block =  nn.Sequential(
                nn.Conv2d(self.in_ch, 32*self.in_ch, kernel_size = self.ks, padding = 1),
                nn.BatchNorm2d(32*self.in_ch),
                nn.ReLU(),
#                nn.MaxPool2d(2, 2)
                )

        self.conv2_block =  nn.Sequential(
                nn.Conv2d(32*self.in_ch, 16*self.in_ch, kernel_size = self.ks, padding = 1),
                nn.BatchNorm2d(16*self.in_ch),
                nn.ReLU(),
#                nn.MaxPool2d(2, 2)
                )       

        self.conv3_block =  nn.Sequential(
                nn.Conv2d(16*self.in_ch, 8*self.in_ch, kernel_size = self.ks, padding = 1),
                nn.BatchNorm2d(8*self.in_ch),
                nn.ReLU(),
#                nn.MaxPool2d(2, 2)
                ) 

        self.conv4_block =  nn.Sequential(
                nn.Conv2d(8*self.in_ch, 4*self.in_ch, kernel_size = self.ks, padding = 1),
                nn.BatchNorm2d(4*self.in_ch),
                nn.ReLU(),
#                nn.MaxPool2d(2, 2)
                )    
        
        self.FClayer = nn.Sequential(
                nn.Linear(32*32*4*self.in_ch, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512,10),
                nn.Softmax()
                )

    def forward(self, x):
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.conv3_block(x)
        x = self.conv4_block(x)
        
        x = x.view(-1, 32*32*4*self.in_ch)
        x = self.FClayer(x)
        return x
#%%
net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)
#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#%%
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(inputs, )
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()
        if i % bs == bs-1:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss /2000))
            running_loss = 0.0

print('Finished Training')
#%%
with open('results.txt', 'w') as f:
    f.writelines(['Image ID (name)', '\t', 'Predicted Label' , '\n'])

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
       
        with open('results.txt', 'a') as f:
            for l, p in zip([classes[i] for i in labels], [classes[i.item()] for i in predicted]):
                f.writelines([l, '\t', '\t', p , '\n'])
        f.close()
#%%                
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
#%%


    












