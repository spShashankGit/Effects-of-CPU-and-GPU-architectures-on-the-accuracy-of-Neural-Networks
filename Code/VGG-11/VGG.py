#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch         
import torchvision
import torchvision.transforms as transforms


# In[2]:


trainset = torchvision.datasets.CIFAR10(train = True, target_transform = transforms.ToTensor(),root="cifar-10", download = True)


# In[3]:


trainset


# In[4]:


trainloader = torch.utils.data.DataLoader(trainset,batch_size=4,shuffle=True,num_workers=2)


# In[5]:


testset = torchvision.datasets.CIFAR10(train = False, target_transform = transforms.ToTensor(),root="cifar-10-test", download = True)


# In[6]:


testset


# In[7]:


testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, num_workers=2)


# In[8]:


labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[9]:


transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Preview CIFAR-10 dataset images

# In[10]:


import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
print(dataiter)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# ## Check if GPU is present - if yes computer on GPU else on CPU

# In[11]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# ###### Setting up the layers of Convolutional neural network

# In[12]:


import torch.nn as nn


# In[13]:


in_size = 3                 # number of channel in the input image
 
hid1_size = 64              # no of output channel from first CNN layer

hid2_size = 128              # no of output channel from second CNN layer

hid3_size = 256

hid4_size = 512

out_size = len(labels)      # no of categories in the dataset
#set_trace()
k_conv_size = 3             # 3x3 convolutional kernel


# In[14]:


class VGG_11(nn.Module):
    
    def __init__(self):
        super(ConvNet,self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_size, hid1_size, k_conv_size),    # conv layer
            nn.BatchNorm2d(hid1_size),                     # Batch normalization
            nn.ReLU(),                                     # Activation layer
            nn.MaxPool2d(kernel_size=3))                   # Pooling layer with kernel size 2x2
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(hid1_size,hid2_size,k_conv_size),
            nn.BatchNorm2d(hid2_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.fullyConnected = nn.Linear(512, out_size)     # Fully connected layer
        
    def forward(self,x):
            out = self.layer1(x)
            #print(out.shape)
            
            out = self.layer2(out)
            #print(out.shape)
            
            out = out.reshape(out.size(0), -1) #######
            #print(out.shape)
            
            out = self.fullyConnected(out)
            #print(out.shape)
            
            return out


# In[ ]:





# ## Define a Convolutional Neural Network

# In[21]:


import torch.nn as nn
import torch.nn.functional as F

in_size = 3                 # number of channel in the input image
 
hid1_size = 64              # no of output channel from first CNN layer

hid2_size = 128              # no of output channel from second CNN layer

out_size = len(labels)      # no of categories in the dataset

k_conv_size = 3             # 3x3 convolutional kernel

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# ## Define a Loss function and optimizer

# In[22]:


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# ## Train the network

# In[23]:


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


# ### save the trained model

# In[24]:


PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)


# ## Test the network on the test data

# In[25]:


dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[ ]:





# In[26]:


outputs = net(images)


# In[27]:


_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# ### Accuracy on whole test-dataset

# In[28]:


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# ### Which classes perform well and which classes did not?

# In[29]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# In[ ]:




