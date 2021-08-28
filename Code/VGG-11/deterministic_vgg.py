import torch  
import torch.nn as nn

import numpy as np
from numpy import load

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from datetime import datetime  
# VGG-11 class
class VGG_11(nn.Module):
    
    def __init__(self):
        super(VGG_11, self).__init__()
        
        # Setting up the layers of Convolutional neural network
        in_size = 3                 # number of channel in the input image
    
        hid1_size = 64              # no of output channel from first CNN layer

        hid2_size = 128             # no of output channel from second CNN layer

        hid3_size = 256             # no of output channel from third CNN layer

        hid4_size = 512             # no of output channel from forth CNN layer

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        out_size = len(classes)     # no of categories in the dataset

        k_conv_size = 3             # 3x3 convolutional kernel

        conv_stride = 1             # conv stride 1

        conv_pad = 1                # conv padding 1

        maxpool_kernel = 2          # maxpool layer kernel size 2 x 2

        maxpool_stride = 2          # maxpool layer stride 2

        self.convLayer = nn.Sequential(
            nn.Conv2d(in_size, hid1_size, k_conv_size, stride=conv_stride, padding=conv_pad),    # conv layer
            nn.BatchNorm2d(hid1_size),
            # nn.LocalResponseNorm(64),
            nn.ReLU(),                              # Activation layer
            nn.MaxPool2d(kernel_size=maxpool_kernel, stride=maxpool_stride),
            
            nn.Conv2d(hid1_size, hid2_size, k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid2_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel, stride=maxpool_stride),
            
            nn.Conv2d(hid2_size, hid3_size, k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid3_size),
            nn.ReLU(),
            
            nn.Conv2d(hid3_size, hid3_size, k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid3_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel,stride=maxpool_stride),
            
            nn.Conv2d(hid3_size, hid4_size, k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid4_size),
            nn.ReLU(),
            
            nn.Conv2d(hid4_size, hid4_size, k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid4_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel, stride=maxpool_stride),
            
            nn.Conv2d(hid4_size, hid4_size, k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid4_size),
            nn.ReLU(),
            
            nn.Conv2d(hid4_size, hid4_size, k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid4_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel, stride=maxpool_stride),
            
            nn.Flatten(),
            
            nn.Linear(512, out_size)
            
        )
        
    def forward(self, x):
        out = self.convLayer(x)
        return out


def main(seed_value=0):
    begin_time = datetime.now()
    batch_size_required = 64
    max_epoch = 1

    #  2- Load dataset from the .npy array
    xtrain_path = './train_data.npy'  # Training data file
    ytrain_path = './train_label.npy'  # Training label file
    xtest_path = './test_data.npy'    # Test data file
    ytest_path = './test_label.npy'   # Test label file

    # Train dataloader
    my_x = load(xtrain_path)  # a list of numpy arrays
    my_y = load(ytrain_path)  # another list of numpy arrays (targets)

    tensor_x = torch.Tensor(my_x)  # transform to torch tensor
    tensor_y = torch.Tensor(my_y)

    my_dataset = TensorDataset(tensor_x, tensor_y) # create your dataset
    training_dataloader = DataLoader(my_dataset, batch_size=batch_size_required, shuffle=False)  # create the dataloader

    # Test dataloader
    test_x = load(xtest_path)  # a list of numpy arrays
    test_y = load(ytest_path)  # another list of numpy arrays (targets)

    tensor__test_x = torch.Tensor(test_x)  # transform to torch tensor
    tensor__test_y = torch.Tensor(test_y)

    dataset_test = TensorDataset(tensor__test_x, tensor__test_y)  # create your dataset
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size_required, shuffle=False)  # create the dataloader

    # 1 - Set seed
    seedVal = seed_value
    torch.manual_seed(seedVal)
    torch.cuda.manual_seed(seedVal)
    torch.cuda.manual_seed_all(seedVal)
    np.random.seed(seedVal)

    # 3 Initialize VGG model on GPU
    vggGPUAda = VGG_11()  # Initialize model on GPU
    vggGPUAda = vggGPUAda.double()
    vggGPUAda = vggGPUAda.to(device="cuda")

    # 5 - Initialize the optimizer
    optimizer = optim.SGD(vggGPUAda.parameters(), lr=0.1, momentum=0.9, nesterov=False)

    # 6 - Initialize loss function
    lossFun = nn.CrossEntropyLoss()

    # 4 Print accuracy
    print('Before: Printing accuracy of the model')
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_dataloader, 0):
            # images, labels = data
            # images = data
            # labels = y_test_gpu[i]
            images, labels = images.cuda(), labels.cuda()  # Transfer images and labels from CPU to GPU
            images = images.double()
            outputs = vggGPUAda(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for idx in range(list(labels.shape)[0]):
                label = labels[idx]
                
                class_correct[int(label.item())] += c[idx].item()
                class_total[int(label.item())] += 1

    for class_idx in range(10):
        print('Accuracy of %5s : %.2f %%' % (
            classes[class_idx], 100 * class_correct[class_idx] / class_total[class_idx]))

    print('Accuracy of the network on the %d test images: %.2f %% \n' % (10000, 100 * correct / total))

    # 9 - Train for 10 epochs
    loss_value = []

    # 8 - Set seed again
    seedVal = seed_value
    torch.manual_seed(seedVal)
    torch.cuda.manual_seed(seedVal)
    torch.cuda.manual_seed_all(seedVal)
    np.random.seed(seedVal)

    stepFileName = "step_loss_of_model" + str(datetime.now())+'.txt'
    file2write=open(stepFileName,'w')
    file2write.write(str('\n'))

    for epoch in range(max_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        count = 1
        for inputs, labels in training_dataloader:

            inputs = inputs.to(device="cuda")
            inputs = inputs.double()
            labels = labels.to(device="cuda", dtype=torch.int64)  # Transfer images and labels from CPU to GPU

            optimizer.zero_grad()

            outputs = vggGPUAda(inputs)

            loss = lossFun(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_value.append(running_loss)

            print('Loss at iteration %d: %0.20f' % (count, running_loss))
            file2write.write(str('Loss at iteration %d: %0.20f \n' %(count, loss.item())))
            count += 1

        print('Loss:', epoch, ":", (running_loss/count))
        
    print('Finished Training on GPU')

    # 10 - Put training model in eval mode
    vggGPUAda.eval()

    # 11 - Print accuracy
    print('After: Printing accuracy of the model')
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_dataloader):
            # images, labels = data

            # images = data
            # labels = y_test_gpu[i]
            images = images.cuda()  # Transfer images and labels from CPU to GPU
            images = images.double()
            labels = labels.cuda()
            outputs = vggGPUAda(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for idx in range(list(labels.shape)[0]):
                label = labels[idx]
                
                class_correct[int(label.item())] += c[idx].item()
                class_total[int(label.item())] += 1

    for class_idx in range(10):
        print('Accuracy of %5s : %.2f %%' % (
            classes[class_idx], 100 * class_correct[class_idx] / class_total[class_idx]))

    print('Accuracy of the network on the %d test images: %.2f %% \n' % (total, 100 * correct / total))
    print('Time required to run the model on GPU is', datetime.now() - begin_time)


# 12 - Load main method
if __name__ == "__main__":
    # execute only if run as a script
    #seeds = [7184, 13474, 32889, 56427, 59667]
    seeds=[7184]
    for seed in seeds:
        main(seed_value=seed)
