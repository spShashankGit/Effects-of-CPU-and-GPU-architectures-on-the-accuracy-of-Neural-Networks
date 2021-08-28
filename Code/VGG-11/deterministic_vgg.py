import torch  
import torch.nn as nn

import numpy as np
from numpy import load

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#VGG-11 class
class VGG_11(nn.Module):
    
    def __init__(self):
        super(VGG_11,self).__init__()
        
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
            #nn.LocalResponseNorm(64),
            nn.ReLU(),                              # Activation layer
            nn.MaxPool2d(kernel_size=maxpool_kernel,stride=maxpool_stride),
            
            nn.Conv2d(hid1_size,hid2_size,k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid2_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel,stride=maxpool_stride),
            
            nn.Conv2d(hid2_size,hid3_size,k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid3_size),
            nn.ReLU(),
            
            nn.Conv2d(hid3_size,hid3_size,k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid3_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel,stride=maxpool_stride),
            
            nn.Conv2d(hid3_size,hid4_size,k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid4_size),
            nn.ReLU(),
            
            nn.Conv2d(hid4_size,hid4_size,k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid4_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel,stride=maxpool_stride),
            
            nn.Conv2d(hid4_size,hid4_size,k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid4_size),
            nn.ReLU(),
            
            nn.Conv2d(hid4_size,hid4_size,k_conv_size, stride=conv_stride, padding=conv_pad),
            nn.BatchNorm2d(hid4_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=maxpool_kernel,stride=maxpool_stride), 
            
            nn.Flatten(),
            
            nn.Linear(512, out_size)
            
        )
        
            
        
    def forward(self,x):
            out = self.convLayer(x)
            
            return out 
    
def main():
    batch_size_required = 64
    max_epoch = 10






    #2- Load dataset from the .npy array
    xtrain_path = '/home/rgb/Documents/Final run/effects-of-cpu-and-gpu-architectures-on-the-accuracy-of-neural-networks/Code/VGG-11/train_data.npy'  # Training data file 
    ytrain_path = '/home/rgb/Documents/Final run/effects-of-cpu-and-gpu-architectures-on-the-accuracy-of-neural-networks/Code/VGG-11/train_label.npy' # Training label file
    xtest_path = '/home/rgb/Documents/Final run/effects-of-cpu-and-gpu-architectures-on-the-accuracy-of-neural-networks/Code/VGG-11/test_data.npy'    # Test data file
    ytest_path = '/home/rgb/Documents/Final run/effects-of-cpu-and-gpu-architectures-on-the-accuracy-of-neural-networks/Code/VGG-11/test_label.npy'   # Test label file


    ## Train dataloader
    my_x = load(xtrain_path) # a list of numpy arrays
    my_y = load(ytrain_path) # another list of numpy arrays (targets)

    tensor_x = torch.Tensor(my_x) # transform to torch tensor
    tensor_y = torch.Tensor(my_y)

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    training_dataloader = DataLoader(my_dataset,batch_size=64,shuffle=False) # create your dataloader

    ## Test dataloader
    test_x = load(xtrain_path) # a list of numpy arrays
    test_y = load(ytrain_path) # another list of numpy arrays (targets)

    tensor__test_x = torch.Tensor(test_x) # transform to torch tensor
    tensor__test_y = torch.Tensor(test_y)

    dataset_test = TensorDataset(tensor__test_x,tensor__test_y) # create your datset
    test_dataloader = DataLoader(dataset_test,batch_size=64,shuffle=False) # create your dataloader




    x_test_gpu = torch.FloatTensor(load(xtest_path)).cuda()         # Load .npy file

    batch_size = int(len(x_test_gpu)/batch_size_required)           # Form batches
    res=[]
    for i in range (batch_size):
        batched_data = x_test_gpu[i*batch_size_required:i*batch_size_required+batch_size_required]
        res.append(batched_data)
    x_test_gpu = res


    y_test_gpu = torch.FloatTensor(load(ytest_path)).cuda()

    batch_size = int(len(y_test_gpu)/batch_size_required)           # Form batches
    res=[]
    
    for i in range (batch_size):
        batched_data = y_test_gpu[i*batch_size_required:i*batch_size_required+batch_size_required]
        res.append(batched_data)
    y_test_gpu = res



    #1 - Set seed
    seedVal = 7184
    torch.manual_seed(seedVal)
    torch.cuda.manual_seed(seedVal)
    torch.cuda.manual_seed_all(seedVal)
    np.random.seed(seedVal)





    #3 Initialize VGG model on GPU
    vggGPUAda = VGG_11().cuda()              #Initialize model on GPU




    #4 Print accuracy
    print('Before: Prining accuracy of the model')
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for i, (images,labels) in enumerate(test_dataloader,0):
            #images, labels = data

            #images = data
            #labels = y_test_gpu[i]
            images, labels = images.cuda(), labels.cuda() # Transfer images and labels from CPU to GPU
            outputs = vggGPUAda(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                
                class_correct[int(label.item())] += c[i].item()
                class_total[int(label.item())] += 1

    for i in range(10):
        print('Accuracy of %5s : %.2f %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    print('Accuracy of the network on the %d test images: %.2f %% \n' % (len(y_test_gpu)*batch_size_required,100 * correct / total))





    #5 - Initialize the optimizer
    optimizer = optim.Adadelta(vggGPUAda.parameters())





    #6 - Initialize loss function
    lossFun = nn.CrossEntropyLoss()





    #7 - Load CIFAR-10 training dataset
    x_train_gpu = torch.FloatTensor(load(xtrain_path)).cuda()

    batch_size = int(len(x_train_gpu)/batch_size_required)           # Form batches
    res=[]
    
    for i in range (batch_size):
        batched_data = x_train_gpu[i*batch_size_required:i*batch_size_required+batch_size_required]
        res.append(batched_data)
    x_train_gpu = res

    y_train_gpu = torch.FloatTensor(load(ytrain_path)).cuda()

    batch_size = int(len(y_train_gpu)/batch_size_required)           # Form batches
    res=[]
    
    for i in range (batch_size):
        batched_data = y_train_gpu[i*batch_size_required:i*batch_size_required+batch_size_required]
        res.append(batched_data)
    y_train_gpu = res





    #8 - Set seed again
    seedVal = 7184
    torch.manual_seed(seedVal)
    torch.cuda.manual_seed(seedVal)
    torch.cuda.manual_seed_all(seedVal)
    np.random.seed(seedVal)





    #9 - Train for 10 epochs
    loss_value = []

    if(torch.cuda.is_available()):
        for epoch in range(max_epoch):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, (inputs,labels) in enumerate(training_dataloader):

                #inputs = data

                #labels = y_train_gpu[i]
                inputs, labels = inputs.cuda(), labels.cuda() # Transfer images and labels from CPU to GPU

                optimizer.zero_grad()

                #vgg_11_gpu = vggInp.cuda()
                outputs = vggGPUAda(inputs)
                
                labels = labels.to(device="cuda", dtype=torch.int64)
                
                loss = lossFun(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loss_value.append(running_loss)
            print('Loss:', epoch, ":", (running_loss/i))
            
        print('Finished Training on GPU')

    else:
        print('GPU not present')



    #10 - Put training model in eval mode
    vggGPUAda.eval()





    #11 - Print accuracy
    print('After: Prining accuracy of the model')
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for i, (images,labels) in enumerate(test_dataloader):
            #images, labels = data

            #images = data
            #labels = y_test_gpu[i]
            images, labels = images.cuda(), labels.cuda() # Transfer images and labels from CPU to GPU

            outputs = vggGPUAda(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                
                class_correct[int(label.item())] += c[i].item()
                class_total[int(label.item())] += 1

    for i in range(10):
        print('Accuracy of %5s : %.2f %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    print('Accuracy of the network on the %d test images: %.2f %% \n' % (len(y_test_gpu)*batch_size_required,100 * correct / total))






#12 - Load main method
if __name__ == "__main__":
    # execute only if run as a script
    main()