from pypads.app.base import PyPads
from datetime import datetime  

# CIFAR 10 dataset - Numpy
#import os
import platform
import pickle

import numpy as np
from numpy import asarray
from numpy import save
from numpy import load

import torch  
import torch.nn as nn
import torch.optim as optim

import os
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
        
#vgg_11 = VGG_11()

# Unpickle a data item
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

# Load CIFAR-10 batch
def load_CIFAR_batch(filename):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3,32,32)
        Y = np.array(Y)
        return X, Y   

# Load full CIFAR-10 dataset
def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=0, num_test=10000):
    # Load the raw CIFAR-10 data
    dirname = os.path.dirname(__file__)
    cifar10_dir = 'CIFAR-10-DS/cifar-10-batches-py/'
    filename = os.path.join(dirname, cifar10_dir)
    
    X_train, y_train, X_test, y_test = load_CIFAR10(filename)

    num_training=X_train.shape[0]
    num_test=X_test.shape[0]

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')

    x_train /= 255
    x_test /= 255

    #Save dataset in npy files
    save('train_data.npy', x_train)
    save('train_label.npy', y_train)

    save('test_data.npy', x_test)
    save('test_label.npy', y_test)

    # Loading from npy files
    x_train = load('train_data.npy')
    y_train = load('train_label.npy')

    x_test = load('test_data.npy')
    y_test = load('test_label.npy')

    #Shuffle data 
    data = list(zip(x_train, y_train))
    np.random.shuffle(data)
    x_train, y_train = zip(*data)

    # Save the shuffled data
    save('train_data.npy', x_train)
    save('train_label.npy', y_train)

    #return x_train, y_train, x_test, y_test 


#Create custom size batches of the dataset
def createBatches(data,batch_size_required,device="cpu"):
    batch_size = int(len(data)/batch_size_required)
    res=[]
    
    for i in range (batch_size):
        batched_data = data[i*batch_size_required:i*batch_size_required+batch_size_required]
        res.append(batched_data)


    if (device == "cpu"):
        res = np.asarray(res)

    elif (device == "cuda"):
        return res
    
    return res


def useLossFunction():
    criterion = nn.CrossEntropyLoss()
    return criterion

def useOptimizerFunction(name, vgg11Optim, learning_rate=0.1):
    #learning_rate = 0.01                        # Learning rate
    #momentum_val = 0.9                          # Momentum value
    print('learning rate ', learning_rate)
    print(' optimizer ', name)
    optimizer=''
    if(name == 'Adadelta'):
        optimizer = optim.Adadelta(vgg11Optim.parameters())

    elif(name=='SGD'):
        momentum_val = 0.5
        optimizer = optim.SGD(vgg11Optim.parameters(), lr=learning_rate, momentum=momentum_val)
    
    elif(name=='NAG'):
        momentum_val = 0.5
        optimizer = optim.SGD(vgg11Optim.parameters(), lr=learning_rate, momentum=momentum_val, dampening=0, weight_decay=0, nesterov=True)

    return optimizer

def trainNetwork(max_epoch, x_train_tensor,y_train_tensor,optimizer,lossFun,dev,vggCpu,milestonesVal,gammaVal):
    running_loss_arr = []
    print('dev ', dev)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestonesVal, gamma=gammaVal)
    print('Milestone values are ', milestonesVal)
    print('Gamma values are ', gammaVal)

    for epoch in range(max_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(x_train_tensor, 0):

            inputs = data
            labels = y_train_tensor[i]
            optimizer.zero_grad()

            outputs = vggCpu(inputs)
            
            #labels = labels.to(device=dev, dtype=torch.int64)
            
            loss = lossFun(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # print statistics        
            # if i % 100 == 99:    # print every 100 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 100))

            #     running_loss = 0.0
            #     loss_value.append(running_loss)
            #     # print statistics        
            #     #if i % 100 == 99:    # print every 100 mini-batches
            #     #    
                    
            #         #running_loss = 0.0
        print('Loss:', epoch, ":", (running_loss/i))
        scheduler.step()
            

    print('Finished Training on CPU')

def trainNetworkOnGPU(max_epoch, x_train_tensor,y_train_tensor,optimizer,lossFun,dev,vggInp,milestoneVal,gammaVal):
    loss_value = []
    if(dev.type == "cuda"):

        # gamma = 0.5 [50,100,125]
        # gamma = 0.5 [50,75,100,125]
        # gamma =  [0.5, 0.6]
        # milestoneVal = [50,75,100,125]
        # gammaVal = 0.8
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestoneVal, gamma=gammaVal)
        print('Milestone values are ', milestoneVal)
        print('Gamma values are ', gammaVal)
        for epoch in range(max_epoch):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(x_train_tensor, 0):

                inputs = data

                labels = y_train_tensor[i]
                labels.cuda()

                optimizer.zero_grad()

                vgg_11_gpu = vggInp.cuda()
                outputs = vgg_11_gpu(inputs)
                
                labels = labels.to(device=dev, dtype=torch.int64)
                
                loss = lossFun(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loss_value.append(running_loss)
                # print statistics        
                #if i % 100 == 99:    # print every 100 mini-batches
                #    
                    
                #running_loss = 0.0
            print('Loss:', epoch, ":", (running_loss/i))
            scheduler.step()
            

        print('Finished Training on GPU')
        return loss_value
    else:
        print('GPU not present')



#Accuracy of individual classes and overall dataset
def AccuracyOfIndividualClassesAndDataset(x_test_t,y_test_t,bs,vgg,msg):
    print(msg)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        for i, data in enumerate(x_test_t, 0):
            #images, labels = data

            images = data
            labels = y_test_t[i]

            outputs = vgg(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                
                class_correct[int(label.item())] += c[i].item()
                class_total[int(label.item())] += 1

    accuracyFileName = "Accuracy_of_model" + str(datetime.now())+'.txt'
    file2write=open(accuracyFileName,'w')
    file2write.write(str(msg + '\n'))
    for i in range(10):
        #print('Accuracy of %5s : %.2f %%' % (
        #    classes[i], 100 * class_correct[i] / class_total[i]))
        file2write.write(str('Accuracy of %5s : %.2f %% \n' % (
            classes[i], 100 * class_correct[i] / class_total[i])))
    
    print('Accuracy of the network on the %d test images: %.2f %%' % (len(y_test_t)*bs,100 * correct / total))
    file2write.write(('Accuracy of the network on the %d test images: %.2f %% \n' % (len(y_test_t)*bs,100 * correct / total)))
    #file2write.write(str(pltdata))
    file2write.close()

def saveTrainedModel(name):
    if (name):
        PATH = './' + name + '.pth'
    else:
         PATH = './cifar_vgg11.pth'
    
# torch.save(vgg_11.state_dict(), PATH)


#Main function
def main():

    begin_time = datetime.now()   

    # Initializing pypads
    #tracker = PyPads( autostart=True)
    #tracker.start_track(experiment_name="Effect of GPUs - VGG 11")

    torch.manual_seed(0)            # to set same random number to all devices [4]
    batch_size = 64                 # Batch size
    device = torch.device("cpu")    # Set the torch device to CPU for CPU run
    max_epoch_num = 1             # Maximun numbe of epochs
    learning_rate_val = 0.1         # Learning rate
    milestoneVal = [50,75,100,125]  # Milestones values for Learning Rate Scheduler
    gammaVal = 0.8                  # Gamma value the Learning Rate  Scheduler


    
    #Get train and test dataset
    #get_CIFAR10_data()

    # Load dataset from npy files
    dirname = os.path.dirname(__file__)

    xtrain_path = 'train_data.npy'
    xtrain_filename = os.path.join(dirname, xtrain_path)
    x_train = load(xtrain_filename)

    ytrain_path = 'train_label.npy'
    ytrain_filename = os.path.join(dirname, ytrain_path)
    y_train = load(ytrain_filename)

    xtest_path = 'test_data.npy'
    xtest_filename = os.path.join(dirname, xtest_path)
    x_test = load(xtest_filename)

    ytest_path = 'test_label.npy'
    ytest_filename = os.path.join(dirname, ytest_path)
    y_test = load(ytest_filename)

    #Divide the dataset into small batches
    x_train = createBatches(x_train,batch_size)
    y_train = createBatches(y_train,batch_size)

    x_test = createBatches(x_test,batch_size)
    y_test = createBatches(y_test,batch_size)


    # Convert npArray to tensor
    x_train_tensor = torch.as_tensor(x_train)
    y_train_tensor = torch.as_tensor(y_train)

    x_test_tensor = torch.as_tensor(x_test)
    y_test_tensor = torch.as_tensor(y_test)

    vggCpu = VGG_11()

    # Adadelta Optimizer
    criterion = useLossFunction()
    AccuracyOfIndividualClassesAndDataset(x_test_tensor,y_test_tensor,batch_size,vggCpu,'Before')
    optimizer=useOptimizerFunction('Adadelta',vggCpu, learning_rate=learning_rate_val)
    trainNetwork(max_epoch_num, x_train_tensor,y_train_tensor,optimizer,criterion,device,vggCpu, milestoneVal,gammaVal)
    AccuracyOfIndividualClassesAndDataset(x_test_tensor,y_test_tensor,batch_size,vggCpu,'After')

    # # SGD Optimizer
    # optimizer=useOptimizerFunction('SGD')
    # trainNetwork(max_epoch_num, x_train_tensor,y_train_tensor,optimizer,criterion,device)
    # AccuracyOfIndividualClassesAndDataset(x_test_tensor,y_test_tensor,batch_size)

    # # NAG optimzer
    # optimizer=useOptimizerFunction('NAG')
    # trainNetwork(max_epoch_num, x_train_tensor,y_train_tensor,optimizer,criterion,device)
    # AccuracyOfIndividualClassesAndDataset(x_test_tensor,y_test_tensor,batch_size)


    print('Time required to run the model on CPU is', datetime.now() - begin_time)

    # If GPU is present run the code on the GPU 
    begin_time = datetime.now()  
    if (torch.cuda.is_available()):
        
        print('******GPU experiment starts from here*******')
        device = torch.device('cuda')  

        # Load data from the numpy files into tensor which are stored on GPU
        # dirname = os.path.dirname(__file__)

        # xtrain_path = 'train_data.npy'
        # xtrain_filename = os.path.join(dirname, xtrain_path)
        x_train_gpu = torch.FloatTensor(load(xtrain_filename)).cuda()

        # ytrain_path = 'train_label.npy'
        # ytrain_filename = os.path.join(dirname, ytrain_path)
        y_train_gpu = torch.FloatTensor(load(ytrain_filename)).cuda()

        # xtest_path = 'test_data.npy'
        # xtest_filename = os.path.join(dirname, xtest_path)
        x_test_gpu = torch.FloatTensor(load(xtest_filename)).cuda()

        # ytest_path = 'test_label.npy'
        # ytest_filename = os.path.join(dirname, ytest_path)
        y_test_gpu = torch.FloatTensor(load(ytest_filename)).cuda()
        
        # x_train_gpu = torch.FloatTensor(load('/home/rgb/Documents/Thesis_Git/effects-of-cpu-and-gpu-architectures-on-the-accuracy-of-neural-networks/Code/VGG-11/train_data.npy')).cuda()
        # y_train_gpu = torch.FloatTensor(load('/home/rgb/Documents/Thesis_Git/effects-of-cpu-and-gpu-architectures-on-the-accuracy-of-neural-networks/Code/VGG-11/train_label.npy')).cuda()
        # x_test_gpu = torch.FloatTensor(load('/home/rgb/Documents/Thesis_Git/effects-of-cpu-and-gpu-architectures-on-the-accuracy-of-neural-networks/Code/VGG-11/test_data.npy')).cuda()
        # y_test_gpu = torch.FloatTensor(load('/home/rgb/Documents/Thesis_Git/effects-of-cpu-and-gpu-architectures-on-the-accuracy-of-neural-networks/Code/VGG-11/test_label.npy')).cuda()

        #Store model on GPU
        #vgg_11 = VGG_11.cuda()
        #Create model class object and store model on GPU
        vgg11 = VGG_11()
        vggGpu = vgg11.cuda()

        #Divide the dataset into small batches'
        x_train_gpu = createBatches(x_train_gpu,batch_size, "cuda")
        y_train_gpu = createBatches(y_train_gpu,batch_size, "cuda")

        x_test_gpu = createBatches(x_test_gpu,batch_size, "cuda")
        y_test_gpu = createBatches(y_test_gpu,batch_size, "cuda")


# Test here 1

        #fileName = 'loss_values_' + str(i)+'.csv'
        #file2write=open(fileName,'w')

        criterion = useLossFunction()

        AccuracyOfIndividualClassesAndDataset(x_test_gpu,y_test_gpu,batch_size,vggGpu, "Before")

        # optimizer = useOptimizerFunction('SGD',vggGpu, learning_rate=learning_rate_val)
        optimizer = useOptimizerFunction('Adadelta',vggGpu, learning_rate=learning_rate_val)
        pltdata = trainNetworkOnGPU(max_epoch_num, x_train_gpu,y_train_gpu,optimizer,criterion,device,vggGpu, milestoneVal,gammaVal)


        AccuracyOfIndividualClassesAndDataset(x_test_gpu,y_test_gpu,batch_size,vggGpu, "After")

        print('Time required to run the model with all different LR on GPU is', datetime.now() - begin_time)


        # # Test here 2


        # # Adadelta Optimizer - GPU
        # criterion = useLossFunction()

        # AccuracyOfIndividualClassesAndDataset(x_test_gpu,y_test_gpu,batch_size,vggGpu, "Before")
        # optimizer=useOptimizerFunction('Adadelta',vggGpu, learning_rate=learning_rate_val)
        # trainNetworkOnGPU(max_epoch_num, x_train_gpu,y_train_gpu,optimizer,criterion,device,vggGpu,milestoneVal,gammaVal)
        # AccuracyOfIndividualClassesAndDataset(x_test_gpu,y_test_gpu,batch_size,vggGpu, "After")

        # # SGD Optimizer - GPU
        # criterion = useLossFunction()

        # AccuracyOfIndividualClassesAndDataset(x_test_gpu,y_test_gpu,batch_size,vggGpu, "Before")

        # optimizer=useOptimizerFunction('SGD',vggGpu, learning_rate=learning_rate_val)
        # trainNetworkOnGPU(max_epoch_num, x_train_gpu,y_train_gpu,optimizer,criterion,device,vggGpu,milestoneVal,gammaVal)

        # AccuracyOfIndividualClassesAndDataset(x_test_gpu,y_test_gpu,batch_size,vggGpu, "After")

        # # NAG optimzer - GPU
        # criterion = useLossFunction()

        # AccuracyOfIndividualClassesAndDataset(x_test_gpu,y_test_gpu,batch_size,vggGpu, "Before")

        # optimizer=useOptimizerFunction('NAG',vggGpu, learning_rate=learning_rate_val)
        # trainNetworkOnGPU(max_epoch_num, x_train_gpu,y_train_gpu,optimizer,criterion,device,vggGpu,milestoneVal,gammaVal)

        # AccuracyOfIndividualClassesAndDataset(x_test_gpu,y_test_gpu,batch_size,vggGpu, "After")


        #print('Time required to run the model on GPU is', datetime.now() - begin_time) # Time taken to run the experiment on GPU
        print('END')

if __name__ == "__main__":
    # execute only if run as a script
    main()