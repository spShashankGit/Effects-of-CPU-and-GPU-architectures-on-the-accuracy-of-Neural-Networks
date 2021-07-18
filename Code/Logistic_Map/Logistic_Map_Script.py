from pypads.app.base import PyPads                                                  # Import pypads
import pandas as pd                                                                 # Import pandas
from datetime import datetime                                                       # Import datetime to calculate the final time of execution
import torch                                                                        # Import PyTorch

tracker = PyPads( autostart=True)
tracker.start_track(experiment_name="Effect of GPUs - Logistic map - stg7")
#PyPads(autostart=True)

# Initializing variables
x_init = 0.4                                                                        # Initial valur of the X or X_0
max_iteration = 2000                                                                # Max number of iterations
r = 3.7                                                                             # Rate of reproduction
seed_value_list = [7184,13474,32889,56427,59667]                                       # Seed values selected randomly between 0 and 65536

## ---------------------------------------------------------------- CPU -----------------------------------------------------------------##

score_cpu = pd.DataFrame(columns =['Iter','Xn', 'r', 'Xn+1'],dtype=float)           # Dataframe to store CPU Xn and Xn+1

# Function to calculate X_new from X_old
def logisticMapCPU(x,score):

    res = score.copy()
    k = True                                                                        # Check if convergence point has occcured
    
    for i in range (max_iteration):
        
        xNew = r*x*(1-x)

        rowData = pd.DataFrame([[i,x,r,xNew]], columns=['Iter','Xn', 'r', 'Xn+1'],dtype=float)
        res = res.append(rowData,ignore_index=True)

        if(xNew == x and k):
            print("Convergence point is at epoch", i, "and the value is",x)
            k=False
        x = xNew
    return res


## ---------------------------------------------------------------- Torch -----------------------------------------------------------------##

def logisticMapPyTorch(x, res):
    k = True                                                                         # Check if convergence point has occcured

    for i in range (max_iteration):

        res[i] = r*x*(1-x)

        if(res[i] == x and k):
            print("Convergence point is at epoch", i, "and the value is",x)
            k=False
        x = res[i]
    return res


for i,seedVal in enumerate(seed_value_list):


    score_cpu = logisticMapCPU(x_init,score_cpu)
    print('CPU ', score_cpu)

    accuracyFileName = "CPU_Logistic_Map" + str(datetime.now())+'.csv'
    #file2write=open(accuracyFileName,'w')
    csvData = score_cpu.to_csv(accuracyFileName,index=False)
    #file2write.write(csvData)
    #file2write.close()


    # If CUDA device is present, print the name of the device and set CUDA default device
    if (torch.cuda.is_available()):

        # Setting the seed value before running the experiment on GPU
        torch.manual_seed(seedVal)
        torch.cuda.manual_seed(seedVal)
        torch.cuda.manual_seed_all(seedVal)

        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        cuda = torch.device('cuda')                                                     # Setting default CUDA device

        score_pytorch_gpu = torch.empty((max_iteration,1),dtype=torch.float32)          # Initialize Tensor on GPU

        score_pytorch_gpu = logisticMapPyTorch(x_init, score_pytorch_gpu)               # Calling logisticMapPyTorch function with initial values

        print('GPU ',score_pytorch_gpu)

        accuracyFileName = "GPU_Logistic_Map" + str(datetime.now())+'.csv'
        gpu_Results = score_pytorch_gpu.numpy()
        gpu_Results = pd.DataFrame(gpu_Results)
        gpu_Results.to_csv(accuracyFileName)