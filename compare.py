#importing files
import pandas as pd
import subprocess
import sys
import os

# reading files
f1 = open("/root/Experiment/effects-of-cpu-and-gpu-architectures-on-the-accuracy-of-neural-networks/requirements_gpuInfoLogger1.txt", "r")  
f2 = open("/root/Experiment/effects-of-cpu-and-gpu-architectures-on-the-accuracy-of-neural-networks/pipFreeze.txt", "r")  

#color coding the terminal output
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def createDictionaryObject(fileName):
    i = 0
    packageList=[]
    for fileLine in fileName:
        tmpHolder = fileLine.split('==')
        if(len(tmpHolder)>1):
            tmpDict=[]
            tmpDict.append(str.strip(tmpHolder[0]))
            tmpDict.append(str.strip(tmpHolder[1]))
            packageList.append(tmpDict)
            i += 1  
    return packageList

def upgrade(package,version):
    try:
        pkgWithVersion = package + "==" + version
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkgWithVersion])
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command 123 '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

def install(package,version):
    try:
        pkgWithVersion = package + "==" + version
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkgWithVersion])
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command 123 '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

def uninstall(package):
    try:
        pkgWithVersion = package
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", pkgWithVersion])
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command 123 '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

refDict = createDictionaryObject(f1)
underVeriDict = createDictionaryObject(f2)

# Converting data dictionary to data frame
refDict = pd.DataFrame(refDict)
underVeriDict = pd.DataFrame(underVeriDict)


try:
    packageIndex = list(refDict[0]).index('appdirs',0)
except ValueError:
    packageIndex = -1

packageIndex


for j in range(len(underVeriDict)):
    try:
        packageIndex = list(refDict[0]).index(underVeriDict[0][j],0)
    except ValueError:
        packageIndex = -1


compatiblePackages = []
conflictingPackages = []
packageNotFound = []

for j in range(len(underVeriDict)):
    try:
        packageIndex = list(refDict[0]).index(underVeriDict[0][j],0)
    except ValueError:
        packageIndex = -1

    if(packageIndex == -1):
        packageNotFound.append([underVeriDict[0][j],underVeriDict[1][j]])
    else:
        #check if the version is same
        
        #if same append it to the compatiblePackages 
        if(refDict[1][packageIndex]==underVeriDict[1][j]):
            #print('Same version ', underVeriDict[0][j], "\t",underVeriDict[1][j])
            compatiblePackages.append(underVeriDict[0][j])

        elif(refDict[1][packageIndex]!=underVeriDict[1][j]):
            conflictingPackages.append([underVeriDict[0][j],refDict[1][packageIndex],underVeriDict[1][j]])

        #if not append it to conflictingPackages and also mention the difference in the driver versions
        #before ending the code, print - compatible, conflicting and not found packages
        

print("Total", len(compatiblePackages), "packages matches with the requirement")



print(f"{bcolors.WARNING}Warning: total", len(conflictingPackages), "packages did not match with the requirement")
print(f"{bcolors.WARNING}Package Name \t\t Expected Version \t\t Actual Version")
for i in conflictingPackages:
    print("%-*s  %-*s  %s"%((23,i[0], 30,i[1], i[2])))


print(f"{bcolors.FAIL}Warning: total", len(packageNotFound), "packages were found on the local machine but are not on the reference list")
print(f"\n{bcolors.FAIL}Package Name \t\t\t\t\t   Version{bcolors.ENDC}")
for i in packageNotFound:
    print("%-*s %-*s"%((50,i[0],20,i[1])))


## write code to reinstall the packages with the correct version
print("Reinstalling....")

for i in conflictingPackages:
    print(" %-s version %-s"%((i[0],i[1])))
    upgrade(i[0],i[1])

## write code to remove the packages which are found extra on the local machine.abs
print("Removing packages that are not extra..")
for i in packageNotFound:
    print(" %-s version %-s"%((i[0],i[1])))
    uninstall(i[0])


#Closing the file
f1.close()                                       
f2.close()
