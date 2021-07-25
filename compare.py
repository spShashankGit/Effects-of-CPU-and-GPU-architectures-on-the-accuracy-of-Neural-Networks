# Importing files
import pandas as pd
import subprocess
import sys
import os

# Reading files
dirname = os.path.dirname(__file__)
referenceFilePath = 'pipFreezeZaire.txt'
fullReferenceFilePath = os.path.join(dirname, referenceFilePath)
f1 = open(fullReferenceFilePath, "r")  

localFilePath = 'pipFreezeLocal.txt'
fullLocalFilePath = os.path.join(dirname, localFilePath)
f2 = open(fullLocalFilePath, "r")  

# Color coding the terminal output
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

compatiblePackages = []
conflictingPackages = []
packageNotFound = []

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
        subprocess.check_call([sys.executable, "-m", "pip3", "install", "--upgrade", pkgWithVersion])
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

def install(package,version):
    try:
        pkgWithVersion = package + "==" + version
        subprocess.check_call([sys.executable, "-m", "pip3", "install", pkgWithVersion])
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

def uninstall(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip3", "uninstall", package])
    
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

def checkIfThePackagesAreSame():
    refDict = createDictionaryObject(f1)
    underVeriDict = createDictionaryObject(f2)

    # Converting data dictionary to data frame
    refDict = pd.DataFrame(refDict)
    underVeriDict = pd.DataFrame(underVeriDict)


    for j in range(len(underVeriDict)):
        # Check if the local package is available in the reference list.
        try:
            packageIndex = list(refDict[0]).index(underVeriDict[0][j],0)
        except ValueError:
            packageIndex = -1

        if(packageIndex == -1):
            packageNotFound.append([underVeriDict[0][j],underVeriDict[1][j]])
        else:
            # Check if the version is same
            # If same append it to the compatiblePackages list
            # Else append it to conflictingPackages list
            if(refDict[1][packageIndex]==underVeriDict[1][j]):
                compatiblePackages.append(underVeriDict[0][j])

            elif(refDict[1][packageIndex]!=underVeriDict[1][j]):
                conflictingPackages.append([underVeriDict[0][j],refDict[1][packageIndex],underVeriDict[1][j]])

    return packageNotFound,compatiblePackages,conflictingPackages 


# Check the status of the packages
packageNotFound,compatiblePackages,conflictingPackages = checkIfThePackagesAreSame()


# Print all matched packages.
print("Total", len(compatiblePackages), "packages matches with the requirement")


# Print packages which have different version than expected
if(len(conflictingPackages) >=1):

    print("Warning: total", len(conflictingPackages), "packages did not match with the requirement")
    print(f"{bcolors.WARNING}Package Name \t\t Expected Version \t\t Actual Version {bcolors.ENDC}")
    for i in conflictingPackages:
        print("%-*s  %-*s  %s"%((23,i[0], 30,i[1], i[2])))
else:
    print("No conflicting package versions found!")


# Print extra packages 
if(len(packageNotFound)>=1):
    print(f"{bcolors.FAIL}Warning: total", len(packageNotFound), "packages were found on the local machine but are not on the reference list {bcolors.ENDC}")
    print(f"\n{bcolors.FAIL}Package Name \t\t\t\t\t   Version {bcolors.ENDC}")
    for i in packageNotFound:
        print("%-*s %-*s"%((50,i[0],20,i[1])))
else:
    print("No Extra packages found!")


# Reinstall the packages with the correct version
if(len(conflictingPackages)>=1):
    print("Reinstalling..")
    for i in conflictingPackages:
        print(" %-s version %-s"%((i[0],i[1])))
        upgrade(i[0],i[1])


# Removing packages which are found extra on the local machine
if(len(packageNotFound)>=1):
    print("Removing packages that are extra..")
    for i in packageNotFound:
        print(" %-s version %-s"%((i[0],i[1])))
        #uninstall(i[0])

# Confirm all packages are installed correctly and environment is safe to run the experiment!
packageNotFound,compatiblePackages,conflictingPackages = checkIfThePackagesAreSame()

if(len(packageNotFound)==0 and len(compatiblePackages)==0 and len(conflictingPackages)==0):
    print("Environment is configured correctly.")
    print("Environment is ready.")

else:
    print("Environment is still not ready.")
    print("Hint: run 'python3 compare.py' again in terminal")

# Closing the file
f1.close()                                       
f2.close()
