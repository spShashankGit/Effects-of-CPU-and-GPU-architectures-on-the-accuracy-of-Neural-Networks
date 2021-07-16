import json    
# reading files
f1 = open("/root/Experiment/effects-of-cpu-and-gpu-architectures-on-the-accuracy-of-neural-networks/requirements_gpuInfoLogger1.txt", "r")  
f2 = open("/root/Experiment/effects-of-cpu-and-gpu-architectures-on-the-accuracy-of-neural-networks/pipFreeze.txt", "r")  
referencePackageList = {}  
i = 0
  
for line1 in f1:
    tmpHolder = line1.split('=')
    if(len(tmpHolder)>1):
        tmpDict={'packageName':tmpHolder[0], 'version':tmpHolder[1]}
        referencePackageList[i]=(tmpDict)
        i += 1  
    
    # print('Splitted msg ', referencePackageList)
    with open('package.json', 'w') as fp:
        json.dump(referencePackageList, fp)

      
    # for line2 in f2:
          
    #     # matching line1 from both files
    #     if line1 == line2:  
    #         # print IDENTICAL if similar
    #         print("Line ", i, ": IDENTICAL")       
        # else:
        #     print("Line ", i, ":")
        #     # else print that line from both files
        #     print("\tFile 1:", line1, end='')
        #     print("\tFile 2:", line2, end='')
        # break

print(referencePackageList.values())
for line2 in f2:
    tmpHolder2 = line1.split('=')
    for key, value in referencePackageList.items():
    	# print("Items:", key,value)
        pass
    	# for k, v in value.items():
    	# 	print(k + ":", value[k])

# closing files
f1.close()                                       
f2.close()      
