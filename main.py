import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# load data from csv files
directory = '/home/user/바탕화면/AVATAR_OFT_WT'
arr = os.listdir(directory)
alldata = np.empty((2000,27,1))
print(alldata.shape)
for file in arr:
    df = pd.read_csv(directory+"/"+file)
    data_np = df.to_numpy()
    dataset = data_np[0:2000,:]
    for i in range(1,4):
        data_block = data_np[i*2000:(i+1)*2000,:]

        dataset = np.dstack([dataset,data_block])
    if(dataset.shape == (2000,27,4)):
        alldata = np.append(alldata, dataset, axis=2)
