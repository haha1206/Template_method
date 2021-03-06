from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from sklearn import preprocessing
from argumentparser import ArgumentParser
import numpy as np
arg = ArgumentParser()
class dectdataset(Dataset):
    def __init__(self,x_data):
        self.len = x_data.shape[0]
        self.x_data = torch.from_numpy(x_data).float().view(x_data.shape[0],x_data.shape[1])

    def __getitem__(self, index):
        data = self.x_data[index:index+arg.window,:]
        if data.shape[0] == arg.window:
            data = torch.reshape(data,[arg.num_size,arg.window])
        else:
            data = torch.zeros(arg.num_size,arg.window)
        return data

    def __len__(self):
        return self.len

if arg.dataset=="air_quality":
    num_data = pd.read_csv("./data/air_quality_data.csv")
    num_data = num_data.dropna(axis=0)
    num_data = num_data._get_numeric_data()
    num_data.astype('float')
    num_data = preprocessing.scale(num_data)
    num_data_train = num_data[:int(num_data.shape[0] * 0.9)]
    num_data_eval = num_data[int(num_data.shape[0] * 0.9):]
    arg.num_size = num_data.shape[1]
    num_data_test = pd.read_csv("./data/air_data_test.csv")
    label_1 = np.array(num_data_test['label'])
    num_data_test = num_data_test._get_numeric_data()
    num_data_test.astype('float')
    num_data_test = preprocessing.scale(num_data_test)
    num_data_test = num_data_test[:, :-1]
    dataset = dectdataset(num_data_train)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=arg.batch_size,
                              drop_last=True,
                              shuffle=True)
    dataset_eval = dectdataset(num_data_eval)
    eval_loader = DataLoader(dataset=dataset_eval,
                             batch_size=arg.batch_size,
                             drop_last=True,
                             shuffle=True)
    dataset_test = dectdataset(num_data_test)
    test_loader = DataLoader(dataset=dataset_test,
                             batch_size=1,
                             drop_last=True,
                             shuffle=False)
    if arg.dataset =="energydata":
        num_data = pd.read_csv("./data/data/energydata_complete.csv")
        num_data = num_data.dropna(axis=0)
        num_data = num_data._get_numeric_data()
        num_data.astype('float')
        num_data = preprocessing.scale(num_data)
        num_data_train = num_data[:int(num_data.shape[0]*0.9)]
        num_data_eval = num_data[int(num_data.shape[0] * 0.9):]
        arg.num_size = num_data.shape[1]
        dataset = dectdataset(num_data_train)
        train_loader = DataLoader(dataset=dataset,
                                  batch_size=arg.batch_size,
                                  drop_last=True,
                                  shuffle=True)
        dataset_eval = dectdataset(num_data_eval)
        eval_loader = DataLoader(dataset=dataset_eval,
                                  batch_size=arg.batch_size,
                                  drop_last=True,
                                  shuffle=True)
