import numpy as np
import torch
import torch.nn as nn
import random

class Rencoder(nn.Module):
    def __init__(self,window,hidden_size,batch_size,num_size,code_size):
        super(Rencoder, self).__init__()
        self.batch_size = batch_size
        self.num_size = num_size
        self.window = window
        self.hidden_size = hidden_size
        self.code_size = code_size
        self.cnn1 = nn.Conv2d(1, 32, 3, padding=1)
        self.cnn2 = nn.Conv2d(32, 64, 3, padding=1)
        self.cnn3 = nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.cnn4 = nn.ConvTranspose2d(32, 1, 3, padding=1)
        self.relu = nn.SELU()
        self.rnn = nn.LSTM(window,
                           hidden_size,
                           num_layers=2,
                           bidirectional=True,
                           dropout=0.2,
                           batch_first=True)
        self.batnorm = nn.LayerNorm([num_size,2*hidden_size])
        self.A = torch.randn((batch_size,window,2*hidden_size),requires_grad=True)
        self.B = torch.randn((batch_size,window,window), requires_grad=True)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(self.num_size*self.hidden_size*2,self.code_size)

    def forward(self,data):
        data1 = torch.reshape(data, [self.batch_size, 1, self.num_size, self.window])
        x = self.cnn1(data1)
        x = self.relu(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.cnn3(x)
        x = self.relu(x)
        x = self.cnn4(x)
        x = self.relu(x)
        data= torch.reshape(data,[self.batch_size, self.num_size, self.window])
        x = torch.reshape(x, [self.batch_size, self.num_size, -1]) + data
        self.attention_w = []
        out, newhid = self.rnn(x)
        out = torch.reshape(out,[self.num_size,2*self.hidden_size,-1])
        for i, eh in enumerate(out):
            eh = torch.reshape(eh,[self.batch_size,2*self.hidden_size,-1])
            self.attention_w.append(torch.bmm(self.A,eh))
        self.attention_w = torch.stack(self.attention_w)
        self.attention_w = torch.reshape(self.attention_w,[self.batch_size,self.num_size,-1])
        x_1 = torch.reshape(x,(self.batch_size,self.window,self.num_size))
        self.attention_w = self.attention_w + torch.bmm(self.B,x_1).reshape(self.batch_size,self.num_size,-1)
        self.attention_w = self.tanh(self.attention_w)
        self.attention_w = nn.Softmax(dim=1)(self.attention_w)
        y_data = torch.reshape(x, [self.batch_size, self.num_size, -1])
        out, _ = self.rnn(torch.mul(y_data, torch.Tensor(self.attention_w)))
        out = self.batnorm(out)
        code = self.linear(out.reshape(self.batch_size,-1))
        code = self.tanh(code)
        binary_code = torch.sign(code)

        return out , binary_code

class Rdecoder(nn.Module):
    def __init__(self,window,hidden_size,num_size):
        super(Rdecoder, self).__init__()
        self.window = window
        self.hidden_size = hidden_size
        self.num_size = num_size
        self.rnn = nn.LSTM(2*hidden_size,
                           2*hidden_size,
                           num_layers=2,
                           dropout=0.2,
                           batch_first=True)
        self.out = nn.Linear(2*hidden_size,64)
        self.out2 = nn.Linear(64,window)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.SELU()
        self.batnorm = nn.BatchNorm1d(num_size)
        self.tanh = nn.Tanh()

    def forward(self, y_data):
        self.attention_w = []
        eout = y_data
        out, newhid = self.rnn(y_data)
        eout = torch.reshape(eout,[self.num_size,-1,2*self.hidden_size])
        out = torch.reshape(out,[self.num_size,-1,2*self.hidden_size])
        for i,eh in enumerate(eout):
            self.attention_w.append(torch.Tensor(out[i]).mul(torch.Tensor(eh)))
        self.attention_w = self.tanh(torch.stack(self.attention_w))
        self.attention_w = nn.Softmax(dim=0)(self.attention_w)
        y_data=torch.reshape(y_data,[self.num_size,-1,2*self.hidden_size])
        out,_ = self.rnn(torch.mul(y_data,torch.Tensor(self.attention_w)))
        out = self.relu(out)
        out = torch.reshape(out, [-1, self.num_size, 2*self.hidden_size])
        out = self.out(out)
        out = self.dropout(out)
        out = self.batnorm(out)
        out = self.relu(out)
        out = self.out2(out)
        out = self.relu(out)
        return out,self.attention_w


class KMediod():

    def __init__(self,data, n_points, k_num_center):
        self.n_points = n_points
        self.k_num_center = k_num_center
        self.data = data

    def ou_distance(self, x, y):

        return np.sqrt(sum(np.square(x - y)))

    def run_k_center(self, func_of_dis):

        indexs = list(range(len(self.data)))
        random.shuffle(indexs)
        init_centroids_index = indexs[:self.k_num_center]
        centroids = self.data[init_centroids_index, :]
        sample_target = []
        if_stop = False
        while (not if_stop):
            if_stop = True
            classify_points = [[centroid] for centroid in centroids]
            sample_target = []
            for sample in self.data:
                distances = [func_of_dis(sample, centroid) for centroid in centroids]
                cur_level = np.argmin(distances)
                sample_target.append(cur_level)
                classify_points[cur_level].append(sample)
            for i in range(self.k_num_center):
                distances = [func_of_dis(point_1, centroids[i]) for point_1 in classify_points[i]]
                now_distances = sum(distances)
                for point in classify_points[i]:
                    distances = [func_of_dis(point_1, point) for point_1 in classify_points[i]]
                    new_distance = sum(distances)

                    if new_distance < now_distances:
                        now_distances = new_distance
                        centroids[i] = point
                        if_stop = False

        return sample_target

    def run(self):

        predict = self.run_k_center(self.ou_distance)
        return predict



