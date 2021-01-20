from model import KMediod
import numpy as np
from ACLED_model import model_test
import torch
from data_processing import num_data_test,test_loader
from test import test
from utils import mscatter,hanmingdis,get_median
from argumentparser import ArgumentParser
import mpl_toolkits.axisartist as axisartist
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
arg = ArgumentParser()
def templates(outlier_it):
    anom_data1=[]
    for it in outlier_it:
        data_y = num_data_test[it:it+arg.window, :]
        data_y = np.array(data_y)

        data_y = np.cov(data_y.T)
        list_y = []
        for i in range(data_y.shape[0]):
            for j in range(data_y.shape[0]):
                if j >= i:
                    list_y.append(data_y[i][j])
        num_y = np.array(list_y)
        anom_data1.append(num_y)

    anom_data1 = np.array(anom_data1)
    pca = PCA(n_components=0.9)
    anom_data = pca.fit_transform(anom_data1)
    y_pred = KMediod(anom_data,n_points=4, k_num_center=arg.num_cluster).run()
    colors = ['navy', 'darkgreen', 'darkred', 'orangered', 'purple', 'darkcyan', 'deeppink', 'peru', 'grey']
    c = list(map(lambda x: colors[x], y_pred))
    m = {0: 'd',1: 'o', 2: 's', 3: 'D', 4: '+',5: 'x',6: 'p',7: 'v',8: '4',-1: '*'}
    cm = list(map(lambda x: m[x], y_pred))
    RS = 20210101
    X = TSNE(random_state=RS).fit_transform(anom_data1)
    fig = plt.figure()
    ax = axisartist.Subplot(fig,111)
    fig.add_axes(ax)
    ax.axis["bottom"].set_axisline_style("->", size=1.5)
    ax.axis["left"].set_axisline_style("->", size=1.5)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)
    plt.xticks([])
    plt.yticks([])
    mscatter(X[:,0]/100, X[:,1]/100, s=75, c='', m=cm, alpha=0.4, linewidths=3, edgecolors=c)
    plt.show()
    return y_pred
def binary_code(data):
    model_test.load_state_dict(torch.load('./model/ACLED_MODEL.pkl'))
    model_test.eval()
    with torch.no_grad():
        _,binary_code = model_test(data)
        binary_code = binary_code.numpy().flatten()
    return binary_code


def similarity_search(y_pred,loss_sum,it_sum1,outlier_it):
    dict ={}
    num_list = {}
    for i in range(arg.num_cluster):
        num_list['y_%d'%i]=[]
        for j, number in enumerate(y_pred):
            if number == i:
                num_list['y_%d'%i].append(j)

    for i in range(arg.num_cluster):
        anomaly_index =[]
        for j in num_list['y_%d'%i]:
            anomaly_index.append(loss_sum[j])
        x = get_median(anomaly_index)
        x_i = it_sum1[x]
        data_anomaly = num_data_test[x_i:x_i + arg.window, :]
        score = "similar_score_%d" % i
        dict[score] = {}
        for j in outlier_it:
            data_y = num_data_test[j:j + arg.window, :]
            dict[score][str(j)] = [hanmingdis(binary_code(torch.from_numpy(data_anomaly).float().view(arg.window, -1)),
                                                  binary_code(torch.from_numpy(data_y).float().view(arg.window, -1))), i]

    y_label = []
    zm = []
    for i in range(arg.num_cluster):
        zm.append(dict["similar_score_%d" % i])
    for i in outlier_it:
        label_it = []
        for j in range(arg.num_cluster):
            label_it.append(dict["similar_score_%d" % j][str(i)][0])
        label = min(label_it)
        label_index = label_it.index(label)
        y_label.append(zm[label_index][str(i)][1])
    return y_label


if __name__=="__main__":
    loss_sum,it_sum1,outlier_it = test(test_loader)

    y_pred = templates(outlier_it)

    print(similarity_search(y_pred, loss_sum, it_sum1, outlier_it))