import torch
import torch.nn as nn
from ACLED_model import model_test
from data_processing import num_data_test,test_loader,label_1
from sklearn.metrics import f1_score,recall_score,precision_score
from utils import font
from argumentparser import ArgumentParser
import matplotlib.pyplot as plt
loss_fn = nn.MSELoss()
arg = ArgumentParser()
def test(dataloader,threshold = 1):
    model_test.load_state_dict(torch.load('./model/ACLED_MODEL.pkl'))
    model_test.eval()
    with torch.no_grad():
        loss_sum = []
        loss_sum1 = []
        it_sum = []
        outlier_it = []
        anom_data = []
        label_2 = []
        it_sum1 = {}
        for it, data in enumerate(dataloader):
            (m_pred,_),_ = model_test(data)
            loss = loss_fn(data, m_pred)
            loss_sum.append(loss.item())
            it_sum.append(it)
            if loss<threshold:
                label_2.append(0)
            if loss >threshold:
                label_2.append(1)
                print(it,loss,'anomaly MTS segment')
                outlier_it.append(it)
                anom_data.append(num_data_test[it,:])
                loss_sum1.append(loss.item())
                it_sum1[loss.item()]=it
    print(f1_score(label_1, label_2),precision_score(label_1, label_2),recall_score(label_1, label_2))
    outlier_it
    plt.xlabel('Test Time', font)
    plt.ylabel('Anomaly Scores', font)
    plt.plot(it_sum, loss_sum, linewidth=2)
    plt.xticks(fontsize=16,weight='bold')
    plt.yticks(fontsize=16,weight='bold')
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.ylim(-0.5,6)
    plt.xlim(0,1000)
    plt.axhline(y=1, ls=":", c="red")
    for i in outlier_it:
        plt.axvline(x=i, ls="-", c="deeppink",alpha=0.1)
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.show()
    return loss_sum1,it_sum1,outlier_it

if __name__=="__main__":
    test(test_loader)