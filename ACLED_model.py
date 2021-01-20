from model import Rencoder,Rdecoder
import torch
import torch.nn as nn
from argumentparser import ArgumentParser
from data_processing import train_loader,eval_loader
arg = ArgumentParser()
class ACLED(nn.Module):

    def __init__(self,encoder,decoder):
        super(ACLED, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x_data):
        encoder_out,code = self.encoder(x_data)
        output = self.decoder(encoder_out)
        return output,code


def train(model,data_loader, num_epoch,opt,loss_fn,best_w=1):
    for epoch in range(num_epoch):
        model.train()
        for it, data in enumerate(data_loader):

            (m_pred,_),_ = model(data)
            opt.zero_grad()
            loss=loss_fn(data,m_pred)

            loss.backward()

            opt.step()

            if it%1000 ==0:
                print('Epoch',epoch,'iteration',it,'loss',loss.item())

            if loss.item()<best_w:
                best_w = loss.item()
                torch.save(model.state_dict(), './model/ACLED_MODEL.pkl')

def eval(model,loss_fn,data_loader,max=0):
    model.eval()
    for it, data in enumerate(data_loader):
        m_pred, _ = model(data)
        loss = loss_fn(data, m_pred)
        if loss.item() > max:
            max = loss.item()
    return max
encoder = Rencoder(window=arg.window,
                       hidden_size=arg.hidden_size,
                       num_size=arg.num_size,
                       batch_size=arg.batch_size,
                       code_size=arg.code_size)
encoder_test = Rencoder(window=arg.window,
                       hidden_size=arg.hidden_size,
                       num_size=arg.num_size,
                       batch_size=1,
                       code_size=arg.code_size)

decoder = Rdecoder(window=arg.window,
                   hidden_size=arg.hidden_size,
                   num_size=arg.num_size)


model_a = ACLED(encoder, decoder)
model_test = ACLED(encoder_test,decoder)
if __name__=="__main__":

    optim =torch.optim.Adam(model_a.parameters(), lr=arg.lr, betas=(0.5, 0.999))

    loss_fn = nn.MSELoss()

    train(model_a,train_loader,arg.num_epoch,optim,loss_fn)

    eval(model_a, loss_fn, eval_loader)