import pandas as pd
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import sqrt
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
random_seed(2020)
class BiGRUModel(nn.Module):
    def __init__(self):
        super(BiGRUModel, self).__init__()
        input_size = 8
        hidden_size = 128             
        self.batch1 = nn.BatchNorm1d(2000)
        self.batch2 = nn.BatchNorm1d(200)
        self.batch3 = nn.BatchNorm1d(100)        
        self.batch4 = nn.BatchNorm1d(52)
        self.batch5 = nn.BatchNorm1d(32)
        self.batch6 = nn.BatchNorm1d(8)
        self.lstm1= nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(128*2 ,2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)        
        self.conv1 = nn.Conv1d(in_channels=4,out_channels=2,kernel_size=2100)       
        self.w_omega = nn.Parameter(torch.Tensor(4410,4410))
        self.u_omega = nn.Parameter(torch.Tensor(4410,4410))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)    
        self.activation = nn.Sigmoid()
        self.fc3 = nn.Linear(4210, 2000)
        self.fc5 = nn.Linear(2000, 500)
        self.fc6 = nn.Linear(500, 200)
        self.fc7 = nn.Linear(200, 100)
        self.fc8 = nn.Linear(100, 52)
        self.fc9 = nn.Linear(52, 32)
        self.fc10 = nn.Linear(32, 8)
        self.fc11= nn.Linear(8, 2)
        self.fc4 = nn.Linear(2, 1)        
    def attention_net(self, x):
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score
        return scored_x
    def forward(self, data):
        data = torch.unsqueeze(data, -1).expand(-1, -1, 8)
        lstm1out,(_,_) = self.lstm1(data)
        lstm2out,(_,_) = self.lstm2(lstm1out)        
        lstm2out = lstm2out.permute(0, 2, 1)
        convout = self.conv1(lstm2out)
        convout = convout.reshape(convout.size(0),-1)
        atten_out = self.attention_net(convout)
        last_batch_size = atten_out.size(0)      
        if last_batch_size != 64:    
            atten_out = atten_out.reshape(last_batch_size, -1)
        else:
            atten_out = atten_out.reshape(64, -1)
        output = self.fc3(atten_out)        
        output = self.batch1(output)
        output = self.fc5(output)        
        output = self.fc6(output)
        output = self.batch2(output)        
        output = self.fc7(output)
        output = self.batch3(output)        
        output = self.fc8(output)
        output = self.batch4(output)        
        output = self.fc9(output)
        output = self.batch5(output)        
        output = self.fc10(output)
        output = self.batch6(output)        
        output = self.fc11(output)
        output = self.fc4(output)
        output = self.activation(output)
        return output
