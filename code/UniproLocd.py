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
class Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]
        return feature, label
def split_features_labels(data):
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    return features, labels
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    
    pred_list1 = []
    label_list1 = []

    for features, labels in tqdm(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)

        labels = torch.unsqueeze(labels, 1)
        labels = labels.float()


        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred_list1.extend(outputs.squeeze().cpu().detach().numpy())
        label_list1.extend(labels.squeeze().cpu().detach().numpy())
    score = cal_score(pred_list1, label_list1)
    return score
def test(model, test_loader, criterion, device):
    model.eval()
    pred_list2 = []
    label_list2 = []
    
    with torch.no_grad():
        for features, labels in tqdm(test_loader):
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            labels = torch.unsqueeze(labels, 1)
            labels = labels.float()
            
            outputs = torch.where(outputs >= 0.5, torch.tensor(1., device=device), torch.tensor(0., device=device))
            

            pred_list2.extend(outputs.squeeze().cpu().detach().numpy())
            label_list2.extend(labels.squeeze().cpu().detach().numpy())

        score = cal_score(pred_list2, label_list2)
        scheduler.step(score)

    return score
def vail(model, vail_loader, device):
    model.eval()
    outlist = []
    pred_list3 = []
    label_list3 = []
    
    
    with torch.no_grad():
        for features, labels in tqdm(vail_loader):
            features = features.to(device)
            labels = labels.to(device)


            outputs = model(features)
            labels = torch.unsqueeze(labels, 1)
            labels = labels.float()
            
            output = torch.where(outputs >= 0.382, torch.tensor(1., device=device), torch.tensor(0., device=device))
            
            

            pred_list3.extend(outputs.squeeze().cpu().detach().numpy())
            outlist.extend(output.squeeze().cpu().detach().numpy())
            label_list3.extend(labels.squeeze().cpu().detach().numpy())

        score = cal_score(pred_list3, label_list3)

    return score
batch_size = 64
num_epochs = 5
score = 0
esm1 = pd.read_csv('esm_output_pos.csv',header = None)
esm2 = pd.read_csv('esm_output_neg.csv',header = None)
esm = pd.concat([esm1, esm2], axis=0)
lables = pd.read_excel('lable.xlsx',header = None)
uni = pd.read_csv('unirep_feature.csv',header = None)
pro = pd.read_csv('protbert_feature.csv',header = None)
esm_reset = esm.reset_index(drop=True)
uni_reset = uni.reset_index(drop=True)
pro_reset = pro.reset_index(drop=True)
feature = pd.concat([esm_reset,uni_reset,pro_reset,lables], axis=1)    
features, labels = split_features_labels(feature)
esmin1 = pd.read_csv('esm_output_in_pos.csv',header = None)
esmin2 = pd.read_csv('esm_output_in_neg.csv',header = None)
esmin = pd.concat([esmin1, esmin2], axis=0)
inuni = pd.read_csv('XUunirep_feature.csv',header = None)
inpro = pd.read_csv('XUprotbert_feature.csv',header = None)
xu_lable = pd.read_excel('xu_lables.xlsx',header = None)
infeature = pd.concat([esmin, xu_lable], axis=0)
inesm_reset = esmin.reset_index(drop=True)
inuni_reset = inuni.reset_index(drop=True)
inpro_reset = inpro.reset_index(drop=True)
xu_lable_reset= xu_lable.reset_index(drop=True)
infeature = pd.concat([inesm_reset,inuni_reset,inpro_reset, xu_lable_reset], axis=1)    
val_features, val_labels = split_features_labels(infeature)
val_features = val_features.to_numpy()
val_labels = val_labels.to_numpy()
model = Model()
val_features = torch.tensor(val_features, dtype=torch.float)
val_labels = torch.tensor(val_labels, dtype=torch.long)
vail_dataset = Dataset(val_features, val_labels)
vail_loader = torch.utils.data.DataLoader(vail_dataset, batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
features = features.to_numpy()
labels = labels.to_numpy()
kf = KFold(n_splits=10, shuffle=True)
for fold, (train_indices, test_indices) in enumerate(kf.split(features, labels)):
    model = Model().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=5e-05)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=2, verbose=True)
    print(f"Fold: {fold+1}")
    train_features = [features[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]   
    test_features = [features[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]    
    train_features = torch.tensor(train_features, dtype=torch.float)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_features = torch.tensor(test_features, dtype=torch.float)
    test_labels = torch.tensor(test_labels, dtype=torch.long)    
    train_dataset = Dataset(train_features, train_labels)
    test_dataset = Dataset(test_features, test_labels)    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(20):
        train_score = train(model, train_loader, criterion, optimizer, device)
        print("train:",train_score)
        test_score = test(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        print("test:",test_score)
        
        if test_score > score:
            score = test_score
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, "model.pth")
print(score)

##test##
model.eval()
best_model_state_dict = torch.load("model.pth")

model.load_state_dict(best_model_state_dict)

model = model.to(device)

vail_score = vail(model, vail_loader , device)
print("vail:",vail_score)
