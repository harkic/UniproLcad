from tape import UniRepModel, TAPETokenizer
import time
import pandas as pd
import torch
import warnings 
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
import warnings
import re
import numpy as np
import pandas as pd
import joblib
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import math
from sklearn.preprocessing import LabelBinarizer
warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
model = model.to(device)
model = model.eval()
from Bio import SeqIO
fasta_sequences_pos = SeqIO.parse(open('/kaggle/input/dataset/pos.fasta'),'fasta')
fasta_sequences_neg = SeqIO.parse(open('/kaggle/input/dataset/neg.fasta'),'fasta')
proBert_seq = []
trainlabel = []
for fasta in fasta_sequences_pos:
    sequence = str(fasta.seq)
    label = 1
    proBert_seq.append(sequence)
    trainlabel.append(label)
for fasta in fasta_sequences_neg:
    sequence = str(fasta.seq)
    label = 0
    proBert_seq.append(sequence)
    trainlabel.append(label)
proBert_seq = np.array(proBert_seq)
trainlabel = np.array(trainlabel)
proBer_train_seq = [" ".join(sequence) for sequence in proBert_seq]
proBer_train_seq = list(filter(None,proBer_train_seq))
all_protein_features = []
for i, seq in enumerate(proBer_train_seq):

    ids = tokenizer1.batch_encode_plus([seq], add_special_tokens=True,
                                      pad_to_max_length=True)  # encode_plus返回1、词的编码，2、
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    seqments_tensor = torch.tensor(ids["token_type_ids"]).to(device)

    with torch.no_grad():
        mbedding = model1(input_ids=input_ids, token_type_ids=seqments_tensor, attention_mask=attention_mask)[0]  # 二维的张量，获取模型最后一层的输出结果
    embedding = embedding.cpu().numpy()  # 因为model是在gpu将其放在CPU上调 embeddnig转为数组
    features = []
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len - 1]
        features.append(seq_emd)

    #         all_protein_features.append(seq_emd)
    output = model2(token_ids)
    unirep_output = output[0]
    unirep_output=torch.squeeze(unirep_output)
    unirep_output= unirep_output.mean(0)
    unirep_output = unirep_output.cpu().numpy()
    
    
    all_protein_features += features  # 把一条一条的特征加到一块

# 把所有提取到的特征放到DataFrame里
sum_features = []
for element in all_protein_features:
    sum_features.append(element.sum(axis=0))  # np.mean(axis=0) protbert的特征可以用加的和平均的
protBertfeature = pd.DataFrame(sum_features)

#################################
labels = trainlabel
#######################
all_labels = pd.DataFrame(labels)
print(protBertfeature.shape)

all_data = pd.concat((protBertfeature, all_labels), axis=1)

with open("./XUprotbert_feature.csv", "w") as pf:
    all_data.to_csv(pf, index=False, header=False)
