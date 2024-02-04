import torch
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
import esm
import numpy as np
def read_fasta_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        protein_name = None
        sequence = []

        for line in lines:
            line = line.strip()
            if line.startswith('>'):

                if protein_name is not None:
                    data.append((protein_name, ''.join(sequence)))
                protein_name = line[1:]  
                sequence = [] 
            else:

                sequence.append(line)


        if protein_name is not None:
            data.append((protein_name, ''.join(sequence)))

    return data


file_path1 = '/kaggle/input/dataset/pos.fasta'
file_path2 = '/kaggle/input/dataset/neg.fasta'
data1 = read_fasta_file(file_path1)
data2 = read_fasta_file(file_path2)
data = data1 + data2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()  
feature = []
for index, value in enumerate(data):
    value = [(value)]
    batch_labels, batch_strs, batch_tokens = batch_converter(value)
    batch_tokens = batch_tokens.to(device)  # Move data to GPU
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
        

    print(sequence_representations)
    feature.append(sequence_representations)
