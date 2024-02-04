from tape import UniRepModel, TAPETokenizer
import time
import time
import pandas as pd
import torch
import warnings
 
warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from Bio import SeqIO

def read_fasta_file(filename):
    seq_dict = {}
    for record in SeqIO.parse(filename, "fasta"):
        seq_dict[record.id] = str(record.seq)
    return seq_dict
  def UniRep_Embed(input_seq):
    T0=time.time()
    UNIREPEB_=[]
    PID = []
    print("UniRep Embedding...")

    model = UniRepModel.from_pretrained('babbler-1900')
    model=model.to(DEVICE)
    tokenizer = TAPETokenizer(vocab='unirep')

    for key,value in input_seq.items():
        PID.append(key)
        sequence = value
        # print(sequence)
        if len(sequence) == 0:
            print('# WARNING: sequence', PID, 'has length=0. Skipping.', file=sys.stderr)
            continue
        with torch.no_grad():
            token_ids = torch.tensor([tokenizer.encode(sequence)])
            token_ids = token_ids.to(DEVICE)
            output = model(token_ids)
            unirep_output = output[0]
            unirep_output=torch.squeeze(unirep_output)
            unirep_output= unirep_output.mean(0)
            unirep_output = unirep_output.cpu().numpy()
            UNIREPEB_.append(unirep_output.tolist())
    unirep_feature=pd.DataFrame(UNIREPEB_)
    col=["UniRep_F"+str(i+1) for i in range(0,1024)]
    unirep_feature.columns=col
    unirep_feature=pd.concat([unirep_feature],axis=1)
    unirep_feature.index=PID
    
    print("Getting Deep Representation Learning Features with UniRep is done.")
    print("it took %0.3f mins.\n"%((time.time()-T0)/60))

    return unirep_feature
