# UniproLocd
Title:Predicting Antimicrobial Peptides: UniproLocd's Fusion of Protein Language Models and Neural Networks.
Antimicrobial peptides (AMPs) are vital components of innate immunotherapy, and computational methods have become essential for predicting them. Existing approaches mainly rely on either deep learning for automatic extraction of sequence features or traditional manual amino acid features combined with machine learning. Recently, the advent of large language models has significantly boosted the representational power of sequence features. In light of this, we present a novel AMP predictor called UniproLcad, which integrates three prominent protein language models—ESM, ProtBert, and Unirep. UniproLcad also leverages deep learning networks, including Long and Short Memory Network (LSTM) and One-Dimensional Convolutional Neural Networks (1D-CNN), and incorporating an attention mechanism. These deep learning frameworks, coupled with pre-trained language models, efficiently extract multi-view features from antimicrobial peptide sequences and assign attention weights to them. Through ten-fold cross-validation and independent testing, UniproLcad demonstrates competitive performance in the field of antimicrobial peptide recognition. This integration of diverse language models and deep learning architectures enhances the accuracy and reliability of predicting antimicrobial peptides, contributing to the advancement of computational methods in this field.
![image](https://github.com/harkic/UniproLocd/assets/99328605/32ef4e25-2f9d-4076-9188-42811e5ee43a)
### UniproLocd uses the following dependencies:
  python 3.6<br>
  pytorch<br>
  scikit-learn<br>
  numpy<br>
  pandas<br>
### Guiding principles:
1. Folder 'data' contains train and test data used in this study.
Folder 'code' contains scripts for training the model and loading data. Load_data.py is the implementation of load raw protein sequences. model.py is the network architecture. Deepro_Glu.py was used to training the proposed model. utils.py is the implementation of calculate model score.
Folder 'feature_extract' in folder 'code' contains the feature extract method. aaindex.py is the implementation of AAindex. be_feature.m and exchange_matrix.m are the implementation of BE. BLOSUM62.py is the implementation of BLOSUM62 matrix. DDE.py is the implementation of dipeptide deviation from expected mean.
Folder 'result' contains model test result. test.py is the implementation of calculate model test result.
You can also run Deepro-Glu model on your own datasets. Just replace the train_example.csv in Deepro_Glu.py with your own dataset.
