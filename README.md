# UniproLcad
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
2. Folder 'code' contains scripts for training the model and loading data. model.py is the network architecture. UniproLocd.py was used to training the proposed model. utils.py is the implementation of calculate model score.
3. Folder 'feature_extract' in folder 'code' contains the feature extraction of pre-trained language model. ProtBert.py is the implementation of ProtBert pre-trained language model. ESM.py is the implementation of ESM-2 pre-trained language model. Unirep.py is the implementation of Unirep pre-trained language model
4. You can also run UniproLocd model on your own datasets. Just change the file path in the code.

### Note
This code is for the article 'UniproLcad: Antimicrobial Peptide Prediction Using a Multi-Protein Language Model within a Deep Learning Framework Stack'.
