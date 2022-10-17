from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List
import joblib
from sklearn import preprocessing
import torch
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import re
import csv
import random
import copy
from math import sqrt
from scipy import stats
import torch.nn.functional as F


CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}



CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)
PT_FEATURE_SIZE = 21
CHARPROTSET = { "A": 1, "C": 2,  "D": 3, "E": 4, "F": 5,"G": 6, 
				 "H": 7, "I": 8, "K": 9, "L": 10, "M": 11, 
				 "N": 12,"P": 13, "Q": 14, "R": 15, "S": 16,  
				 "T": 17, "V": 18, "W": 19, 
				"Y": 20, "X": 21
				 }

CHARPROTLEN = 21

def label_sequence(line, MAX_SEQ_LEN):   
    X = np.zeros(MAX_SEQ_LEN, dtype=np.int)
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = CHARPROTSET[ch]
    return X

def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=np.int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] 
    return X



class MyDataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len, max_smi_len):
        data_path = Path(data_path)

        affinity = {}
        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity


        ligands_df = pd.read_csv(data_path / f"{phase}_smi.csv")
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        self.smi = ligands
        self.max_smi_len = max_smi_len
        self.pdbids = ligands_df['pdbid'].values
        self.max_seq_len = max_seq_len

        prot_df = pd.read_csv(data_path / f"{phase}_seq_.csv")
        prots = {i["id"]: i["seq"] for _, i in prot_df.iterrows()}
        self.prots = prots
        if phase == 'test105' or phase == 'test71' :
            pkt = pd.read_csv(data_path / f"zy_{phase}_pkt.csv")
        else:
            pkt = pd.read_csv(data_path / f"zy_{phase}_pkt_2.csv")
            disabled_features = ['F43', 'F44', 'F87','F88', 'F131', 'F132']
            pkt = pkt.drop(disabled_features, axis=1)
        
        a = pkt['pdbid'].apply(lambda x:x[:4]).tolist()
        pkt = pkt.drop(labels='pdbid',axis=1)
        pkt['pdbid']=a  
        pkt.set_index(['pdbid'], inplace = True)
         
        self.pkt = pkt
		
        comp900 = pd.read_csv(f"../data/comp900_{phase}.csv")

        comp900.set_index(['pdbid'], inplace = True)
        self.comp900 = comp900
        self.length = len(self.smi)


    def __getitem__(self, idx):
        pdbid = self.pdbids[idx]    
        aug_smile =   self.smi[pdbid]
        protseq = self.prots[pdbid]

        return (
                np.array(self.pkt.loc[pdbid], dtype=np.float32),
                np.array(self.comp900.loc[pdbid], dtype=np.float32),
                label_smiles(aug_smile, self.max_smi_len),
                label_sequence(protseq,self.max_seq_len),
                np.array(self.affinity[pdbid], dtype=np.float32))

    def __len__(self):
        return self.length
  
