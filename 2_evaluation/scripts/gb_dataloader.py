import torch
from torch.utils.data import Dataset
from random import randrange

#Dataset loader for Gutenberg structure-split data. Accepts individual train, test and dev jsonl files
#Adapted from https://github.com/timinar/BabyLlama/blob/main/babylm_dataset.py




class GBDataset(Dataset):

    def __init__(self, data_file, seq_length, offset=0, random_chunk=False):
        self.seq_length = seq_length
        self.offset = offset
        self.random_chunk = random_chunk
        if isinstance(data_file, str):
            self.data = torch.load(data_file)
        else:
            self.data = data_file

    def __len__(self):
        if self.random_chunk:
            return len(self.data) // self.seq_length - 1
        else:
            return (len(self.data) - self.offset) // self.seq_length

    def __getitem__(self, i):
        if self.random_chunk:
            offset = randrange(self.seq_length)
            return self.data[i*self.seq_length+offset:(i+1)*self.seq_length+offset]
        else:
            return self.data[i*self.seq_length+self.offset:(i+1)*self.seq_length+self.offset]
        
    
