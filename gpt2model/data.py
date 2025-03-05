import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset, DataLoader
class GPTDataset(Dataset):
    def __init__(self, text,tokenizer ,context_len, stride):
        super().__init__()
        # tokenizer = tiktoken.get_encoding('gpt2')
        ids = tokenizer.encode(text)
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(ids)-context_len, stride):
            input_id = ids[i:i+context_len]
            target_id = ids[i+1:i+context_len+1]
            self.input_ids.append(torch.tensor(input_id))
            self.target_ids.append(torch.tensor(target_id))
        # self.input_ids, self.target_ids
    
    def __getitem__(self,idx):
        return self.input_ids[idx], self.target_ids[idx]

    def __len__(self, ):
        return len(self.input_ids)
        

def create_dataloader_v1(txt, batch_size=4, max_length=256,
    stride=128, shuffle=True, drop_last=True,num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
  
   
			 
