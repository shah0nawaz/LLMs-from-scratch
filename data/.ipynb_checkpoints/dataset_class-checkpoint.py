import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset

class GPTDataset(Dataset):
    def __init__(self, text, context_len, stride):
        super().__init__()
        tokenizer = tiktoken.get_encoding('gpt2')
        ids = tokenizer.encode(text)
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(in_ids)-context_len, stride):
            input_id = ids[i:i+context_len]
            target_id = ids[i+1:i+context_len+1]
            self.input_ids.append(input_id)
            self.target_ids.append(target_id)
        print(self.input_ids, self.target_ids)
    
    def __getitem__(self,):
        return self.input_ids, self.target_ids

    def __len__(self, ):
        return len(self.input_ids)
        
        
if __name__=='__main__':
    text = 'this is a computer'
    gptdataset = GPTDataset(text=text, context_len=4, stride=1)
