import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import GPTDataset
from GPT import GPTModel
import tiktoken

def get_tokenizer():
    return tiktoken.get_encoding('gpt2')

def load_data(data_path, cfg):
    with open(data_path + '.txt', 'r') as f:
        text_data = f.read()
    
    dataset = GPTDataset(text_data, cfg['context_length'], 200)
    dataloader  = DataLoader(dataset, batch_size=1, shuffle=False, )
    return dataloader

def load_model(cfg):
    gpt = GPTModel(cfg)
    return gpt


def get_model_configuration():
    return {
    'vocab_size':50257,
    'context_length': 1024,
    'emb_dim': 1600,#1280, #1024, #768,
    'n_layers': 48, #36, #24, # 12
    'n_heads': 25,#20, #16, # 12
    'drop_rate': 0.1,
    'qkv_bias': False
    }

def calculate_model_parameters(model):
    return sum( p.numel() for p in model.parameters())  - sum(p.numel() for p in model.final_linear_layer.parameters())

def model_size(model):
    total_params = sum( p.numel() for p in model.parameters())

    return (total_params*4) / (1024*1024)


def generate_sample_text(cfg,model, max_sequence_length=10):
    text = 'I am shah'
    tokenizer = get_tokenizer()
    in_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    for _ in range(max_sequence_length):
        ids_limit = in_ids[:, -cfg['context_length']:]
        model.eval()
        with torch.no_grad():
            out = model(ids_limit)
        logits = out[:,-1,:]
        # print(f'Logits: {logits.shape}')
        probs = torch.softmax(logits,dim=-1)
        next_idx = torch.argmax(probs, keepdim=True)
        # print(next_idx[0][0])
        in_ids = torch.cat((in_ids, next_idx), dim=1)
    return in_ids



def main():
    cfg = get_model_configuration()
    dataloader = load_data('the-verdict', cfg)
    model = load_model(cfg)
    model_parameters = calculate_model_parameters(model)
    print(f'The total number of parameters: {model_parameters}')

    size = model_size(model)
    print(f'The Model size: {size} MB')


main()





