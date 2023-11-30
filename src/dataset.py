from torch.utils.data import Dataset
import pandas as pd 
from transformers import GPT2Tokenizer
import torch
import sentencepiece as spm
import os

class FakeReviewsDataset(Dataset):
    def __init__(self, file, model:str):
        super().__init__()
        self.file = file 
        self.data = pd.read_csv(self.file)
        if model == 'Langboat/mengzi-gpt-neo-base':
            self.tokenizer = spm.SentencePieceProcessor(model_file=os.path.join('./model/', "../model/mengzi_gpt.model"))
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        current = self.data.iloc[idx]
        text = current.text
        label = current.label
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            encoded_input = torch.tensor(self.tokenizer.encode(text))
        else:
            encoded_input = self.tokenizer(text, return_tensors='pt', max_length=2048, truncation=True)
        return {'input':encoded_input, 'label':label}
    

if __name__ == '__main__':
    train_data = "./data/train.csv"
    data = FakeReviewsDataset(train_data)
    a = data[0]
    print(a)