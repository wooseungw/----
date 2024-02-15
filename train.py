import os
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')


class BaseModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        
        
        
        
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    
if __name__ == '__main__':
    print("hi")