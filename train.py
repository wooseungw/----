import os
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from pytorch_lightning import LightningModule, Trainer


class TextClassificationModel(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        output = self(input_ids, attention_mask, labels)
        loss = output.loss
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=2e-5)

    def train_dataloader(self):
        # 여기서는 DataLoader를 반환해야 합니다. 예를 들어:
        # dataset = SomeDataset(...)
        # return DataLoader(dataset, batch_size=32, shuffle=True)
        pass
