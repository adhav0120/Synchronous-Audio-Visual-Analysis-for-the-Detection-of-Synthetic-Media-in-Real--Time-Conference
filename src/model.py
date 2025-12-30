# src/model.py

import torch
import torch.nn as nn
import math
import config

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class VoiceClfTransformer(nn.Module):
    def __init__(self, input_dim=config.N_MELS, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super(VoiceClfTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.d_model = d_model
        self.classifier = nn.Linear(d_model, 2) # 2 classes: bonafide, spoof

    def forward(self, src):
        # src shape: [batch_size, n_mels, time_steps] -> permute to [batch_size, time_steps, n_mels]
        src = src.permute(0, 2, 1)
        
        # Project input dimension to d_model
        src = self.input_proj(src)
        
        # Add positional encoding
        # src shape for pos_encoder: [time_steps, batch_size, d_model] -> permute
        src = src.permute(1, 0, 2)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2) # permute back to [batch_size, time_steps, d_model]

        # Transformer encoder
        output = self.transformer_encoder(src) # [batch_size, time_steps, d_model]
        
        # Use the output of the [CLS] token or average pooling over time
        output = output.mean(dim=1) # Average pooling -> [batch_size, d_model]
        
        # Classifier
        logits = self.classifier(output)
        return logits
