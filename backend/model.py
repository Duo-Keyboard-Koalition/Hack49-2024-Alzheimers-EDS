import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + attn_output  # Add & Normalize
        x = self.layer_norm(x)
        return x

class TimeSeriesClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, output_dim):
        super(TimeSeriesClassifier, self).__init__()
        self.self_attention = SelfAttention(input_dim, num_heads)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        x = x.permute(1, 0, 2)  # Change to [seq_len, batch_size, input_dim]
        x = self.self_attention(x)
        x = x.permute(1, 0, 2)  # Change back to [batch_size, seq_len, input_dim]
        x = torch.mean(x, dim=1)  # Global Average Pooling over the time dimension
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x

# Define the combined Encoder-Decoder model
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        emission, _ = self.encoder(x)
        x = self.decoder(emission)
        return x