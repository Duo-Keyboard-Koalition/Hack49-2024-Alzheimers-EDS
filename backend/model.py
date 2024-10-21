import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Linear(29, 15)
        self.layer2 = nn.Linear(15, 8)
        self.output_layer = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output_layer(x)
        x = torch.mean(x, dim=1)  # Global Average Pooling over the variable dimension
        return x[0]


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