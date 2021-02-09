import torch
import torch.nn as nn
from torch import optim

device = torch.device("cuda")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(torch.load('/content/nfs/machine-learning/coop/data_fasttext/input_de_text.pt'))
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input_word_vector, hidden):
        embedded = input_word_vector.view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)