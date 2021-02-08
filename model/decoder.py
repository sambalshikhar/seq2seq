import torch
import torch.nn as nn
from torch import optim

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, 
                 output_size_1,
                 output_size_2,
                 output_size_3,
                 output_size_4,
                 output_size_5):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding.from_pretrained(torch.load('/content/nfs/machine-learning/coop/data_fasttext/target_combined_text.pt'))
        print(self.embedding.weight.size())

        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out_1 = nn.Linear(hidden_size, output_size_1)
        self.out_2 = nn.Linear(hidden_size, output_size_2)
        self.out_3 = nn.Linear(hidden_size, output_size_3)
        self.out_4 = nn.Linear(hidden_size, output_size_4)
        self.out_5 = nn.Linear(hidden_size, output_size_5)

        self.outs = nn.ModuleList()
        self.outs.append(self.out_1)
        self.outs.append(self.out_2)
        self.outs.append(self.out_3)
        self.outs.append(self.out_4)
        self.outs.append(self.out_5)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden,i):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.outs[i](output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)