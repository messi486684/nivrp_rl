import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.base_models import Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Critic_VRP(nn.Module):

    def __init__(self, SEQ_size, dynamic_size, hidden_size):
        super(Critic_VRP, self).__init__()

        self.dynamic_1_encoder = Encoder(SEQ_size, hidden_size)
        self.dynamic_0_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)
        
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, dynamic_1, dynamic_0):

        # Use the probabilities of visiting each
        dynamic_1_hidden = self.dynamic_1_encoder(dynamic_1)
        dynamic_0_hidden = self.dynamic_0_encoder(dynamic_0.float())

        hidden = torch.cat((dynamic_1_hidden, dynamic_0_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output
        
