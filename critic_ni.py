
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.base_models import Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Critic_NIVRP(nn.Module):

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(Critic_NIVRP, self).__init__()

        self.encoder = Encoder(1, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)
        
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def interdiction(self, static, static_1, dynamic_1):
        batch_size, input_size, seq_len = static.size()
        
        # Use the probabilities of visiting each
        static_hidden = self.encoder(static_1.view(batch_size, 1, seq_len * seq_len).float())
        dynamic_hidden = self.encoder(dynamic_1.view(batch_size, 1, seq_len * seq_len).float())

        hidden = torch.cat((static_hidden, dynamic_hidden), 2)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output
        
