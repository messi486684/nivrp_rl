import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.base_models import Encoder, Pointer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeepRL(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models."""

    def __init__(self, static_size, dynamic_size, SEQ_size, hidden_size,
                 update_fn = None, mask_fn = None, num_layers=1, dropout=0.):
        super(DeepRL, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn
        self.mask_fn = mask_fn            
        # Define the encoder & decoder models
        self.dynamic_1_encoder = Encoder(SEQ_size, hidden_size)
        self.dynamic_0_encoder = Encoder(dynamic_size, hidden_size)
        
        self.decoder = Encoder(static_size, hidden_size)
        self.pointer = Pointer(hidden_size, num_layers, dropout)


        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified
        self.x0 = torch.zeros((1, static_size, 1), requires_grad = True, device = device)
  
    def forward(self, static, dynamic_0, dynamic_1, static_1, decoder_input = None, 
                last_hh = None, interdicted_arcs = None):

        batch_size, input_size, sequence_size = static.size()

        if decoder_input is None:
            decoder_input = self.x0.expand(batch_size, -1, -1)

        # Always use a mask - if no function is provided, we don't update it
        mask = torch.ones(batch_size, sequence_size, device = device)

        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []
        max_steps = sequence_size if self.mask_fn is None else 1000

        # all 'pointing' iterations. When / if the dynamic elements change,
        # their representations will need to get calculated again.
        dynamic_1_hidden = self.dynamic_1_encoder(dynamic_1)
        dynamic_0_hidden = self.dynamic_0_encoder(dynamic_0.float())
        
        
        for _ in range(max_steps):

            if not mask.byte().any():
                break

            # ... but compute a hidden rep for each element added to sequence
            decoder_hidden = self.decoder(decoder_input)

            probs, last_hh = self.pointer(dynamic_1_hidden,
                                          dynamic_0_hidden,
                                          decoder_hidden, last_hh)
            probs = F.softmax(probs + mask.log(), dim=1)

            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:
                m = torch.distributions.Categorical(probs)

                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                selected_node = m.sample()  ## ptr is selected action
                while not torch.gather(mask, 1, selected_node.data.unsqueeze(1)).byte().all():
                    selected_node = m.sample()
                logp = m.log_prob(selected_node) ## or: logp = torch.log(probs[ptr])
            else:
                prob, selected_node = torch.max(probs, 1)  # Greedy
                logp = prob.log()
                
               
            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                dynamic_0, distance = self.update_fn(dynamic_0, dynamic_1, static_1, interdicted_arcs, selected_node.data)
                dynamic_0_hidden = self.dynamic_0_encoder(dynamic_0.float())

                # Since we compute the VRP in minibatches, some tours may have
                # number of stops. We force the vehicles to remain at the depot 
                # in these cases, and logp := 0
                is_done = dynamic_0[:, 1].sum(1).eq(0).float()
                logp = logp * (1. - is_done)

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, dynamic_0, selected_node.data).detach()
                
                
            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(selected_node.data.unsqueeze(1))

            decoder_input = torch.gather(static, 2,
                                  selected_node.view(-1, 1, 1).expand(-1, input_size, 1)).detach()

        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp, distance
