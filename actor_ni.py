import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.base_models import Encoder, Pointer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DeepRL_NIVRP(nn.Module):
    """Defines the main Encoder, Decoder, and Pointer combinatorial models."""
    
    def __init__(self, static_size, dynamic_size, SEQ_size, hidden_size, reward_fn, args, kwargs,
                  update_fn_idct = None, mask_fn_idct = None, num_layers=1, dropout=0.):
        super(DeepRL_NIVRP, self).__init__()
        
        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')
       
        self.args = args
        self.kwargs = kwargs
        self.update_fn_idct = update_fn_idct
        self.mask_fn_idct = mask_fn_idct
        self.hidden_size = hidden_size
        
        self.dynamic_1_encoder_idct = Encoder(1, 128)
        self.static_1_encoder_idct = Encoder(1, 128)       
        self.decoder_idct = Encoder(1, 128)     
        self.pointer = Pointer(128, num_layers, dropout)
        
        
        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)
        
        self.x00 = torch.zeros((1, 1, 1), requires_grad = True, device = device)
        self.x0 = torch.zeros((1, static_size, 1), requires_grad = True, device = device)
  
    ##################################################################################################
    def interdiction(self, model, model_path, tour, static, dynamic_0, 
                     dynamic_1,  static_1, static_2, dynamic_2, reward_fn, args,
                     decoder_input = None, last_hh = None):
        
 
      
         
        batch_size, input_size, sequence_size = static.size()
    
        if decoder_input is None:
            decoder_input = self.x00.expand(batch_size, -1, -1)
    
        # Always use a mask - if no function is provided, we don't update it
        mask_idct = torch.ones(batch_size, sequence_size, sequence_size, device = device)
        mask_idct = mask_idct * (1 - torch.eye(sequence_size))
        mask_idct[:,0,:] = 0
        mask_idct[:,:,0] = 0
#        min_distance, IND = torch.min(dynamic_1 + 10 * torch.eye(sequence_size), 2)
        
        # Structures for holding the output sequences
        ind_logp = []
        interdicted_arcs = torch.zeros(batch_size, sequence_size, sequence_size)
        distance = dynamic_1.clone()
        remained_ind_budg = dynamic_2.clone()
        max_steps = sequence_size * sequence_size if self.mask_fn_idct is None else args.num_interdiction
        
        
        #current_arcs = torch.zeros([batch_size, sequence_size, sequence_size])
        #for i in range(batch_size):
        #    for j in range(tour.shape[1] - 1):
        #        if tour[i][j] != tour[i][j + 1]:
        #           current_arcs[i, tour[i][j], tour[i][j + 1]] = 1
                 
                
        # tour: [batch_size, tour_length]
            # خروجی: [batch_size, sequence_size, sequence_size]
        batch_size, tour_length = tour.shape
        
            # استخراج گره‌های مبدا و مقصد برای هر گام از مسیر
        from_nodes = tour[:, :-1]  # [B, T-1]
        to_nodes = tour[:, 1:]     # [B, T-1]
        
            # حذف یال‌های تکراری (مثلاً گره به خودش)
        mask = from_nodes != to_nodes
        
            # ساخت اندیس‌ها
        batch_idx = torch.arange(batch_size).unsqueeze(1).expand_as(from_nodes)  # [B, T-1]
        
            # مقداردهی صفر
        current_arcs = torch.zeros(batch_size, sequence_size, sequence_size, dtype=torch.bool, device=tour.device)

            # مقداردهی فقط به یال‌های معتبر
        current_arcs[batch_idx[mask], from_nodes[mask], to_nodes[mask]] = 1


 
        dynamic_0_hidden = self.dynamic_1_encoder_idct(current_arcs.view(batch_size, 1, sequence_size * sequence_size).float())
        dynamic_2_hidden = self.dynamic_1_encoder_idct(dynamic_1.view(batch_size, 1, sequence_size * sequence_size).float())
        dynamic_1_hidden = dynamic_0_hidden + dynamic_2_hidden
        static_1_hidden = self.static_1_encoder_idct(static_2.view(batch_size, 1, sequence_size * sequence_size).float())

        
        count = 0
        for _ in range(max_steps): 
            count += 1

            if not mask_idct.byte().any():
                break
    
            # ... but compute a hidden rep for each element added to sequence
            
 
            decoder_hidden = self.decoder_idct(decoder_input)
            
            probs_idct, last_hh = self.pointer.forward(static_1_hidden,
                                                       dynamic_1_hidden,
                                                       decoder_hidden,
                                                       last_hh)

            probs_idct = F.softmax(probs_idct + mask_idct.view(batch_size, sequence_size * sequence_size).log(), dim = 1)   
   
            if self.training:
                if self.args.decode_type == 'stochastic':
                    m = torch.distributions.Categorical(probs_idct)   
                    selected_arc = m.sample()  ## selected arc for interdiction.
                    while not torch.gather(mask_idct.view(batch_size, -1), 1, selected_arc.data.unsqueeze(1)).byte().all():
                        selected_arc = m.sample()
                    logp = m.log_prob(selected_arc) ## or: logp = torch.log(probs[selected_arc])
                
                elif self.args.decode_type == 'greedy':
                    prob_idct, selected_arc = torch.max(probs_idct, 1)  # Greedy
                    logp = prob_idct.log() 
                    
            else:
                prob_idct, selected_arc = torch.max(probs_idct, 1)  # Greedy
                logp = prob_idct.log()               

            chosen_arc = torch.concat((selected_arc.data.unsqueeze(1) // sequence_size, selected_arc.data.unsqueeze(1) % sequence_size), dim = 1)
    
            for i in range(batch_size):
                interdicted_arcs[i, chosen_arc.data[i, 0], chosen_arc.data[i, 1]] = 1
                
            # After visiting a node update the dynamic representation
            if self.update_fn_idct is not None:
                distance, remained_ind_budg = self.update_fn_idct(distance, static_1, static_2, remained_ind_budg, chosen_arc, interdicted_arcs, tour)
#                dynamic_1_hidden = self.dynamic_1_encoder_idct(distance.view(batch_size, 1, sequence_size * sequence_size).float())
          
            
            
            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn_idct is not None:
                start = torch.zeros(tour.size(0), dtype = int).unsqueeze(1)
                tour = torch.cat((start, tour, start), dim=1)                  
                remained_ind_budg, mask_idct = self.mask_fn_idct(interdicted_arcs, 
                                                                 static_1, 
                                                                 remained_ind_budg, 
                                                                 chosen_arc)
            current_arcs = torch.zeros([batch_size, sequence_size, sequence_size])
            for i in range(batch_size):
                for j in range(tour.shape[1] - 1):
                    if tour[i][j] != tour[i][j + 1]:
                       current_arcs[i, tour[i][j], tour[i][j + 1]] = 1
     
            dynamic_0_hidden = self.dynamic_1_encoder_idct(current_arcs.view(batch_size, 1, sequence_size * sequence_size).float())
            dynamic_2_hidden = self.dynamic_1_encoder_idct(dynamic_1.view(batch_size, 1, sequence_size * sequence_size).float())
            dynamic_1_hidden = dynamic_0_hidden + dynamic_2_hidden
                          
                                
            ind_logp.append(logp.unsqueeze(1))
            
            ch = chosen_arc[:, 0] * sequence_size + chosen_arc[:, 1]
            decoder_input = torch.gather(static_1.view(batch_size, 1, sequence_size * sequence_size).float(), 2, ch.view(batch_size, 1, 1)).detach()

            ############################################################################################               
#        print('reward after interdiction equals to: ', reward)
        ind_logp = torch.cat(ind_logp, dim = 1)   #(batch_size, seq_len)
#        tour = torch.cat(tour, dim=1)  # (batch_size, seq_len)
        return tour, ind_logp, distance, interdicted_arcs   

