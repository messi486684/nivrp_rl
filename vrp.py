"""Defines the main task for the VRP."""
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import xlwt
from matplotlib import font_manager

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use('Qt5Agg')




class VehicleRoutingDataset(Dataset):
    def __init__(self, num_samples, input_size, args, max_load = 20, max_demand=9, 
                  max_budg = 2000, seed=None):
        super(VehicleRoutingDataset, self).__init__()

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.input_size = input_size
        self.max_load = max_load
        self.max_demand = max_demand
        self.max_budg = max_budg
        self.args = args
        

        # Depot location will be the first node in each
        locations = torch.rand((num_samples, 2, input_size + 1))      
        self.static = torch.tensor(locations)
        
        # All states will broadcast the drivers current load
        # Note that we only use a load between [0, 1] to prevent large
        # numbers entering the neural network
        dynamic_shape = (num_samples, 1, input_size + 1)
        loads = torch.full(dynamic_shape, 1., dtype = torch.float64)

        # All states will have their own intrinsic demand in [1, max_demand), 
        # then scaled by the maximum load. E.g. if load=10 and max_demand=30, 
        # demands will be scaled to the range (0, 3)
        demands = torch.randint(1, max_demand + 1, dynamic_shape, dtype = torch.int64)
        demands = demands / float(max_load)
        
        demands[:, 0, 0] = 0  # depot starts with a demand of 0
       
        
        self.dynamic_0 = torch.tensor(np.concatenate((loads, demands), axis=1))
        self.device = self.dynamic_0.device
        
        dynamic1_shape = (num_samples, input_size + 1, input_size + 1)
        
        distance = torch.zeros(num_samples, input_size + 1, input_size + 1)         
        distance = torch.cdist(locations.transpose(1,2), locations.transpose(1,2), p=2)
        
        penalty = 3 * torch.rand(dynamic1_shape)
        penalty = penalty * (1 - torch.eye(input_size + 1))
 
        max_ind = 65
        min_ind = 40
        idct_cost = torch.randint(min_ind, max_ind, dynamic1_shape, dtype = torch.int64)
        idct_cost = idct_cost * (1 - torch.eye(input_size + 1))
        a = torch.zeros(dynamic1_shape)
        a[:,0,:] = 3 * max_budg
        a[:,:, 0] = 3 * max_budg
        idct_cost = a + idct_cost
        Budgets = torch.full(dynamic1_shape, max_budg, dtype = torch.int64)
        
        self.dynamic_1 = distance
        self.static_1 = penalty
        self.static_2 = idct_cost
        self.dynamic_2 = Budgets

        wb = xlwt.Workbook()
        sh1 = wb.add_sheet('time')
        sh2 = wb.add_sheet('ind_cost')
        sh3 = wb.add_sheet('penalty')
        sh4 = wb.add_sheet('demand')
        
        for i in range(input_size + 1):
            for j in range(input_size + 1):
                 sh1.write(i+2, j+2, self.dynamic_1[0,i,j].tolist())
                 sh2.write(i+2, j+2, self.static_2[0,i,j].tolist())
                 sh3.write(i+2, j+2, self.static_1[0,i,j].tolist())
        
        for i in range(input_size + 1):
            sh4.write(i, 1, self.dynamic_0[0, 1, i].tolist())
        
 #       wb.save('Data_NI7.xls')
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (self.static[idx], self.dynamic_0[idx], self.dynamic_1[idx], self.static_1[idx], self.static_2[idx], self.dynamic_2[idx], self.static[idx, :, 0:1])
    
    def update_mask(self, mask, dynamic, chosen_idx=None):
        """Updates the mask used to hide non-valid states.
        """

        # Convert floating point to integers for calculations
        loads = dynamic.data[:, 0]  # (batch_size, seq_len)
        demands = dynamic.data[:, 1]  # (batch_size, seq_len)

        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        if demands.eq(0).all():
            return demands * 0.

        # Otherwise, we can choose to go anywhere where demand is > 0
        new_mask = demands.ne(0) * demands.lt(loads)

        # We should avoid traveling to the depot back-to-back
        repeat_home = chosen_idx.ne(0)

        if repeat_home.any():
            new_mask[repeat_home.nonzero(), 0] = 1.
        if ~repeat_home.any():
            new_mask[~repeat_home.nonzero(), 0] = 0.

        # ... unless we're waiting for all other samples in a minibatch to finish
        has_no_load = loads[:, 0].eq(0).float()
        has_no_demand = demands[:, 1:].sum(1).eq(0).float()

        combined = (has_no_load + has_no_demand).gt(0)
        if combined.any():
            new_mask[combined.nonzero(), 0] = 1.
            new_mask[combined.nonzero(), 1:] = 0.

        return new_mask.float()

    def update_dynamic(self, dynamic, dynamic_1, static_1, interdicted_arcs, chosen_idx):
        """Updates the (load, demand) dataset values."""

        # Update the dynamic elements differently for if we visit depot vs. a city
        visit = chosen_idx.ne(0)
        depot = chosen_idx.eq(0)

        # Clone the dynamic variable so we don't mess up graph
        # clone can copy inputs
        all_loads = dynamic[:, 0].clone()
        all_demands = dynamic[:, 1].clone()
        all_distances = dynamic_1.clone()

        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))

        # Across the minibatch - if we've chosen to visit a city, try to satisfy
        # as much demand as possible
        if visit.any():

            new_load = torch.clamp(load - demand, min=0)
            new_demand = torch.clamp(demand - load, min=0)

            # Broadcast the load to all nodes, but update demand seperately
            visit_idx = visit.nonzero().squeeze()

            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = new_demand[visit_idx].view(-1)
            all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1)

        # Return to depot to fill vehicle load
        if depot.any():
            all_loads[depot.nonzero().squeeze()] = 1.
            all_demands[depot.nonzero().squeeze(), 0] = 0.
            
        if interdicted_arcs is not None:
           all_distances = all_distances + static_1 * interdicted_arcs
            

        tensor = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
        return torch.tensor(tensor.data, device=dynamic.device), torch.tensor(all_distances)

#########################################################################################################
    def update_mask_idct(self, interdicted_arcs, static_2, remained_ind_budg, chosen_arc):

#        all_distances = distances.clone()           # (batch_size, seq_len, seq_len)
        all_ind_cost = static_2.clone()             # (batch_size, seq_len, seq_len)
   #     all_budgets = remained_ind_budg.clone()     # (batch_size, seq_len, seq_len)

#        batch_size = all_distances.size(0)
        seq_len = all_ind_cost.size(1)
        
#        current_arcs = torch.zeros([batch_size, seq_len, seq_len])
#        for i in range(batch_size):
#            for j in range(tour.shape[1] - 1):
#                if tour[i][j] != tour[i][j + 1]:
#                   current_arcs[i, tour[i][j], tour[i][j + 1]] = 1
       
#        new_mask = current_arcs.ne(0) * all_ind_cost.lt(all_budgets) * interdicted_arcs.eq(0)
        new_mask = all_ind_cost.lt(remained_ind_budg) * interdicted_arcs.eq(0)

                   
        new_mask = new_mask * (1 - torch.eye(seq_len)) 
        new_mask[:,0,:] = 0.
        new_mask[:,:,0] = 0.
        
        all_masked = new_mask.float().sum(dim = 1).sum(dim = 1).le(0).float()
        if all_masked.any():
            new_mask[all_masked.nonzero(), 0 , :] = -0.01
#            new_mask[all_masked.nonzero(), : , 0] = -1.
            new_mask[all_masked.nonzero(), 1:, 1:] = 0. 

            
        return remained_ind_budg, new_mask.float()


    def update_dynamic_idct(self, distance, static_1, static_2, remained_budget, chosen_arc, interdicted_arcs, tour_vrp):
            """Updates the (distance and interdiction budget dataset values."""

#            all_distances = dynamic_1.clone()
#            all_penalties = static_1.clone()
            all_ind_cost = static_2.clone()
#            all_budgets = dynamic_2.clone()

#            distances = torch.tensor([(all_penalties[i, val[0], val[1]] + all_distances[i, val[0], val[1]]).tolist() for i, val in enumerate(chosen_arc)])
            budgets = torch.tensor([(torch.clamp(remained_budget[i, val[0], val[1]] - all_ind_cost[i, val[0], val[1]], min = 0)).tolist() for i, val in enumerate(chosen_arc)])

            for i, val in enumerate(chosen_arc): 
#                all_distances[i, val[0].tolist(), val[1].tolist()] = distances[i]
                remained_budget[i, :, :] = budgets[i]
                
            batch_size, seq_len, seq_len = interdicted_arcs.size()
#            current_arcs = torch.zeros([batch_size, seq_len, seq_len])
            
           
            distance = distance + torch.tensor(static_1 * interdicted_arcs)
    

            return torch.tensor(distance.data, device = self.device), torch.tensor(remained_budget.data, device = self.device)       

#########################################################################################################

def reward(distance, static, tour_indices):
    """Distance between all cities / nodes given by tour_indices"""

    # Convert the indices back into a tour
##    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    start = torch.zeros(static.size(0)).unsqueeze(1)
    y = torch.cat((start, tour_indices, start), dim=1)

    # Distance between each consecutive point
    tour_len = torch.tensor([distance[i, y[i, :-1].tolist() , y[i, 1:].tolist()].sum() for i in range(len(distance))])
    
    return tour_len


def reward_idct(static_1, interdicted_arcs, tour):
    batch_size, seq_len, seq_len = interdicted_arcs.size()
    current_arcs = torch.zeros([batch_size, seq_len, seq_len])
    
    start = torch.zeros(batch_size).unsqueeze(1)
    tour = torch.cat((start, tour, start), dim=1)
    
    for i in range(batch_size):
        for j in range(tour.shape[1] - 1):
            if tour[i][j] != tour[i][j + 1]:
               current_arcs[i, tour[i][j].int().tolist(), tour[i][j + 1].int().tolist()] = 1
   
    total_penalty = torch.tensor(static_1 * interdicted_arcs * current_arcs).sum(dim = 2).sum(dim = 1)

    return total_penalty
#######################################################################################################
def render(static, tour_indices, save_path):
    """Plots the found solution."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        # Assign each subtour a different colour & label in order traveled
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]

        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder = 1, label=j)

        ax.legend(loc = "upper right", fontsize=3, framealpha = 0.2)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        plt.rcParams['xtick.labelsize'] = 5
        plt.rcParams['ytick.labelsize'] = 5
        ax.scatter(x, y, s=20, c='#17becf', edgecolors='k', zorder = 1)
        ax.scatter(x[0], y[0], s=20, c='k', marker = '*', edgecolors = 'r', zorder = 1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)

########################################################################
class NewDataset(Dataset):
    def __init__(self, num_samples, static, dynamic_0, dynamic_1, static_1, static_2, dynamic_2):
        super(NewDataset, self).__init__()

        self.num_samples = num_samples
        self.static = static
        self.dynamic_0 = dynamic_0
        self.dynamic_1 = dynamic_1
        self.static_1 = static_1
        self.static_2 = static_2
        self.dynamic_2 = dynamic_2      
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (self.static[idx], self.dynamic_0[idx], self.dynamic_1[idx], self.static_1[idx], self.static_2[idx], self.dynamic_2[idx], self.static[idx, :, 0:1])
    