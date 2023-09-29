import torch
import torch.nn as nn

import logging

from .lorentz_metric import dot4
from ..layers import Net2to2, Eq2to0, MessageNet
from ..trainer import init_weights

class PELICANNano(nn.Module):
    """
    Permutation Invariant, Lorentz Invariant/Covariant Aggregator Network
    """
    def __init__(self, n_hidden,
                 activate_agg=False, activate_lin=True, activation='leakyrelu', add_beams=True, config='s', config_out='s', average_nobj=49, factorize=False, masked=True,
                 activate_agg_out=True, activate_lin_out=False,
                 scale=1, dropout = True, drop_rate=0.05, drop_rate_out=0.05, batchnorm=None,
                 device=torch.device('cpu'), dtype=None):
        super().__init__()

        logging.info('Initializing network!')

        self.device, self.dtype = device, dtype
        self.n_hidden = n_hidden
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.scale = scale
        self.add_beams = add_beams
        self.config = config
        self.config_out = config_out
        self.average_nobj = average_nobj
        self.factorize = factorize
        self.masked = masked

        if dropout:
            self.dropout_layer = nn.Dropout(drop_rate)
            self.dropout_layer_out = nn.Dropout(drop_rate_out)

        # This is the main part of the network -- a sequence of permutation-equivariant 2->2 blocks
        # Each 2->2 block consists of a component-wise messaging layer that mixes channels, followed by the equivariant aggegration over particle indices
        self.net2to2 = Net2to2([1, n_hidden], [[]], activate_agg=activate_agg, activate_lin=activate_lin, activation = activation, dropout=dropout, drop_rate=drop_rate, batchnorm = batchnorm, config=config, average_nobj=average_nobj, factorize=factorize, masked=masked, device = device, dtype = dtype)
        
        # The final equivariant block is 2->1 and is defined here manually as a messaging (BatchNorm) layer followed by the 2->0 aggregation layer
        # This messaging layer actually reduces to nothing but a BatchNorm
        self.msg_2to0 = MessageNet([n_hidden], activation=activation, batchnorm=batchnorm, device=device, dtype=dtype)   
        # This aggregation layer applies 2 aggregators and mixes them down to 1 output classification score (positive=predicted signal)    
        self.agg_2to0 = Eq2to0(n_hidden, 1, activate_agg=activate_agg_out, activate_lin=activate_lin_out, activation = activation, config=config_out, factorize=False, average_nobj=average_nobj, device = device, dtype = dtype)
        
        self.apply(init_weights)

        logging.info('_________________________\n')
        for n, p in self.named_parameters(): logging.info(f'{"Parameter: " + n:<80} {p.shape}')
        logging.info('Model initialized. Number of parameters: {}'.format(sum(p.nelement() for p in self.parameters())))
        logging.info('_________________________\n')

    def forward(self, data, covariance_test=False):
        """
        Runs a forward pass of the network.
        """
        # Get and prepare the data
        particle_scalars, particle_mask, edge_mask, event_momenta = self.prepare_input(data)
        dot_products = dot4(event_momenta.unsqueeze(1), event_momenta.unsqueeze(2))
        inputs = dot_products.unsqueeze(-1)
        print(f"Jet mass {inputs[0].sum().sqrt()}")
        print(f"is_signal {data['is_signal'][0]}")
        # regular multiplicity
        nobj = particle_mask.sum(-1, keepdim=True)

        # Apply the sequence of PELICAN equivariant 2->2 blocks with the IRC weighting.
        act1 = self.net2to2(inputs, mask = edge_mask.unsqueeze(-1), nobj = nobj)
        
        # The last equivariant 2->0 block is constructed here by hand: message layer, dropout, and Eq2to0.
        act2 = self.msg_2to0(act1, mask=edge_mask.unsqueeze(-1))
        if self.dropout:
            act2 = self.dropout_layer(act2)
        act3 = self.agg_2to0(act2, nobj = nobj)

        # The output layer applies dropout and an MLP.
        if self.dropout:
            act3 = self.dropout_layer_out(act3)
        prediction = torch.cat([-act3, act3], axis = -1)

        if torch.isnan(prediction).any():
            logging.info(f"inputs: {torch.isnan(inputs).any()}")
            logging.info(f"act1: {torch.isnan(act1).any()}")
            logging.info(f"act2: {torch.isnan(act2).any()}")
            logging.info(f"prediction: {torch.isnan(prediction).any()}")
        assert not torch.isnan(prediction).any(), "There are NaN entries in the output! Evaluation terminated."

        if covariance_test:
            return {'predict': prediction, 'inputs': inputs, 'act1': act1, 'act2': act2, 'act3': act3}
        else:
            return {'predict': prediction}

    def prepare_input(self, data):
        """
        Extracts input from data class

        Parameters
        ----------
        data : ?????
            Information on the state of the system.

        Returns
        -------
        scalars : :obj:`torch.Tensor`
            Tensor of scalars for each particle.
        particle_mask : :obj:`torch.Tensor`
            Mask used for batching data.
        edge_mask: :obj:`torch.Tensor`
            Mask used for batching data.
        particle_ps: :obj:`torch.Tensor`
            4-momenta of the particles
        """
        device, dtype = self.device, self.dtype

        particle_ps = data['Pmu'].to(device, dtype)

        data['Pmu'].requires_grad_(True)
        particle_mask = data['particle_mask'].to(device, torch.bool)
        edge_mask = data['edge_mask'].to(device, torch.bool)

        if 'scalars' in data.keys():
            scalars = data['scalars'].to(device, dtype)
        else:
            scalars = None
        return scalars, particle_mask, edge_mask, particle_ps

def expand_var_list(var):
    if type(var) is list:
        var_list = var
    else:
        raise ValueError('Incorrect type {}'.format(type(var)))
    return var_list
