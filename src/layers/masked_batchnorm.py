import torch
import torch.nn as nn

# def lengths_to_mask(lengths, max_len=None, dtype=None):
#     """
#     Converts a "lengths" tensor to its binary mask representation.
    
#     Based on: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397
    
#     :lengths: N-dimensional tensor
#     :returns: N*max_len dimensional tensor. If max_len==None, max_len=max(lengtsh)
#     """
#     assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
#     max_len = max_len or lengths.max().item()
#     mask = torch.arange(
#         max_len,
#         device=lengths.device,
#         dtype=lengths.dtype)\
#     .expand(len(lengths), max_len) < lengths.unsqueeze(1)
#     if dtype is not None:
#         mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
#     return mask


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """
    Masked verstion of the 1D Batch normalization.
    
    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py
    
    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.
    
    Check pytorch's BatchNorm1d implementation for argument details.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, device=None, dtype=None):
        super(MaskedBatchNorm1d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            device,
            dtype
        )
        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, inp, mask):
        self._check_input_dim(inp)
        
        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.

        # mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)
        
        assert len(mask.shape) == 3, f'Expected 3 dimensions in mask, instead got {len(mask.shape)} and shape {mask.shape}'

        mask_bool = mask
        n = mask.sum()
        mask = mask / n
        mask = mask.unsqueeze(-1).expand(inp.shape)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and n > 1:
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            mean = (mask * inp).sum([0, 1])
            # ...whereas Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = (mask * inp ** 2).sum([0, 1]) - mean ** 2
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, None, :]) / (torch.sqrt(var[None, None, :] + self.eps))
        if self.affine:
            inp = inp * self.weight[None, None, :] + self.bias[None, None, :]

        inp = torch.where(mask, inp, self.zero)

        return inp

class MaskedBatchNorm2d(nn.BatchNorm2d):
    """
    Masked version of the 2D Batch normalization.
    
    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py
    
    inp: input tensor where the last dimension is assumed to be the channel dim (i.e. this is the only dimension that will *not* be averaged over)
    mask: mask tensor whose shape has to be broadcastable with inp, but for correct averaging need mask.shape[:-1]=inp.shape[:-1] and mask.shape[-1]=1 (because the denominator of the average is mask.sum())
    
    Check pytorch's BatchNorm2d implementation for argument details.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, device=None, dtype=None):
        super(MaskedBatchNorm2d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            device,
            dtype
        )
        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, inp, mask):
        self._check_input_dim(inp)
        
        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.

        # mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)

        assert len(mask.shape) == 4, f'Expected 4 dimensions in mask, instead got {len(mask.shape)} and shape {mask.shape}'
        
        mask_bool = mask
        n = mask.sum()
        mask = mask / n

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and n > 1:
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            mean = (mask * inp).sum([0, 1, 2])
            # ...whereas Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = (mask * inp ** 2).sum([0, 1, 2]) - mean ** 2
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, None, None, :]) / (torch.sqrt(var[None, None, None, :] + self.eps))

        if self.affine:
            inp = inp * self.weight[None, None, None, :] + self.bias[None, None, None, :]

        inp = torch.where(mask_bool, inp, self.zero)

        return inp
class MaskedBatchNorm3d(nn.BatchNorm3d):
    """
    Masked verstion of the 3D Batch normalization.
    
    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py
    
    inp: input tensor where the last dimension is assumed to be the channel dim (i.e. this is the only dimension that will *not* be averaged over)
    mask: mask tensor whose shape has to be broadcastable with inp, but for correct averaging need mask.shape[:-1]=inp.shape[:-1] and mask.shape[-1]=1 (because the denominator of the average is mask.sum())
    
    Check pytorch's BatchNorm3d implementation for argument details.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True, device=None, dtype=None):
        super(MaskedBatchNorm3d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats,
            device,
            dtype
        )
        self.zero = torch.tensor(0, device=device, dtype=dtype)

    def forward(self, inp, mask):
        self._check_input_dim(inp)
        
        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.

        # mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)
        if mask is None: mask = (inp != 0.)
        assert len(mask.shape) == 5, f'Expected 5 dimensions in mask, instead got {len(mask.shape)} and shape {mask.shape}'
        
        mask_bool = mask
        n = mask.sum(dim=(0,1,2,3))
        mask = mask / n

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and (n > 1).all():
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            mean = (mask * inp).sum([0, 1, 2, 3])
            # ...whereas Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = (mask * inp ** 2).sum([0, 1, 2, 3]) - mean ** 2
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, None, None, None, :]) / (torch.sqrt(var[None, None, None, None, :] + self.eps))

        if self.affine:
            inp = inp * self.weight[None, None, None, None, :] + self.bias[None, None, None, None, :]

        inp = torch.where(mask_bool, inp, self.zero)

        return inp
