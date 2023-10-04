import torch
import torch.nn as nn
import numpy as np


class ImplicitNormalizationConv(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:        
        inp_norm = False
        if inp_norm :
            divisor = torch.norm(x.reshape(x.shape[0], -1), dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)  
            x = x*np.sqrt(x.numel()/x.shape[0])/(divisor+1e-6)
        
        weight_norms = (self.weight**2).sum(dim=(1, 2, 3), keepdim=True).transpose(0, 1).sqrt()

        conv = super().forward(x)
        return conv/(weight_norms+1e-6)
    

class TexpNormalization(nn.Module):
    r"""Applies tilted exponential normalization over an input signal composed of several input
    planes.

    Args:
        tilt: Tilt of the exponential function, must be > 0.


    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)`


    """

    def __init__(
        self,  
        tilt: float = 1.0,
        texp_across_filts_only: bool = True
        ) -> None:
        super(TexpNormalization, self).__init__()

        self.tilt = tilt
        self.texp_across_filts_only = texp_across_filts_only


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Returns softmax of input tensor, for each image in batch"""
        if self.texp_across_filts_only:
            return torch.exp(self.tilt*input)/torch.sum(torch.exp(self.tilt*input),dim=(1),keepdim=True)
        else:
            return torch.exp(self.tilt*input)/torch.sum(torch.exp(self.tilt*input),dim=(1,2,3),keepdim=True)

    '''def __repr__(self) -> str:
        s = "TexpNormalization("
        s += f'tilt={self.tilt}_filts_only_{self.texp_across_filts_only}'
        s += ")"
        return s'''



class AdaptiveThreshold(nn.Module):
    r"""
    Thresholds values x[x>threshold]
    """

    def __init__(self, std_scalar: float = 0.0, mean_plus_std: bool=True) -> None:
        super(AdaptiveThreshold, self).__init__()

        self.std_scalar = std_scalar # misnomer, it is a scale for std.
        self.means_plus_std = mean_plus_std

    def _thresholding(self, x, threshold):
        return x*(x > threshold)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        means = input.mean(dim=(2, 3), keepdim=True)
        stds = input.std(dim=(2, 3), keepdim=True)
        if self.means_plus_std:
            return self._thresholding(input, means + self.std_scalar*stds)
        else:
            return self._thresholding(input, means)

    '''    def __repr__(self) -> str:
        if self.means_plus_std:
            s = f"AdaptiveThreshold(mean_plus_std)"
        else:
            s = f"AdaptiveThreshold(std_scalar={self.std_scalar})"
        return s'''
