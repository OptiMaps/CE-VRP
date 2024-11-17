##TODO: action masking encoder
'''
1. label encoder
2. one-hot encoder
3. linear encoder
'''
import torch
import torch.nn as nn

class MaskingEncoder:
    """Encoder for masking reasons"""
    
    def __init__(self, hidden_size: int = 8):
        self.linear = nn.Linear(num_reasons, hidden_size)

    @staticmethod
    def label_encode(reasons_dict: dict) -> torch.Tensor:
        """Encodes reasons into integer labels"""
        batch_size, num_loc = next(iter(reasons_dict.values())).shape
        labels = torch.zeros((batch_size, num_loc), dtype=torch.long)
        
        # label encoding
        for idx, reason in enumerate(reasons_dict.keys(), start=1):
            labels[reasons_dict[reason]] = idx
            
        return labels

    @staticmethod
    def onehot_encode(reasons_dict: dict) -> torch.Tensor:
        """Encodes reasons into one-hot vectors"""
        batch_size, num_loc = next(iter(reasons_dict.values())).shape
        num_reasons = len(reasons_dict)
        
        # Stack the values of reasons_dict along the last dimension
        onehot = torch.stack(list(reasons_dict.values()), dim=-1)
            
        return onehot

    @staticmethod
    def linear_encode(reasons_dict: dict, linear_layer: nn.Linear) -> torch.Tensor:
        """Encodes reasons using a linear layer"""
        onehot = MaskingEncoder.onehot_encode(reasons_dict)
        batch_size, num_loc, num_reasons = onehot.shape

        # linear projection
        encoded = linear_layer(onehot)
        
        return encoded
