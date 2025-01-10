import torch
import torch.nn as nn

class MaskingEncoder:
    """Encoder for masking reasons"""

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
        try:
            keys = list(reasons_dict.keys())
            if not keys:
                raise ValueError("The reasons_dict has no keys.")
            
            # Check if all keys have values
            for key in keys:
                if reasons_dict[key] is None or reasons_dict[key].numel() == 0:
                    raise ValueError(f"Key '{key}' has no values.")
            
            # Stack the values and convert to float
            onehot = torch.stack(list(reasons_dict.values()), dim=-1).float()  # float로 변환
            
            return onehot
            
        except Exception as e:
            raise ValueError(f"Error in onehot_encode: {str(e)}")
