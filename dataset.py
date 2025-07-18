import torch
from torch.utils.data import Dataset
import pandas as pd

class PatientDataset(Dataset):
    def __init__(self, id_list, wsi_features, omics_dict, label_dict, input_dims):
        """
        id_list: list of patient IDs
        wsi_features: dict of {patient_id: Tensor [variable_length, 512]}
        omics_dict: dict of {omics_type: {patient_id: Tensor}}
            omics_type ∈ {'rna', 'methyl', 'protein', 'mirna'}
        label_dict: dict of {patient_id: {'til': Tensor, 'response': Tensor}}
        input_dims: dict of omics dimensions for zero vector fallback
        """
        self.ids = id_list
        self.wsi_features = wsi_features
        self.omics_dict = omics_dict
        self.label_dict = label_dict
        self.input_dims = input_dims

        self.omics_keys = ['rna', 'methyl', 'protein', 'mirna']

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]

        sample = {}

        # WSI
        wsi_tensor = self.wsi_features.get(pid, None)
        if wsi_tensor is not None:
            sample['wsi_feat'] = wsi_tensor.mean(dim=0)
        else:
            sample['wsi_feat'] = torch.zeros(512, dtype=torch.float32)

        # Omics
        for k in self.omics_keys:
            feat = self.omics_dict.get(k, {}).get(pid, None)
            if feat is not None and feat.numel() == self.input_dims[k]:
                sample[k] = feat.float()
            elif feat is not None and feat.numel() != self.input_dims[k]:
                # pad or truncate
                fixed = torch.zeros(self.input_dims[k], dtype=torch.float32)
                length = min(self.input_dims[k], feat.numel())
                fixed[:length] = feat[:length]
                sample[k] = fixed
            else:
                sample[k] = torch.zeros(self.input_dims[k], dtype=torch.float32)

        # Labels
        label = self.label_dict.get(pid, {})
        sample['til_label'] = label.get('til', torch.tensor([0, 0, 0, 0], dtype=torch.float32))  # default: no TIL
        sample['response_label'] = label.get('response', torch.tensor([-1.0], dtype=torch.float32))  # mask if unavailable
        sample['status_label'] = label.get('status', torch.tensor([-1.0], dtype=torch.float32))  # mask if unavailable
        sample['survival_time'] = label.get('survival_time', torch.tensor([-1.0], dtype=torch.float32))
        if sample['survival_time'].item() != -1.0:
            sample['survival_time'] = torch.log1p(sample['survival_time'])

        # Add patient ID for tracking
        sample['id'] = pid

        return sample
