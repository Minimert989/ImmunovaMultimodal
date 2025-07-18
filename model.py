import torch
import torch.nn as nn
import torch.nn.functional as F


class OmicsSubEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, n_heads=4, n_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.project = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, F) → treat F as sequence
        x = x.unsqueeze(1)  # (B, 1, F)
        x = self.project(x)  # (B, 1, hidden_dim)
        x = self.transformer(x)  # (B, 1, hidden_dim)
        return x.squeeze(1)  # (B, hidden_dim)

class OmicsEncoder(nn.Module):
    def __init__(self, rna_dim=1000, methyl_dim=512, prot_dim=256, mirna_dim=128, hidden_dim=512):
        super().__init__()
        self.rna = OmicsSubEncoder(rna_dim, hidden_dim)
        self.methyl = OmicsSubEncoder(methyl_dim, hidden_dim)
        self.prot = OmicsSubEncoder(prot_dim, hidden_dim)
        self.mirna = OmicsSubEncoder(mirna_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, rna, methyl, prot, mirna):
        rna_h = self.rna(rna)
        methyl_h = self.methyl(methyl)
        prot_h = self.prot(prot)
        mirna_h = self.mirna(mirna)
        concat = torch.cat([rna_h, methyl_h, prot_h, mirna_h], dim=-1)
        return self.fusion(concat)



class AttentionMIL(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, N, D) where N is number of patches
        h = self.tanh(self.fc1(x))        # (B, N, H)
        a = self.attn(h)                  # (B, N, 1)
        a = torch.softmax(a, dim=1)       # (B, N, 1)
        z = torch.sum(a * x, dim=1)       # (B, D)
        return z


class CrossModalFusion(nn.Module):
    def __init__(self, input_dim=512, n_heads=4, n_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

    def forward(self, wsi_feat, omics_feat):
        # wsi_feat: (B, N, D), omics_feat: (B, D)
        B = wsi_feat.size(0)
        omics_feat = omics_feat.unsqueeze(1)  # (B, 1, D)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # shape 로그 추가
       

        tokens = torch.cat([cls_token, wsi_feat, omics_feat], dim=1)  # (B, 1+N+1, D)
        fused = self.transformer(tokens)  # (B, 1+N+1, D)

        return fused[:, 0]  # return CLS token


class ImmunovaMultimodalModel(nn.Module):
    def __init__(self, input_dims, hidden_dim=512, til_classes=4):
        super().__init__()
        self.omics_encoder = OmicsEncoder(*input_dims, hidden_dim=hidden_dim)
        self.wsi_patch_proj = nn.Linear(512, hidden_dim)  # 추가: WSI patch feature projection
        self.fusion = CrossModalFusion(input_dim=hidden_dim)

        self.til_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, til_classes),
        )

        self.response_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # binary classification
        )

        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # regression
        )

    def forward(self, wsi_feat=None, rna=None, methyl=None, prot=None, mirna=None):
        omics_h = self.omics_encoder(rna, methyl, prot, mirna)  # (B, hidden_dim)

        if wsi_feat is not None:
            wsi_feat_proj = self.wsi_patch_proj(wsi_feat)  # (B, D)
            wsi_feat_proj = wsi_feat_proj.unsqueeze(1)     # (B, 1, D)
            fused = self.fusion(wsi_feat_proj, omics_h)
        else:
            fused = omics_h  # fallback for omics-only case

        til_pred = self.til_head(fused)
        response_pred = self.response_head(fused)
        survival_pred = self.survival_head(fused)
        return til_pred, response_pred, survival_pred