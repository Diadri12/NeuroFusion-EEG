"""
Model Architecture for EEG Classification
Auto-generated - DO NOT MODIFY
"""

import torch
import torch.nn as nn

class BranchA_LayerNorm(nn.Module):
    """CNN branch for raw signal processing"""
    def __init__(self, signal_length):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 7, 2, 3)
        self.norm1 = nn.GroupNorm(1, 32)
        self.conv2 = nn.Conv1d(32, 64, 5, 1, 2)
        self.norm2 = nn.GroupNorm(1, 64)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, 1)
        self.norm3 = nn.GroupNorm(1, 128)
        self.conv4 = nn.Conv1d(128, 256, 3, 1, 1)
        self.norm4 = nn.GroupNorm(1, 256)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(nn.Linear(256, 64), nn.ReLU())
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.dropout(self.pool(self.relu(self.norm1(self.conv1(x)))))
        x = self.dropout(self.pool(self.relu(self.norm2(self.conv2(x)))))
        x = self.dropout(self.pool(self.relu(self.norm3(self.conv3(x)))))
        x = self.dropout(self.pool(self.relu(self.norm4(self.conv4(x)))))
        return self.embedding(self.gap(x).squeeze(-1))

class BranchB_Features(nn.Module):
    """MLP branch for handcrafted features"""
    def __init__(self, n_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
    
    def forward(self, x):
        return self.model(x)

class DualBranchModel(nn.Module):
    """Complete dual-branch architecture"""
    def __init__(self, signal_length, n_features, n_classes):
        super().__init__()
        self.branch_a = BranchA_LayerNorm(signal_length)
        self.branch_b = BranchB_Features(n_features)
        self.classifier = nn.Sequential(
            nn.Linear(96, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, signals, features):
        emb_a = self.branch_a(signals)
        emb_b = self.branch_b(features)
        fused = torch.cat([emb_a, emb_b], dim=1)
        return self.classifier(fused)
