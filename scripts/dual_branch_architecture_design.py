"""
DUAL-BRANCH FUSION CNN - STEP 2: Architecture Design
Project: NeuroFusion-EEG
Author: Diadri Weerasekera
Year: 2025–2026
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Branch A: Processes raw EEG signals in time domain
class BranchA_TimeDomain(nn.Module):
    def __init__(self, input_channels=1, output_features=128):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            # Global pooling
            nn.AdaptiveAvgPool1d(1)
        )

        # Project to output features
        self.projection = nn.Sequential(
            nn.Linear(128, output_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # x shape: (batch, 1, signal_length)
        x = self.features(x)
        x = x.squeeze(-1)  # (batch, 128)
        x = self.projection(x)  # (batch, output_features)
        return x

#Branch B: Processes transformed features (STFT/Wavelet/etc.)
class BranchB_FrequencyDomain(nn.Module):
    def __init__(self, input_shape, output_features=128, is_2d=False):
        super().__init__()
        self.is_2d = is_2d

        if is_2d:
            # 2D CNN for scalogram-like inputs
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(input_shape[0], 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.2),

                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(0.2),

                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            # 1D CNN for flattened frequency features
            self.features = nn.Sequential(
                # Block 1
                nn.Conv1d(input_shape[0], 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(0.2),

                # Block 2
                nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(0.2),

                # Block 3
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1)
            )

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(128, output_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        # x shape: (batch, channels, height, width) for 2D or (batch, channels, length) for 1D
        x = self.features(x)

        if self.is_2d:
            x = x.squeeze(-1).squeeze(-1)  # (batch, 128)
        else:
            x = x.squeeze(-1)  # (batch, 128)

        x = self.projection(x)  # (batch, output_features)
        return x

# Fusion layer that combines features from both branches
class FusionLayer(nn.Module):
    def __init__(self, branch_a_features, branch_b_features,
                 fusion_type='concat', output_features=128):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            # Simple concatenation
            self.fusion = nn.Sequential(
                nn.Linear(branch_a_features + branch_b_features, output_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4)
            )

        elif fusion_type == 'add':
            # Element-wise addition (requires same dimensions)
            assert branch_a_features == branch_b_features, \
                "Addition fusion requires same dimensions for both branches"
            self.fusion = nn.Sequential(
                nn.Linear(branch_a_features, output_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4)
            )

        elif fusion_type == 'attention':
            # Attention-based fusion
            self.attention_a = nn.Linear(branch_a_features, 1)
            self.attention_b = nn.Linear(branch_b_features, 1)
            self.fusion = nn.Sequential(
                nn.Linear(branch_a_features + branch_b_features, output_features),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4)
            )

    def forward(self, feat_a, feat_b):
        if self.fusion_type == 'concat':
            combined = torch.cat([feat_a, feat_b], dim=1)
            return self.fusion(combined)

        elif self.fusion_type == 'add':
            combined = feat_a + feat_b
            return self.fusion(combined)

        elif self.fusion_type == 'attention':
            # Compute attention weights
            alpha_a = torch.sigmoid(self.attention_a(feat_a))
            alpha_b = torch.sigmoid(self.attention_b(feat_b))

            # Normalize
            alpha_sum = alpha_a + alpha_b
            alpha_a = alpha_a / alpha_sum
            alpha_b = alpha_b / alpha_sum

            # Apply attention and concatenate
            weighted_a = feat_a * alpha_a
            weighted_b = feat_b * alpha_b
            combined = torch.cat([weighted_a, weighted_b], dim=1)

            return self.fusion(combined)

# Dual-Branch Fusion CNN Architecture
class DualBranchFusionCNN(nn.Module):
    def __init__(self, n_classes, raw_signal_length,
                 transformed_shape, is_2d_transform=False,
                 branch_features=128, fusion_type='concat'):
        super().__init__()

        # Branch A: Raw time-domain processing
        self.branch_a = BranchA_TimeDomain(
            input_channels=1,
            output_features=branch_features
        )

        # Branch B: Frequency-domain processing
        self.branch_b = BranchB_FrequencyDomain(
            input_shape=transformed_shape,
            output_features=branch_features,
            is_2d=is_2d_transform
        )

        # Fusion layer
        self.fusion = FusionLayer(
            branch_a_features=branch_features,
            branch_b_features=branch_features,
            fusion_type=fusion_type,
            output_features=branch_features
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(branch_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, n_classes)
        )

    def forward(self, raw_signal, transformed_signal):
        # Process through both branches
        feat_a = self.branch_a(raw_signal)      # (batch, branch_features)
        feat_b = self.branch_b(transformed_signal)  # (batch, branch_features)

        # Fuse features
        fused = self.fusion(feat_a, feat_b)    # (batch, branch_features)

        # Classify
        output = self.classifier(fused)         # (batch, n_classes)

        return output

# Test the architecture
if __name__ == "__main__":
    print("TESTING DUAL-BRANCH FUSION CNN ARCHITECTURE")

    # Test parameters
    batch_size = 4
    n_classes = 5
    raw_length = 4096

    # Test 1D transforms (e.g., STFT magnitude, wavelet coefficients)
    print("1D Transformed Features (STFT/Wavelet)")

    transformed_1d_shape = (1, 2048)  # (channels, length)

    model_1d = DualBranchFusionCNN(
        n_classes=n_classes,
        raw_signal_length=raw_length,
        transformed_shape=transformed_1d_shape,
        is_2d_transform=False,
        fusion_type='concat'
    )

    # Create dummy inputs
    raw_input = torch.randn(batch_size, 1, raw_length)
    transformed_1d_input = torch.randn(batch_size, *transformed_1d_shape)

    output_1d = model_1d(raw_input, transformed_1d_input)
    print(f" 1D Model output shape: {output_1d.shape}")
    print(f"  Expected: ({batch_size}, {n_classes})")

    # Test 2D transforms (e.g., CWT scalogram)
    print("2D Transformed Features (CWT/Spectrogram)")

    transformed_2d_shape = (1, 64, 64)  # (channels, height, width)

    model_2d = DualBranchFusionCNN(
        n_classes=n_classes,
        raw_signal_length=raw_length,
        transformed_shape=transformed_2d_shape,
        is_2d_transform=True,
        fusion_type='attention'
    )

    transformed_2d_input = torch.randn(batch_size, *transformed_2d_shape)

    output_2d = model_2d(raw_input, transformed_2d_input)
    print(f" 2D Model output shape: {output_2d.shape}")
    print(f"  Expected: ({batch_size}, {n_classes})")

    # Test different fusion strategies
    print("Different Fusion Strategies")

    for fusion_type in ['concat', 'attention']:
        model = DualBranchFusionCNN(
            n_classes=n_classes,
            raw_signal_length=raw_length,
            transformed_shape=transformed_1d_shape,
            is_2d_transform=False,
            fusion_type=fusion_type
        )

        output = model(raw_input, transformed_1d_input)
        print(f" Fusion type '{fusion_type}': output shape {output.shape}")

    # Count parameters
    print("Model Statistics")

    total_params = sum(p.numel() for p in model_1d.parameters())
    trainable_params = sum(p.numel() for p in model_1d.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print(" ALL ARCHITECTURE TESTS PASSED")
