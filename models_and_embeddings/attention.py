import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionBlock(nn.Module):
    """
    Attention mechanism that learns which slices are most informative.
    """
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x: (Batch, 16, Features)
        weights = self.attention(x)  # (Batch, 16, 1)
        weights = torch.softmax(weights, dim=1)  # Normalize across 16 slices
        
        # Weighted sum: model chooses which slices to listen to
        out = torch.sum(x * weights, dim=1)  # (Batch, Features)
        return out, weights


class DualAttentionResNet18MRI(nn.Module):
    """
    ResNet18 with DUAL Attention-Based Multi-Instance Learning.
    
    Key Innovation: Two attention heads that can focus on different aspects
    """
    def __init__(self, num_classes=3):
        super().__init__()
        
        # 1. Load pretrained ResNet18
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Extract feature extractor (everything except final FC layer)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        
        # 2. Replace first conv for 1-channel grayscale MRI
        self.feature_extractor[0] = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 3. DUAL Attention Heads (can learn different patterns)
        self.attention_head_1 = AttentionBlock(512, hidden_dim=128)
        self.attention_head_2 = AttentionBlock(512, hidden_dim=128)
        
        # 4. Fusion layer to combine both attention outputs
        self.fusion = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 5. Final classifier
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, S, C, H, W) where S=16 slices
            return_attention: If True, also returns attention weights
        
        Returns:
            logits: (B, num_classes)
            attn_weights: tuple of (attn1, attn2) - optional
        """
        # Handle both 5D and 4D inputs
        if len(x.shape) == 4:
            # Single slice input (B, C, H, W)
            feats = self.feature_extractor(x)
            feats = feats.view(feats.size(0), -1)
            # For single slice, bypass attention
            feats_combined = torch.cat([feats, feats], dim=1)
            feats_fused = self.fusion(feats_combined)
            logits = self.classifier(feats_fused)
            
            if return_attention:
                return logits, (None, None)
            return logits
        
        # Multi-slice input (B, S, C, H, W)
        B, S, C, H, W = x.shape
        
        # Flatten batch and slices: (B*S, C, H, W)
        x = x.view(B * S, C, H, W)
        
        # Extract features for all slices
        feats = self.feature_extractor(x)  # (B*S, 512, 1, 1)
        feats = feats.view(B, S, 512)      # (B, S, 512)
        
        # --- DUAL ATTENTION BREAKTHROUGH ---
        # Two attention heads can focus on different aspects
        aggregated_feat_1, attn_weights_1 = self.attention_head_1(feats)
        aggregated_feat_2, attn_weights_2 = self.attention_head_2(feats)
        
        # Combine both attention outputs
        feats_combined = torch.cat([aggregated_feat_1, aggregated_feat_2], dim=1)
        feats_fused = self.fusion(feats_combined)
        
        # Final classification
        logits = self.classifier(feats_fused)
        
        if return_attention:
            return logits, (attn_weights_1, attn_weights_2)
        return logits

if __name__ == '__main__':
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DualAttentionResNet18MRI(num_classes=3)
        model = model.to(device)

        # Count parameters
        def count_parameters(model):
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total:,}")
            print(f"Trainable parameters: {trainable:,}")
            return trainable

        count_parameters(model)
        print(f"Dual-Attention ResNet18 MRI loaded on {device}")
