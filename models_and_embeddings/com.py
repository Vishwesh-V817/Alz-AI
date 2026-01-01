import torch
from attention import DualAttentionResNet18MRI

# 1. Load your current heavy file
checkpoint = torch.load('dual_attention_59_59.pth', map_location=torch.device('cpu'))

# 2. Extract just the weights (state_dict)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    weights = checkpoint['model_state_dict']
else:
    # If it's the whole model object, get the state_dict
    weights = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint

# 3. Save ONLY the weights
torch.save(weights, 'dual_attention_59_59_slim.pth')    