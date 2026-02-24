"""
Model factory for LatBOND Lite v8.
Creates any of the 3 architectures with consistent interface.
"""

from .streaming_cnn import StreamingCNN, create_streaming_cnn
from .causal_transformer import CausalTransformer, create_causal_transformer
from .tcn import TCN, create_tcn


MODEL_REGISTRY = {
    "streaming_cnn": create_streaming_cnn,
    "causal_transformer": create_causal_transformer,
    "tcn": create_tcn,
}


def create_model(arch_name, config):
    """
    Create a model by architecture name.
    
    Args:
        arch_name: One of 'streaming_cnn', 'causal_transformer', 'tcn'
        config: Config object
    Returns:
        nn.Module
    """
    if arch_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {arch_name}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[arch_name](config)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
