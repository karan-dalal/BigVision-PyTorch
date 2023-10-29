import ml_collections
import os

def get_s16_config():   
    """Returns the ViT-S/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patch_size = 16
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1536
    config.transformer.num_heads = 6
    config.transformer.num_layers = 12
    return config

def get_t16_config():   
    """Returns the ViT-T/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patch_size = 16
    config.hidden_size = 192
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 3
    config.transformer.num_layers = 12
    return config