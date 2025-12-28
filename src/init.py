import torch
import torch.nn as nn
from typing import List, Union


def reinit_qk_orthogonal(
    model: nn.Module,
    config,
    first_k: int,
    layers: Union[List[int], int],
    init_scale: float = 1.0,
) -> None:
    """
    Reinitialize the first k attention heads' K and Q projection matrices to be orthogonal to each other.

    This function creates Q and K projections that are orthogonal to each other by sampling from
    different subspaces of a single orthogonal matrix.

    Args:
        model: OLMo transformer model
        config: TransformerConfig object with n_heads and d_model attributes
        first_k: Number of attention heads to reinitialize (the first k heads)
        layers: Layer indices to apply reinitialization to. Can be a single int or list of ints.
        init_scale: Scaling factor for orthogonal initialization (gain parameter)

    Assumes OLMo architecture:
        - model.transformer.blocks[layer_idx].attention.{w_q, w_k} or model.blocks[layer_idx].attention.{w_q, w_k}
        - config with n_heads and d_model attributes
    """
    # Normalize layers to a list
    if isinstance(layers, int):
        layers = [layers]

    # Get architecture parameters from config
    n_heads = config.block.attention.n_heads
    d_model = config.d_model
    head_dim = d_model // n_heads

    if first_k > n_heads:
        raise ValueError(
            f"first_k ({first_k}) cannot be greater than n_heads ({n_heads})"
        )

    # Number of dimensions to reinitialize
    d_k = first_k * head_dim

    # Need at least 2*d_k dimensions in d_model
    if 2 * d_k > d_model:
        raise ValueError(
            f"Need at least {2 * d_k} dimensions in d_model to create orthogonal Q and K, "
            f"but d_model={d_model}"
        )

    # Access transformer blocks
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        blocks = model.transformer.blocks
    elif hasattr(model, "blocks"):
        blocks = model.blocks
    else:
        raise ValueError(
            "Could not find transformer blocks. Expected model.transformer.blocks or model.blocks"
        )

    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= len(blocks):
            raise ValueError(f"Layer index {layer_idx} out of range [0, {len(blocks)})")

        block = blocks[f"{layer_idx}"]
        attn = block.attention

        # OLMo uses w_q and w_k (not q_proj/k_proj)
        if not (hasattr(attn, "w_q") and hasattr(attn, "w_k")):
            raise ValueError(
                f"Could not find w_q and w_k in layer {layer_idx}. "
                f"Available attributes: {dir(attn)}"
            )

        # Reinitialize Q and K projections to be orthogonal to each other
        _reinit_qk_mutually_orthogonal(
            attn.w_q, attn.w_k, first_k, head_dim, d_model, init_scale
        )

        print(
            f"Reinitialized first {first_k} heads in layer {layer_idx} with mutually orthogonal Q and K"
        )


def _reinit_qk_mutually_orthogonal(
    q_proj: nn.Linear,
    k_proj: nn.Linear,
    first_k_heads: int,
    head_dim: int,
    d_model: int,
    init_scale: float,
) -> None:
    """
    Reinitialize Q and K projections to be orthogonal to each other.

    Creates a single orthogonal matrix and assigns different orthogonal subspaces to Q and K,
    ensuring their column spaces are orthogonal.

    Args:
        q_proj: Q projection layer
        k_proj: K projection layer
        first_k_heads: Number of heads to reinitialize
        head_dim: Dimension of each attention head
        d_model: Model dimension
        init_scale: Scaling factor for orthogonal initialization
    """
    q_weight = q_proj.weight.data
    k_weight = k_proj.weight.data
    original_dtype = q_weight.dtype
    device = q_weight.device

    # Number of dimensions to reinitialize
    d_k = first_k_heads * head_dim
    head_slice = slice(0, d_k)

    # Create a full orthogonal matrix (d_model x d_model)
    # Handle bfloat16 (QR decomposition doesn't support it on CPU or CUDA)
    if original_dtype == torch.bfloat16:
        full_matrix = torch.empty(d_model, d_model, dtype=torch.float32, device=device)
    else:
        full_matrix = torch.empty(d_model, d_model, dtype=original_dtype, device=device)

    nn.init.orthogonal_(full_matrix, gain=init_scale)

    # Split into two orthogonal subspaces
    # Q gets the first d_k columns, K gets the next d_k columns
    # Transpose to get rows (since weights are [out_features, in_features])
    Q = full_matrix[:, :d_k].T.clone()  # Shape: [d_k, d_model]
    K = full_matrix[:, d_k : 2 * d_k].T.clone()  # Shape: [d_k, d_model]

    # Convert back to original dtype if necessary
    if original_dtype == torch.bfloat16:
        Q = Q.to(original_dtype)
        K = K.to(original_dtype)

    # Assign to weight matrices
    q_weight[head_slice] = Q
    k_weight[head_slice] = K

    # Reinitialize biases if they exist
    if q_proj.bias is not None:
        q_proj.bias.data[head_slice].zero_()
    if k_proj.bias is not None:
        k_proj.bias.data[head_slice].zero_()


def reinit_qk_scale(
    model: nn.Module,
    config,
    first_k: int,
    layers: Union[List[int], int],
    init_scale: float = 1.0,
) -> None:
    """
    Scale the first k attention heads' K and Q projection matrices by a constant factor (in-place).

    Args:
        model: OLMo transformer model
        config: TransformerConfig object with n_heads and d_model attributes
        first_k: Number of attention heads to scale (the first k heads)
        layers: Layer indices to apply scaling to. Can be a single int or list of ints.
        init_scale: Scaling factor to multiply the weights by

    Assumes OLMo architecture:
        - model.transformer.blocks[layer_idx].attention.{w_q, w_k} or model.blocks[layer_idx].attention.{w_q, w_k}
        - config with n_heads and d_model attributes
    """
    # Normalize layers to a list
    if isinstance(layers, int):
        layers = [layers]

    # Get architecture parameters from config
    n_heads = config.block.attention.n_heads
    d_model = config.d_model
    head_dim = d_model // n_heads

    if first_k > n_heads:
        raise ValueError(
            f"first_k ({first_k}) cannot be greater than n_heads ({n_heads})"
        )

    # Access transformer blocks
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        blocks = model.transformer.blocks
    elif hasattr(model, "blocks"):
        blocks = model.blocks
    else:
        raise ValueError(
            "Could not find transformer blocks. Expected model.transformer.blocks or model.blocks"
        )

    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= len(blocks):
            raise ValueError(f"Layer index {layer_idx} out of range [0, {len(blocks)})")

        block = blocks[f"{layer_idx}"]
        attn = block.attention

        # OLMo uses w_q and w_k (not q_proj/k_proj)
        if not (hasattr(attn, "w_q") and hasattr(attn, "w_k")):
            raise ValueError(
                f"Could not find w_q and w_k in layer {layer_idx}. "
                f"Available attributes: {dir(attn)}"
            )

        # Scale Q and K projections for the first k heads
        _scale_projection(attn.w_q, first_k, head_dim, init_scale)
        _scale_projection(attn.w_k, first_k, head_dim, init_scale)

        print(
            f"Scaled first {first_k} heads in layer {layer_idx} by factor {init_scale}"
        )


def _scale_projection(
    proj: nn.Linear, first_k_heads: int, head_dim: int, scale: float
) -> None:
    """
    Scale the first k heads of a projection matrix by a constant factor (in-place).

    Args:
        proj: Linear projection layer (w_q or w_k)
        first_k_heads: Number of heads to scale
        head_dim: Dimension of each attention head
        scale: Scaling factor
    """
    weight = proj.weight.data

    # Scale the first k heads
    # Each head's parameters span head_dim rows
    head_slice = slice(0, first_k_heads * head_dim)

    # Multiply weights by scale factor
    weight[head_slice].mul_(scale)

    # Also scale bias if it exists (though OLMo typically uses bias=False)
    if proj.bias is not None:
        proj.bias.data[head_slice].mul_(scale)


def reinit_qk_zeros(
    model: nn.Module,
    config,
    first_k: int,
    layers: Union[List[int], int],
    init_scale: float = 1.0,
) -> None:
    """
    Reinitialize the first k attention heads' K and Q projection matrices to zeros.

    Args:
        model: OLMo transformer model
        config: TransformerConfig object with n_heads and d_model attributes
        first_k: Number of attention heads to reinitialize (the first k heads)
        layers: Layer indices to apply reinitialization to. Can be a single int or list of ints.
        init_scale: Unused parameter kept for API compatibility with reinit_qk_orthogonal

    Assumes OLMo architecture:
        - model.transformer.blocks[layer_idx].attention.{w_q, w_k} or model.blocks[layer_idx].attention.{w_q, w_k}
        - config with n_heads and d_model attributes
    """
    # Normalize layers to a list
    if isinstance(layers, int):
        layers = [layers]

    # Get architecture parameters from config
    n_heads = config.block.attention.n_heads
    d_model = config.d_model
    head_dim = d_model // n_heads

    if first_k > n_heads:
        raise ValueError(
            f"first_k ({first_k}) cannot be greater than n_heads ({n_heads})"
        )

    # Access transformer blocks
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        blocks = model.transformer.blocks
    elif hasattr(model, "blocks"):
        blocks = model.blocks
    else:
        raise ValueError(
            "Could not find transformer blocks. Expected model.transformer.blocks or model.blocks"
        )

    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= len(blocks):
            raise ValueError(f"Layer index {layer_idx} out of range [0, {len(blocks)})")

        block = blocks[f"{layer_idx}"]
        attn = block.attention

        # OLMo uses w_q and w_k (not q_proj/k_proj)
        if not (hasattr(attn, "w_k")):
            raise ValueError(
                f"Could not find w_q and w_k in layer {layer_idx}. "
                f"Available attributes: {dir(attn)}"
            )

        # Reinitialize K to zeros for the first k heads
        _reinit_projection_zeros(attn.w_k, first_k, head_dim)

        print(f"Reinitialized first {first_k} heads to zeros in layer {layer_idx}")


def _reinit_projection_zeros(
    proj: nn.Linear, first_k_heads: int, head_dim: int
) -> None:
    """
    Reinitialize the first k heads of a projection matrix to zeros.

    Args:
        proj: Linear projection layer (w_q or w_k)
        first_k_heads: Number of heads to reinitialize
        head_dim: Dimension of each attention head
    """
    weight = proj.weight.data

    # Reinitialize the first k heads
    # Each head's parameters span head_dim rows
    head_slice = slice(0, first_k_heads * head_dim)

    # Set weights to zero
    weight[head_slice].zero_()

    # Also reinitialize bias if it exists (though OLMo typically uses bias=False)
    if proj.bias is not None:
        proj.bias.data[head_slice].zero_()
