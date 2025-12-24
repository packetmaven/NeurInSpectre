from typing import List, Tuple, Optional, Union, Dict, Any, TYPE_CHECKING
import torch
# from .cli.device_utils import select_device  # Remove this line
import warnings
import logging

# Only import the actual implementation when needed
try:
    from .ecc_activation_steganography import (
        eccactivationsteganography as ActivationSteganography,
        steganographyresult
    )
    ECC_AVAILABLE = True
except ImportError:
    ECC_AVAILABLE = False
    if TYPE_CHECKING:
        # For type checking
        class ActivationSteganography: ...
        class steganographyresult: ...
    # Provide a minimal runtime fallback for encode
    class ActivationSteganography:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def encode_payload(self, prompt: str, payload_bits: List[int], target_neurons: List[int]) -> str:
            marker = ','.join(map(str, payload_bits))
            return f"{prompt} [STEG:{marker}]"

# Lazy import function for device_utils
def _get_select_device():
    """Lazy import of select_device to avoid CLI import at module level"""
    try:
        from .cli.device_utils import select_device
        return select_device
    except ImportError:
        # Fallback device selection if CLI not available
        def fallback_select_device():
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return fallback_select_device 