from typing import Tuple

import torch
from torch.cuda.amp import custom_fwd, custom_bwd

from ._ext import cforward, cbackward

class AiLUTTransformFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx,
                img: torch.Tensor,
                lut: torch.Tensor,
                vertices: torch.tensor) -> torch.Tensor:

        img = img.contiguous()
        lut = lut.contiguous()
        vertices = vertices.contiguous()

        assert img.ndimension() == 4, \
            "only support 2D image with batch and channel dimensions (4D tensor)"
        assert lut.ndimension() in [5], \
            "only support 3D lookup table with batch dimension (5D tensor)"
        assert vertices.ndimension() == 3, \
            "only support 1D vertices list with batch and channel dimensions (3D tensor)"

        output = img.new_zeros((img.size(0), lut.size(1), img.size(2), img.size(3)))
        cforward(img, lut, vertices, output)

        ctx.save_for_backward(img, lut, vertices)

        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:

        grad_output = grad_output.contiguous()

        img, lut, vertices = ctx.saved_tensors

        grad_img = torch.zeros_like(img)
        grad_lut = torch.zeros_like(lut)
        grad_ver = torch.zeros_like(vertices)
        
        # Warning('The backward implementation of AiLUT-Transform is currently '
        #         'not available until submission acceptance. All gradients will '
        #         'be set to zeros.')
        cbackward(grad_output, img, lut, vertices,
            grad_img, grad_lut, grad_ver)

        return grad_img, grad_lut, grad_ver
    
    
def ailut_transform(
    img: torch.Tensor,
    lut: torch.Tensor,
    vertices: torch.Tensor) -> torch.Tensor:
    r"""Adaptive Interval 3D Lookup Table Transform (AiLUT-Transform).
    
    Args:
        img (torch.Tensor): input image of shape (b, 3, h, w).
        lut (torch.Tensor): output values of the 3D LUT, shape (b, 3, d, d, d).
        vertices (torch.Tensor): sampling coordinates along each dimension of
            the 3D LUT, shape (b, 3, d).
    Returns:
        torch.Tensor: transformed image of shape (b, 3, h, w).
    """
    return AiLUTTransformFunction.apply(img, lut, vertices)


def lut_transform(
    img: torch.Tensor,
    lut: torch.Tensor) -> torch.Tensor:
    r"""Standard 3D Lookup Table Transform.
    
    Args:
        img (torch.Tensor): input image of shape (b, 3, h, w).
        lut (torch.Tensor): output values of the 3D LUT, shape (b, 3, d, d, d).
    Returns:
        torch.Tensor: transformed image of shape (b, 3, h, w).
    """
    n_vertices = lut.shape[-1]
    uniform_vertices = torch.arange(n_vertices, device=img.device) \
                            .div(n_vertices - 1).unsqueeze(0) \
                            .repeat(img.shape[0], img.shape[1], 1)
    return AiLUTTransformFunction.apply(img, lut, uniform_vertices)
