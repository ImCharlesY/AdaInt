import os.path as osp
from typing import Tuple

import torch
from torch.cuda.amp import custom_fwd, custom_bwd

try:
    import ailut
    from pkg_resources import get_distribution
    ailut.__version__ = get_distribution('ailut').version
except ImportError:
    from torch.utils.cpp_extension import load
    cdir = osp.join(osp.dirname(__file__), 'csrc')
    ailut = load(
        name='ailut',
        sources=[
            osp.join(cdir, 'ailut_transform_cuda.cpp'),
            osp.join(cdir, 'ailut_transform_kernel.cu')
        ]
    )
    ailut.__version__ = 'dev'


class AILutTransformFunction(torch.autograd.Function):

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
        ailut.forward(img, lut, vertices, output)

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

        ailut.backward(grad_output, img, lut, vertices,
            grad_img, grad_lut, grad_ver)

        return grad_img, grad_lut, grad_ver


ailut_transform =  AILutTransformFunction.apply
