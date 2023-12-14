import os
import torch
import numpy as np

from torch import Tensor


def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    # if len(x.shape) < 2:
    #     x = x.reshape(-1, 1)
    # if len(xp.shape) < 2:
    #     xp = xp.reshape(-1, 1)

    # m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    # b = fp[:-1] - (m * xp[:-1])

    # indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    # indicies = torch.clamp(indicies, 0, len(m) - 1)
    # print("x", x, "\nxp:", xp)

    # return m[indicies] * x + b[indicies]
    return torch.from_numpy(
        np.array(np.interp(x.cpu().numpy(), xp.cpu().numpy(), fp.cpu().numpy()))
    )


def make_folder(name):
    original_name = name

    count = 1
    while os.path.exists(name):
        name = original_name + "(%d)" % count
        count += 1

    os.makedirs(name)
    return name
