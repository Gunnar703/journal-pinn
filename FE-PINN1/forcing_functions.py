import torch
import numpy as np

from utils import interp


def _get_gaussian(t0, t1, amplitude, total_dimensions, force_dimension, device="cuda"):
    mean = (t1 - t0) / 2

    def gaussian_force_signal(t):
        f = torch.zeros((total_dimensions, 1 if not t.shape else len(t)), device=device)
        f[force_dimension] = torch.exp(-((t.squeeze() - mean) ** 2)) * amplitude
        return f

    return gaussian_force_signal


def _get_sine(t0, t1, amplitude, total_dimensions, force_dimension, device="cuda"):
    time_scale = t1 - t0
    frequency = 2 * torch.pi / time_scale

    def sine_force_signal(t):
        f = torch.zeros((total_dimensions, 1 if not t.shape else len(t)), device=device)
        f[force_dimension] = torch.sin(frequency * t) * amplitude
        return f

    return sine_force_signal


def _get_chirp(t0, t1, amplitude, total_dimensions, force_dimension, device="cuda"):
    load = np.loadtxt("load.txt")
    time = np.linspace(t0, t1, len(load))

    load = torch.as_tensor(load, device=device) / load.max()
    time = torch.as_tensor(time, device=device)

    # print("time:", time.shape, "\nload:", load.shape)

    def chirp_signal(t):
        f = torch.zeros((total_dimensions, 1 if not t.shape else len(t)), device=device)
        v = interp(t, time, load).squeeze() * amplitude
        # print("f:", f.shape, "\nv:", v.shape)

        f[force_dimension] = v
        return f

    return chirp_signal


functions = {"gaussian": _get_gaussian, "sine": _get_sine, "chirp": _get_chirp}


def get_function(
    name, t0, t1, amplitude, total_dimensions, force_dimension, device="cuda"
):
    return functions[name](t0, t1, amplitude, total_dimensions, force_dimension, device)
