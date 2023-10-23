from functools import lru_cache
import torch


def set_up_4_dof():
    x = torch.as_tensor((0, 5), device="cuda")
    y = torch.as_tensor((0, 5), device="cuda")
    x, y = torch.meshgrid(x, y, indexing="xy")

    nodes_y, nodes_x = x.shape
    elements_y, elements_x = nodes_y - 1, nodes_x - 1
    _fixed_indices = torch.as_tensor((0, 1, 2, 3))
    ndim = 2 * nodes_y * nodes_x
    free_indices = torch.arange(ndim)
    free_indices = free_indices[~torch.isin(free_indices, _fixed_indices)]

    sens_dim = torch.as_tensor((1, 3))
    ndim = len(free_indices)
    fdim = 3

    E = torch.ones((elements_y, elements_x), device="cuda") * 6.923076923076923192e07
    nu = torch.ones((elements_y, elements_x), device="cuda") * 0.3
    rho = torch.ones((elements_y, elements_x), device="cuda") * 2e3
    a0, a1 = rayleigh_damping_parameters()
    return x, y, E, nu, rho, a0, a1, fdim, free_indices, ndim, sens_dim


def _diff(m, c, k, f, X, t, dim):
    u = X[:dim]
    u_t = X[dim:]
    u_tt = torch.linalg.solve(m, f - c @ u_t - k @ u)
    return torch.vstack((u_t, u_tt))


def integrate_rk4(m, c, k, ffun, t, device="cuda"):
    dim = len(k)
    n_timesteps = len(t)
    X_history = torch.zeros((2 * dim, n_timesteps), device=device)

    for i in range(1, n_timesteps):
        dt = t[i] - t[i - 1]
        x = X_history[:, i - 1 : i]
        time = t[i - 1]
        f = ffun(time)

        k1 = _diff(m, c, k, f, x, time, dim)
        k2 = _diff(m, c, k, f, x + dt * k1 / 2, time + dt / 2, dim)
        k3 = _diff(m, c, k, f, x + dt * k2 / 2, time + dt / 2, dim)
        k4 = _diff(m, c, k, f, x + dt * k3, time + dt, dim)
        X_history[:, i : i + 1] = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    u = X_history[:dim, :]
    u_t = X_history[dim:, :]
    return u, u_t


def get_mck(
    x: torch.Tensor,
    y: torch.Tensor,
    E: torch.Tensor,
    nu: torch.Tensor,
    rho: torch.Tensor,
    a0: float,
    a1: float,
    free_indices: torch.Tensor,
    device="cuda",
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Get the mass, stiffness, and damping matrices for an FE model consisting of square quad elements arranged in a rectangular grid.

    Args:
        x (torch.Tensor): N x M grid of nodal x-coordinates. Must be monotonically increasing along axis 1.
        y (torch.Tensor): N x M grid of nodal y-coordinates. Must be monotonically increasing along axis 0.
        E (torch.Tensor): (N - 1) x (M - 1) grid of elemental Young's moduli.
        nu (torch.Tensor): (N - 1) x (M - 1) grid of elemental Poisson's ratios.
        rho (torch.Tensor): (N - 1) x (M - 1) grid of elemental Young's densities.
        a0 (float): mass damping coefficient.
        a1 (float): stiffness damping coefficient.
        free_indices (torch.Tensor): 1D tensor of non-fixed indices.
        device (str, optional): Device to make tensors on. Defaults to "cuda".

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: _description_
    """
    m = full_lumped_mass(x, y, rho, device=device)
    k = full_stiffness(x, y, E, nu, device=device)

    # mask out fixed dims
    m = m[free_indices]
    m = m[:, free_indices]
    k = k[free_indices]
    k = k[:, free_indices]

    c = a0 * m + a1 * k
    return m, c, k


def rayleigh_damping_parameters(
    omega1=2 * torch.pi * 1.5, omega2=2 * torch.pi * 14, damp1=0.01, damp2=0.02
) -> "tuple[float, float]":
    """Get Rayleigh damping parameters.

    Args:
        omega1 (float, optional): Defaults to 2*torch.pi*1.5.
        omega2 (float, optional): Defaults to 2*torch.pi*14.
        damp1 (float, optional): Defaults to 0.01.
        damp2 (float, optional): Defaults to 0.02.

    Returns:
        tuple[float, float]: mass and stiffness damping coefficients
    """
    a0 = (2 * damp1 * omega1 * (omega2**2) - 2 * damp2 * omega2 * (omega1**2)) / (
        (omega2**2) - (omega1**2)
    )
    a1 = (2 * damp2 * omega2 - 2 * damp1 * omega1) / ((omega2**2) - (omega1**2))
    return a0, a1


def fixed_indices(nodes_y: int, nodes_x: int) -> torch.Tensor:
    """Obtain fixed indices for the 98-dof problem. Fixed nodes are those along the left, right, and bottom of the domain.

    Args:
        nodes_y (int): number of nodes in the y-direction.
        nodes_x (int): number of nodes in the x-direction.

    Returns:
        torch.Tensor: 1D tensor of fixed indices (not node numbers).
    """
    left_side_nn = [i * nodes_x for i in range(nodes_y)]
    right_side_nn = [i * nodes_x + nodes_x - 1 for i in range(nodes_y)]
    bottom_side_nn = [j for j in range(1, nodes_x - 1)]

    fixed_nodes = left_side_nn + right_side_nn + bottom_side_nn
    fixed_nodes = torch.as_tensor(fixed_nodes)

    fixed_indices = torch.concatenate((fixed_nodes * 2, fixed_nodes * 2 + 1))
    return fixed_indices


def full_lumped_mass(
    x: torch.Tensor,
    y: torch.Tensor,
    rho: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Full general mass matrix of the model. Elements are defined as follows
    ```
    (x[i + 1, j], y[i + 1, j]) -------------------- (x[i + 1, j + 1], y[i + 1, j + 1])
                |                                                   |
                |                                                   |
                |           E[i, j], nu[i, j], rho[i, j]            |
                |                                                   |
                |                                                   |
        (x[i, j], y[i, j]) ---------------------------  (x[i, j + 1], y[i, j + 1])
    ```
    Args:
        x (torch.Tensor): N x M grid of nodal x-coordinates. Must be monotonically increasing along axis 1.
        y (torch.Tensor): N x M grid of nodal y-coordinates. Must be monotonically increasing along axis 0.
        rho (torch.Tensor): (N - 1) x (M - 1) grid of elemental Young's densities.
        device (str, optional): Device to make tensors on. Defaults to "cuda".

    Returns:
        torch.Tensor: (2*N*M) x (2*N*M) full general lumped mass matrix of the FE model.
    """
    NUMNODES = torch.prod(torch.as_tensor(x.shape, device=device))
    NUMEL_X = x.shape[1] - 1
    NUMEL_Y = y.shape[0] - 1

    nodes = torch.arange(NUMNODES, device=device).reshape(x.shape)

    global_m = torch.zeros((2 * NUMNODES, 2 * NUMNODES), device=device)
    for i in range(NUMEL_Y):
        for j in range(NUMEL_X):
            x1, y1 = x[i, j], y[i, j]
            x3, y3 = x[1 + i, j + 1], y[1 + i, j + 1]
            _rho = rho[i, j]

            global_indices = torch.as_tensor(
                (
                    nodes[i, j] * 2,  # x-component of bottom-left node
                    nodes[i, j] * 2 + 1,  # y-component of bottom-left node
                    nodes[i + 1, j] * 2,  # x-component of top-left node
                    nodes[i + 1, j] * 2 + 1,  # y-component of top-left node
                    nodes[i + 1, j + 1] * 2,  # x-component of top-right node
                    nodes[i + 1, j + 1] * 2 + 1,  # y-component of top-right node
                    nodes[i, j + 1] * 2,  # x-component of bottom-right node
                    nodes[i, j + 1] * 2 + 1,  # y-component of bottom-right node
                ),
                device=device,
            ).type(torch.int)

            local_m = single_element_lumped_mass(x1, y1, x3, y3, _rho, device=device)
            for k in range(len(global_indices)):
                for l in range(len(global_indices)):
                    global_m[global_indices[k], global_indices[l]] = (
                        global_m[global_indices[k], global_indices[l]] + local_m[k, l]
                    )
    return global_m


def full_stiffness(
    x: torch.Tensor,
    y: torch.Tensor,
    E: torch.Tensor,
    nu: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Full general stiffness matrix of the FE model. Elements are defined as follows.
    ```
    (x[i + 1, j], y[i + 1, j]) -------------------- (x[i + 1, j + 1], y[i + 1, j + 1])
                |                                                   |
                |                                                   |
                |           E[i, j], nu[i, j], rho[i, j]            |
                |                                                   |
                |                                                   |
        (x[i, j], y[i, j]) ---------------------------  (x[i, j + 1], y[i, j + 1])
    ```
    Args:
        x (torch.Tensor): N x M grid of nodal x-coordinates. Must be monotonically increasing along axis 1.
        y (torch.Tensor): N x M grid of nodal y-coordinates. Must be monotonically increasing along axis 0.
        E (torch.Tensor): (N - 1) x (M - 1) grid of elemental Young's moduli.
        nu (torch.Tensor): (N - 1) x (M - 1) grid of elemental Poisson's ratios.
        device (str, optional): Device to make tensors on. Defaults to "cuda".

    Returns:
        torch.Tensor: (2*N*M) x (2*N*M) full general stiffness matrix of the FE model.
    """
    NUMNODES = torch.prod(torch.as_tensor(x.shape, device=device))
    NUMEL_X = x.shape[1] - 1
    NUMEL_Y = y.shape[0] - 1

    nodes = torch.arange(NUMNODES, device=device).reshape(x.shape)

    global_k = torch.zeros((2 * NUMNODES, 2 * NUMNODES), device=device)
    for i in range(NUMEL_Y):
        for j in range(NUMEL_X):
            x1, y1 = x[i, j], y[i, j]
            x3, y3 = x[1 + i, j + 1], y[1 + i, j + 1]
            _nu, _E = nu[i, j], E[i, j]

            global_indices = torch.as_tensor(
                (
                    nodes[i, j] * 2,  # x-component of bottom-left node
                    nodes[i, j] * 2 + 1,  # y-component of bottom-left node
                    nodes[i + 1, j] * 2,  # x-component of top-left node
                    nodes[i + 1, j] * 2 + 1,  # y-component of top-left node
                    nodes[i + 1, j + 1] * 2,  # x-component of top-right node
                    nodes[i + 1, j + 1] * 2 + 1,  # y-component of top-right node
                    nodes[i, j + 1] * 2,  # x-component of bottom-right node
                    nodes[i, j + 1] * 2 + 1,  # y-component of bottom-right node
                ),
                device=device,
            ).type(torch.int)

            local_k = single_element_stiffness(x1, y1, x3, y3, _nu, _E, device=device)
            for k in range(len(global_indices)):
                for l in range(len(global_indices)):
                    global_k[global_indices[k], global_indices[l]] = (
                        global_k[global_indices[k], global_indices[l]] + local_k[k, l]
                    )
    return global_k


@lru_cache()
def single_element_lumped_mass(
    x1: float, y1: float, x3: float, y3: float, rho: float, device="cuda"
) -> torch.Tensor:
    """Lumped mass matrix for a single, square, quadrilateral element, constructed as
    ```
    (x2, y2) ----- (x3, y3)
        |              |
        |  E, nu, rho  |
        |              |
    (x1, y1) ----- (x4, y4)
    ```
    Args:
        x1 (float): x-coordinate of the bottom-left corner of the element
        y1 (float): y-coordinate of the bottom-left corner of the element
        x3 (float): x-coordinate of the top-right corner of the element
        y3 (float): y-coordinate of the top-right corner of the element
        rho (float): Density of the element
        device (str, optional): Device to make tensors on. Defaults to "cuda".

    Returns:
        torch.Tensor: 8x8 lumped mass matrix of the element.
    """
    mass_per_node = rho * (x3 - x1) * (y3 - y1) / 4
    mat = torch.eye(8, 8, device=device) * mass_per_node
    return mat


@lru_cache
def single_element_stiffness(
    x1: float, y1: float, x3: float, y3: float, nu: float, E: float, device="cuda"
) -> torch.Tensor:
    """Stiffness matrix of a single, square, quadrilateral element, constructed as
    ```
    (x2, y2) ----- (x3, y3)
        |              |
        |  E, nu, rho  |
        |              |
    (x1, y1) ----- (x4, y4)
    ```
    Args:
        x1 (float): x-coordinate of the bottom-left corner of the element
        y1 (float): y-coordinate of the bottom-left corner of the element
        x3 (float): x-coordinate of the top-right corner of the element
        y3 (float): y-coordinate of the top-right corner of the element
        nu (float): Poisson's ratio of the element
        E (float): Young's Modulus of the element
        device (str, optional): Device to make tensors on. Defaults to "cuda".

    Returns:
        torch.Tensor: 8x8 stiffness matrix for the element.
    """
    mat = (
        [
            [
                E
                * (
                    2 * nu * x1**2
                    - 4 * nu * x1 * x3
                    + 2 * nu * x3**2
                    + 2 * nu * y1**2
                    - 4 * nu * y1 * y3
                    + 2 * nu * y3**2
                    - x1**2
                    + 2 * x1 * x3
                    - x3**2
                    - 2 * y1**2
                    + 4 * y1 * y3
                    - 2 * y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                -E / (16 * nu**2 + 8 * nu - 8),
                E
                * (
                    -2 * nu * x1**2
                    + 4 * nu * x1 * x3
                    - 2 * nu * x3**2
                    + nu * y1**2
                    - 2 * nu * y1 * y3
                    + nu * y3**2
                    + x1**2
                    - 2 * x1 * x3
                    + x3**2
                    - y1**2
                    + 2 * y1 * y3
                    - y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (4 * nu - 1) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    -2 * nu * x1**2
                    + 4 * nu * x1 * x3
                    - 2 * nu * x3**2
                    - 2 * nu * y1**2
                    + 4 * nu * y1 * y3
                    - 2 * nu * y3**2
                    + x1**2
                    - 2 * x1 * x3
                    + x3**2
                    + 2 * y1**2
                    - 4 * y1 * y3
                    + 2 * y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    2 * nu * x1**2
                    - 4 * nu * x1 * x3
                    + 2 * nu * x3**2
                    - 4 * nu * y1**2
                    + 8 * nu * y1 * y3
                    - 4 * nu * y3**2
                    - x1**2
                    + 2 * x1 * x3
                    - x3**2
                    + 4 * y1**2
                    - 8 * y1 * y3
                    + 4 * y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (1 - 4 * nu) / (8 * (2 * nu**2 + nu - 1)),
            ],
            [
                -E / (16 * nu**2 + 8 * nu - 8),
                E
                * (
                    2 * nu * x1**2
                    - 4 * nu * x1 * x3
                    + 2 * nu * x3**2
                    + 2 * nu * y1**2
                    - 4 * nu * y1 * y3
                    + 2 * nu * y3**2
                    - 2 * x1**2
                    + 4 * x1 * x3
                    - 2 * x3**2
                    - y1**2
                    + 2 * y1 * y3
                    - y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (1 - 4 * nu) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    -4 * nu * x1**2
                    + 8 * nu * x1 * x3
                    - 4 * nu * x3**2
                    + 2 * nu * y1**2
                    - 4 * nu * y1 * y3
                    + 2 * nu * y3**2
                    + 4 * x1**2
                    - 8 * x1 * x3
                    + 4 * x3**2
                    - y1**2
                    + 2 * y1 * y3
                    - y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    -2 * nu * x1**2
                    + 4 * nu * x1 * x3
                    - 2 * nu * x3**2
                    - 2 * nu * y1**2
                    + 4 * nu * y1 * y3
                    - 2 * nu * y3**2
                    + 2 * x1**2
                    - 4 * x1 * x3
                    + 2 * x3**2
                    + y1**2
                    - 2 * y1 * y3
                    + y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (4 * nu - 1) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    nu * x1**2
                    - 2 * nu * x1 * x3
                    + nu * x3**2
                    - 2 * nu * y1**2
                    + 4 * nu * y1 * y3
                    - 2 * nu * y3**2
                    - x1**2
                    + 2 * x1 * x3
                    - x3**2
                    + y1**2
                    - 2 * y1 * y3
                    + y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
            ],
            [
                E
                * (
                    -2 * nu * x1**2
                    + 4 * nu * x1 * x3
                    - 2 * nu * x3**2
                    + nu * y1**2
                    - 2 * nu * y1 * y3
                    + nu * y3**2
                    + x1**2
                    - 2 * x1 * x3
                    + x3**2
                    - y1**2
                    + 2 * y1 * y3
                    - y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (1 - 4 * nu) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    2 * nu * x1**2
                    - 4 * nu * x1 * x3
                    + 2 * nu * x3**2
                    + 2 * nu * y1**2
                    - 4 * nu * y1 * y3
                    + 2 * nu * y3**2
                    - x1**2
                    + 2 * x1 * x3
                    - x3**2
                    - 2 * y1**2
                    + 4 * y1 * y3
                    - 2 * y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    2 * nu * x1**2
                    - 4 * nu * x1 * x3
                    + 2 * nu * x3**2
                    - 4 * nu * y1**2
                    + 8 * nu * y1 * y3
                    - 4 * nu * y3**2
                    - x1**2
                    + 2 * x1 * x3
                    - x3**2
                    + 4 * y1**2
                    - 8 * y1 * y3
                    + 4 * y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (4 * nu - 1) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    -2 * nu * x1**2
                    + 4 * nu * x1 * x3
                    - 2 * nu * x3**2
                    - 2 * nu * y1**2
                    + 4 * nu * y1 * y3
                    - 2 * nu * y3**2
                    + x1**2
                    - 2 * x1 * x3
                    + x3**2
                    + 2 * y1**2
                    - 4 * y1 * y3
                    + 2 * y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                -E / (16 * nu**2 + 8 * nu - 8),
            ],
            [
                E * (4 * nu - 1) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    -4 * nu * x1**2
                    + 8 * nu * x1 * x3
                    - 4 * nu * x3**2
                    + 2 * nu * y1**2
                    - 4 * nu * y1 * y3
                    + 2 * nu * y3**2
                    + 4 * x1**2
                    - 8 * x1 * x3
                    + 4 * x3**2
                    - y1**2
                    + 2 * y1 * y3
                    - y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    2 * nu * x1**2
                    - 4 * nu * x1 * x3
                    + 2 * nu * x3**2
                    + 2 * nu * y1**2
                    - 4 * nu * y1 * y3
                    + 2 * nu * y3**2
                    - 2 * x1**2
                    + 4 * x1 * x3
                    - 2 * x3**2
                    - y1**2
                    + 2 * y1 * y3
                    - y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (1 - 4 * nu) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    nu * x1**2
                    - 2 * nu * x1 * x3
                    + nu * x3**2
                    - 2 * nu * y1**2
                    + 4 * nu * y1 * y3
                    - 2 * nu * y3**2
                    - x1**2
                    + 2 * x1 * x3
                    - x3**2
                    + y1**2
                    - 2 * y1 * y3
                    + y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                -E / (16 * nu**2 + 8 * nu - 8),
                E
                * (
                    -2 * nu * x1**2
                    + 4 * nu * x1 * x3
                    - 2 * nu * x3**2
                    - 2 * nu * y1**2
                    + 4 * nu * y1 * y3
                    - 2 * nu * y3**2
                    + 2 * x1**2
                    - 4 * x1 * x3
                    + 2 * x3**2
                    + y1**2
                    - 2 * y1 * y3
                    + y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
            ],
            [
                E
                * (
                    -2 * nu * x1**2
                    + 4 * nu * x1 * x3
                    - 2 * nu * x3**2
                    - 2 * nu * y1**2
                    + 4 * nu * y1 * y3
                    - 2 * nu * y3**2
                    + x1**2
                    - 2 * x1 * x3
                    + x3**2
                    + 2 * y1**2
                    - 4 * y1 * y3
                    + 2 * y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    2 * nu * x1**2
                    - 4 * nu * x1 * x3
                    + 2 * nu * x3**2
                    - 4 * nu * y1**2
                    + 8 * nu * y1 * y3
                    - 4 * nu * y3**2
                    - x1**2
                    + 2 * x1 * x3
                    - x3**2
                    + 4 * y1**2
                    - 8 * y1 * y3
                    + 4 * y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (1 - 4 * nu) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    2 * nu * x1**2
                    - 4 * nu * x1 * x3
                    + 2 * nu * x3**2
                    + 2 * nu * y1**2
                    - 4 * nu * y1 * y3
                    + 2 * nu * y3**2
                    - x1**2
                    + 2 * x1 * x3
                    - x3**2
                    - 2 * y1**2
                    + 4 * y1 * y3
                    - 2 * y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                -E / (16 * nu**2 + 8 * nu - 8),
                E
                * (
                    -2 * nu * x1**2
                    + 4 * nu * x1 * x3
                    - 2 * nu * x3**2
                    + nu * y1**2
                    - 2 * nu * y1 * y3
                    + nu * y3**2
                    + x1**2
                    - 2 * x1 * x3
                    + x3**2
                    - y1**2
                    + 2 * y1 * y3
                    - y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (4 * nu - 1) / (8 * (2 * nu**2 + nu - 1)),
            ],
            [
                E / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    -2 * nu * x1**2
                    + 4 * nu * x1 * x3
                    - 2 * nu * x3**2
                    - 2 * nu * y1**2
                    + 4 * nu * y1 * y3
                    - 2 * nu * y3**2
                    + 2 * x1**2
                    - 4 * x1 * x3
                    + 2 * x3**2
                    + y1**2
                    - 2 * y1 * y3
                    + y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (4 * nu - 1) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    nu * x1**2
                    - 2 * nu * x1 * x3
                    + nu * x3**2
                    - 2 * nu * y1**2
                    + 4 * nu * y1 * y3
                    - 2 * nu * y3**2
                    - x1**2
                    + 2 * x1 * x3
                    - x3**2
                    + y1**2
                    - 2 * y1 * y3
                    + y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                -E / (16 * nu**2 + 8 * nu - 8),
                E
                * (
                    2 * nu * x1**2
                    - 4 * nu * x1 * x3
                    + 2 * nu * x3**2
                    + 2 * nu * y1**2
                    - 4 * nu * y1 * y3
                    + 2 * nu * y3**2
                    - 2 * x1**2
                    + 4 * x1 * x3
                    - 2 * x3**2
                    - y1**2
                    + 2 * y1 * y3
                    - y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (1 - 4 * nu) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    -4 * nu * x1**2
                    + 8 * nu * x1 * x3
                    - 4 * nu * x3**2
                    + 2 * nu * y1**2
                    - 4 * nu * y1 * y3
                    + 2 * nu * y3**2
                    + 4 * x1**2
                    - 8 * x1 * x3
                    + 4 * x3**2
                    - y1**2
                    + 2 * y1 * y3
                    - y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
            ],
            [
                E
                * (
                    2 * nu * x1**2
                    - 4 * nu * x1 * x3
                    + 2 * nu * x3**2
                    - 4 * nu * y1**2
                    + 8 * nu * y1 * y3
                    - 4 * nu * y3**2
                    - x1**2
                    + 2 * x1 * x3
                    - x3**2
                    + 4 * y1**2
                    - 8 * y1 * y3
                    + 4 * y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (4 * nu - 1) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    -2 * nu * x1**2
                    + 4 * nu * x1 * x3
                    - 2 * nu * x3**2
                    - 2 * nu * y1**2
                    + 4 * nu * y1 * y3
                    - 2 * nu * y3**2
                    + x1**2
                    - 2 * x1 * x3
                    + x3**2
                    + 2 * y1**2
                    - 4 * y1 * y3
                    + 2 * y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                -E / (16 * nu**2 + 8 * nu - 8),
                E
                * (
                    -2 * nu * x1**2
                    + 4 * nu * x1 * x3
                    - 2 * nu * x3**2
                    + nu * y1**2
                    - 2 * nu * y1 * y3
                    + nu * y3**2
                    + x1**2
                    - 2 * x1 * x3
                    + x3**2
                    - y1**2
                    + 2 * y1 * y3
                    - y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (1 - 4 * nu) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    2 * nu * x1**2
                    - 4 * nu * x1 * x3
                    + 2 * nu * x3**2
                    + 2 * nu * y1**2
                    - 4 * nu * y1 * y3
                    + 2 * nu * y3**2
                    - x1**2
                    + 2 * x1 * x3
                    - x3**2
                    - 2 * y1**2
                    + 4 * y1 * y3
                    - 2 * y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E / (8 * (2 * nu**2 + nu - 1)),
            ],
            [
                E * (1 - 4 * nu) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    nu * x1**2
                    - 2 * nu * x1 * x3
                    + nu * x3**2
                    - 2 * nu * y1**2
                    + 4 * nu * y1 * y3
                    - 2 * nu * y3**2
                    - x1**2
                    + 2 * x1 * x3
                    - x3**2
                    + y1**2
                    - 2 * y1 * y3
                    + y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                -E / (16 * nu**2 + 8 * nu - 8),
                E
                * (
                    -2 * nu * x1**2
                    + 4 * nu * x1 * x3
                    - 2 * nu * x3**2
                    - 2 * nu * y1**2
                    + 4 * nu * y1 * y3
                    - 2 * nu * y3**2
                    + 2 * x1**2
                    - 4 * x1 * x3
                    + 2 * x3**2
                    + y1**2
                    - 2 * y1 * y3
                    + y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E * (4 * nu - 1) / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    -4 * nu * x1**2
                    + 8 * nu * x1 * x3
                    - 4 * nu * x3**2
                    + 2 * nu * y1**2
                    - 4 * nu * y1 * y3
                    + 2 * nu * y3**2
                    + 4 * x1**2
                    - 8 * x1 * x3
                    + 4 * x3**2
                    - y1**2
                    + 2 * y1 * y3
                    - y3**2
                )
                / (
                    12
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
                E / (8 * (2 * nu**2 + nu - 1)),
                E
                * (
                    2 * nu * x1**2
                    - 4 * nu * x1 * x3
                    + 2 * nu * x3**2
                    + 2 * nu * y1**2
                    - 4 * nu * y1 * y3
                    + 2 * nu * y3**2
                    - 2 * x1**2
                    + 4 * x1 * x3
                    - 2 * x3**2
                    - y1**2
                    + 2 * y1 * y3
                    - y3**2
                )
                / (
                    6
                    * (
                        2 * nu**2 * x1 * y1
                        - 2 * nu**2 * x1 * y3
                        - 2 * nu**2 * x3 * y1
                        + 2 * nu**2 * x3 * y3
                        + nu * x1 * y1
                        - nu * x1 * y3
                        - nu * x3 * y1
                        + nu * x3 * y3
                        - x1 * y1
                        + x1 * y3
                        + x3 * y1
                        - x3 * y3
                    )
                ),
            ],
        ],
    )

    k = torch.zeros(8, 8, device=device)
    for i in range(8):
        for j in range(8):
            k[i, j] = mat[0][i][j]

    return k
