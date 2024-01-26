import os
import csv
import torch
import numpy as np
import time as timer
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from random import random
from typing import Callable
from matplotlib import pyplot as plt

from utils import make_folder
from forcing_functions import get_function
from finite_element_code import set_up_4_dof, get_mck, integrate_rk4


def system_ode_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    f: torch.Tensor,
    m: torch.Tensor,
    c: torch.Tensor,
    k: torch.Tensor,
    u_max: float,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.mse_loss,
):
    """Computes loss due to the residual

    M y_xx + C y_x + K y - f / u_max == 0
    """
    y_x, y_xx = y * 0, y * 0
    for dim in range(y_x.shape[1]):
        y_x[:, dim] = gradient(y[:, dim], x).squeeze()
        y_xx[:, dim] = gradient(y_x[:, dim], x).squeeze()

    residual = torch.mm(m, y_xx.T) + torch.mm(c, y_x.T) + torch.mm(k, y.T) - (f / u_max)
    return loss_fn(residual, residual * 0)


def latin_hypercube_1D(t0, t1, n, device="cuda"):
    endpts = torch.linspace(t0, t1, n + 1, device=device)
    sample_pts = torch.as_tensor(
        [random() * (endpts[i + 1] - endpts[i]) + endpts[i] for i in range(n)],
        device=device,
    )
    return sample_pts


def get_4_dof_model(
    hidden_layers: int = 3,
    hidden_features: int = 32,
    first_omega_0: float = 30.0,
    hidden_omega_0: float = 30.0,
    device: str = "cuda",
):
    out_features = 4
    model = Siren(
        in_features=1,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        out_features=out_features,
        outermost_linear=True,
        first_omega_0=first_omega_0,
        hidden_omega_0=hidden_omega_0,
    ).to(device)
    return model


class SineLayer(nn.Module):
    """
    Adapted from
    https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
    """

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class Siren(nn.Module):
    """
    Adapted from
    https://github.com/vsitzmann/siren/blob/master/explore_siren.ipynb
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)

        return output, coords


def run_experiment(
    functions: "list[str]",
    given_data: "list[list[int]]",
    noise_percent: "list[float]",
    initialization_error: "list[float]",
    run_name: str,
    device="cuda",
    alpha=1e-14,
    learning_rate=1e-4,
    max_iterations=15_000,
    iterations_til_log=1_000,
    iterations_til_update=10,
    patience=1_000,
):
    # Get model parameters
    x, y, E, nu, rho, a0, a1, fdim, free_indices, ndim, _ = set_up_4_dof()

    # Construct mass, damping, and stiffness matrices
    m, c, k = get_mck(x, y, E, nu, rho, a0, a1, free_indices)

    t0 = 0
    t1 = 2.9
    ntc = 290
    ntp = int((t1 - t0) * 500)
    tc = torch.linspace(t0, t1, ntc, device="cuda")
    tp = latin_hypercube_1D(t0, t1, ntp, device="cuda")

    for forcing_function in functions:
        # Get forcing function
        ffun = get_function(
            name=forcing_function,
            t0=t0,
            t1=t1,
            amplitude=-1e3,
            total_dimensions=ndim,
            force_dimension=fdim,
        )

        # Integrate to get time-histories
        displacements, _ = integrate_rk4(m, c, k, ffun, tc)

        for node_list in given_data:
            for noise in noise_percent:
                for init_error in initialization_error:
                    print("FUNCTION:   ", forcing_function)
                    print("GIVEN NODES:", node_list)
                    print("NOISE:      ", noise, "%")
                    print("INIT ERROR  ", init_error, "%")
                    print(
                        "torch.cuda.memory_allocated() ->",
                        torch.cuda.memory_allocated(),
                    )

                    # Make folder to store data
                    node_folder_name = make_folder(
                        os.path.join(
                            "outputs",
                            run_name,
                            forcing_function,
                            f"{len(node_list)}_nodes_given",
                            f"{noise}%_noise",
                            f"{init_error}%_init_error",
                        )
                    )

                    # Introduce noise to displacements
                    noise_vector = displacements.abs().max() * noise / 100
                    noisy_displacements = displacements + noise_vector

                    # Get maximum displacement - used for normalizing outputs. Only consider amplitude of given nodes.
                    if len(node_list) > 0:
                        max_displacement = noisy_displacements[node_list].abs().max()
                    else:
                        max_displacement = torch.as_tensor(
                            2e-5, device=device
                        )  # estimate

                    # Create PINN
                    model = get_4_dof_model()

                    # Introduce initialization error and create trainable E
                    percent = 1 - init_error / 100
                    E_train = E * percent
                    E_train = E_train / 1e8
                    E_train = E_train.reshape(-1, 1)
                    E_train.requires_grad = True

                    # Instantiate optimizer
                    optimizer = torch.optim.Adam(
                        list(model.parameters()) + [E_train],
                        lr=learning_rate,
                    )

                    # Perform training
                    csv_filename = os.path.join(node_folder_name, "loss_history.csv")
                    model_save_file = os.path.join(node_folder_name, "model.pt")
                    plot_save_file = os.path.join(node_folder_name, "result.png")

                    zero = torch.zeros((1, 1), device=device)

                    with open(csv_filename, "w") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(
                            [
                                "Epoch",
                                "Collocation Loss",
                                "Physics Loss",
                                "Total Loss",
                                "Alpha",
                                "E",
                                "Percent Error (E)",
                            ]
                        )

                    # Get start time
                    start = timer.time()

                    convergence_counter = (
                        0  # number of iterations for which percent error in E <= 2%
                    )

                    pbar = tqdm(range(max_iterations))
                    for step in pbar:
                        optimizer.zero_grad()

                        # Shuffle training data - helps a lot with convergence
                        shuffle_idx_c = torch.randperm(ntc)
                        shuffle_idx_p = torch.randperm(ntp)

                        # Enforce Collocation
                        model_output, _ = model(tc[shuffle_idx_c].reshape(-1, 1))
                        target = noisy_displacements.T[shuffle_idx_c] / max_displacement

                        zpred, _ = model(zero)

                        collocation_loss = F.mse_loss(
                            model_output[:, node_list], target[:, node_list]
                        ) + F.mse_loss(zpred, 0 * zpred)

                        # Enforce Physics
                        m, c, k = get_mck(
                            x, y, E_train * 1e8, nu, rho, a0, a1, free_indices
                        )

                        model_output_phys, coords = model(
                            tp[shuffle_idx_p].reshape(-1, 1)
                        )
                        physical_loss = system_ode_loss(
                            coords,
                            model_output_phys,
                            ffun(tp[shuffle_idx_p]),
                            m,
                            c,
                            k,
                            max_displacement,
                        )

                        # Total loss
                        loss = collocation_loss + alpha * physical_loss

                        # Print out an update, save model, + log to file
                        if step % iterations_til_update == 0:
                            pbar.set_description_str(
                                "Iteration %d: L_c = %.5g -- L_p = %.5g -- E = [%.5g]"
                                % (
                                    step,
                                    collocation_loss,
                                    physical_loss,
                                    E_train[0, 0] * 1e8,
                                )
                            )

                        # Percent error of E_train vs. true E
                        PE = abs(E - E_train.item() * 1e8) / E * 100

                        if step % iterations_til_log == 0:
                            with open(csv_filename, "a") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow(
                                    [
                                        step,
                                        collocation_loss.item(),
                                        physical_loss.item(),
                                        loss.item(),
                                        alpha,
                                        E_train[0, 0] * 1e8,
                                        PE,
                                    ]
                                )
                            torch.save(model.state_dict(), model_save_file)

                        # Backpropagate
                        loss.backward()

                        # Update model parameters
                        optimizer.step()

                        # Stop early if E_train is within 2% of E for `patience` iterations
                        if PE <= 2:
                            convergence_counter += 1
                        else:
                            convergence_counter = 0

                        if convergence_counter == patience:
                            pbar.close()
                            break

                    # Training finished.
                    end = timer.time()
                    print("Training completed in %.5g seconds." % (end - start))

                    # Make plot of the results
                    u_pred, _ = model(tc.reshape(-1, 1))

                    fig, ax = plt.subplots(
                        ndim // 2, 2, sharex=True, sharey=True, figsize=(10, 5)
                    )

                    for i in range(ndim // 2):
                        # Plot solution
                        ax[i, 0].plot(
                            tc.cpu(),
                            displacements[2 * i].cpu(),
                            color="gray",
                            alpha=0.4,
                            label="Solution",
                        )
                        ax[i, 1].plot(
                            tc.cpu(),
                            displacements[2 * i + 1].cpu(),
                            color="gray",
                            alpha=0.4,
                            label="Solution",
                        )

                        # Plot model prediction
                        ax[i, 0].plot(
                            tc.cpu(),
                            u_pred.detach()[:, 2 * i].cpu() * max_displacement.cpu(),
                            color="green",
                            label="Prediction",
                        )
                        ax[i, 1].plot(
                            tc.cpu(),
                            u_pred.detach()[:, 2 * i + 1].cpu()
                            * max_displacement.cpu(),
                            color="green",
                            label="Prediction",
                        )

                        # Format plots
                        ax[i, 0].set_ylim(
                            displacements.cpu().min(), displacements.cpu().max()
                        )
                        ax[i, 1].set_ylim(
                            displacements.cpu().min(), displacements.cpu().max()
                        )
                        ax[i, 0].set_ylabel("Node %d" % (2 + i))

                        if i == 0:
                            ax[i, 0].set_title("X-Displacement (m)")
                            ax[i, 1].set_title("Y-Displacement (m)")

                        if 2 * i in node_list:
                            # Plot given data
                            ax[i, 0].plot(
                                tc.cpu(),
                                noisy_displacements[2 * i].cpu(),
                                color="orange",
                                label="Data",
                                marker=".",
                                markersize=1,
                                linestyle="None",
                            )

                        if (2 * i + 1) in node_list:
                            ax[i, 1].plot(
                                tc.cpu(),
                                noisy_displacements[2 * i + 1].cpu(),
                                color="orange",
                                label="Data",
                                marker=".",
                                markersize=1,
                                linestyle="None",
                            )

                        # Legend
                        if i == 0:
                            ax[i, 1].legend(
                                loc="upper center", ncols=1, bbox_to_anchor=(1.2, 1)
                            )

                    plt.savefig(plot_save_file, bbox_inches="tight")
                    plt.close()

                    # Delete all unneeded variables
                    del (
                        model,
                        fig,
                        ax,
                        u_pred,
                        start,
                        end,
                        optimizer,
                        loss,
                        physical_loss,
                        collocation_loss,
                        target,
                        model_output,
                        shuffle_idx_c,
                        shuffle_idx_p,
                        writer,
                        E_train,
                    )

                    torch.cuda.empty_cache()


class ExperimentSet:
    def __init__(
        self,
        functions: "list[str]",
        given_data: "list[list[int]]",
        initial_E_percent_error: "list[float]",
        noise: "list[float]",
        run_name: str,
        device="cuda",
    ):
        self.functions = functions
        self.given_data = given_data
        self.initial_E_percent_error = initial_E_percent_error
        self.noise = noise
        self.run_name = run_name
        self.device = device

        (
            self.x,
            self.y,
            self.E,
            self.nu,
            self.rho,
            self.a0,
            self.a1,
            self.fdim,
            self.free_indices,
            self.ndim,
            _,
        ) = set_up_4_dof()

        self.m, self.c, self.k = get_mck(
            self.x,
            self.y,
            self.E,
            self.nu,
            self.rho,
            self.a0,
            self.a1,
            self.free_indices,
        )

        # Set up experiment hyper-parameters
        self.t0 = 0
        self.t1 = 2.9
        self.ntc = 290
        self.ntp = int((self.t1 - self.t0) * 500)
        self.tc = torch.linspace(self.t0, self.t1, self.ntc, device=self.device)
        self.tp = latin_hypercube_1D(self.t0, self.t1, self.ntp, device=self.device)

    def run(
        self,
        alpha,
        learning_rate,
        max_iterations,
        iterations_til_summary,
    ):
        ## LOOP 0: FORCING FUNCTIONS
        for forcing_function in self.functions:
            ## LOOP 1: NOISE LEVEL
            for noise_level in self.noise:
                ## LOOP 2: INITIAL ERROR
                for initial_error in self.initial_E_percent_error:
                    ## LOOP 3: GIVEN DATA
                    for node_list in self.given_data:
                        # Get the current forcing function
                        ffun = get_function(
                            name=forcing_function,
                            t0=self.t0,
                            t1=self.t1,
                            amplitude=-1e3,
                            total_dimensions=self.ndim,
                            force_dimension=self.fdim,
                        )

                        # Integrate to get time-histories
                        displacements, _ = integrate_rk4(
                            self.m, self.c, self.k, ffun, self.tc
                        )

                        # Add noise
                        # displacements += (
                        #     torch.rand(
                        #         *displacements.shape, device=self.device
                        #     )  # Random numbers in [-1, 1]
                        #     * (
                        #         torch.abs(displacements)
                        #     ).max()  # Multiply by max(abs(u)). Now noise is random numbers in [-max(abs(u)), max(abs(u))]
                        #     * (
                        #         noise_level / 100
                        #     )  # Multiply by desired percent noise. Now noise is random numbers in [-noise_level*max(abs(u)), noise_level*max(abs(u))]
                        # )

                        # Name of the folder for the current training cycle
                        folder_name = os.path.join(
                            "outputs",
                            self.run_name,
                            forcing_function,
                            f"{str(noise_level).replace('.', '-')}%_noise",
                            f"{str(initial_error).replace('.', '-')}%_init_err",
                            f"{len(node_list)}_nodes_given",
                        )
                        folder_name = make_folder(folder_name)

                        # Name of files to store training run data in
                        csv_filename = os.path.join(folder_name, "loss_history.csv")
                        model_save_file = os.path.join(folder_name, "model.pt")
                        plot_save_file = os.path.join(folder_name, "result.png")

                        print("FUNCTION:", forcing_function)
                        print("NOISE:", noise_level, "%")
                        print("INIT ERROR:", initial_error, "%")
                        print("GIVEN DATA:", node_list)
                        print(
                            "torch.cuda.memory_allocated() ->",
                            torch.cuda.memory_allocated(),
                        )

                        start_time = timer.time()

                        # Get maximum displacement - used for normalizing outputs. Only consider amplitude of given nodes.
                        if len(node_list) > 0:
                            max_displacement = displacements[node_list].abs().max()
                        else:
                            max_displacement = torch.as_tensor(
                                2e-5, device=self.device
                            )  # estimate

                        # Create SIREN
                        model = get_4_dof_model()

                        # Create elastic modulus trainable parameter
                        E_train = (
                            self.E
                            * (1 - initial_error / 100)  # incorporate percent error
                            * (
                                1e-8
                            )  # normalize E so that the parameter updated during training is in [-1, 1] (makes computation more efficient)
                        )
                        E_train = E_train.reshape(-1, 1)
                        E_train.requires_grad = True

                        # Instantiate optimizer
                        optimizer = torch.optim.Adam(
                            list(model.parameters()) + [E_train],
                            lr=learning_rate,
                        )

                        # Zero tensor, defined here for convenience
                        zero = torch.zeros((1, 1), device=self.device)

                        # Write title row
                        with open(csv_filename, "w") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(
                                [
                                    "Epoch",
                                    "Collocation Loss",
                                    "Physics Loss",
                                    "Total Loss",
                                    "Alpha",
                                    "E",
                                    "Percent Error",
                                ]
                            )

                        # Training loop
                        for step in range(max_iterations):
                            # Clear the gradient
                            optimizer.zero_grad()

                            # Shuffle training data - helps a lot with convergence
                            shuffle_idx_c = torch.randperm(self.ntc)
                            shuffle_idx_p = torch.randperm(self.ntp)

                            # Enforce Collocation
                            model_output, _ = model(
                                self.tc[shuffle_idx_c].reshape(
                                    -1, 1
                                )  # predict u at shuffled times
                            )

                            # Target value of displacement = actual displacements
                            # given in the same order as predicted displacements
                            # also normalized so that |u| < 1 (generally)
                            target = displacements.T[shuffle_idx_c] / max_displacement

                            # Get prediction at t=0
                            zpred, _ = model(zero)

                            collocation_loss = F.mse_loss(
                                # compute loss ONLY AT THE GIVEN NODES
                                model_output[:, node_list],
                                target[:, node_list],
                            ) + F.mse_loss(
                                zpred, 0 * zpred
                            )  # also enforce that u(0) = 0

                            # Get m, c, and k assuming E = E_train
                            m, c, k = get_mck(
                                self.x,
                                self.y,
                                E_train * 1e8,
                                self.nu,
                                self.rho,
                                self.a0,
                                self.a1,
                                self.free_indices,
                            )

                            # Get the model prediction at the physics points, shuffled
                            model_output_phys, coords = model(
                                self.tp[shuffle_idx_p].reshape(-1, 1)
                            )

                            # Loss from the ODE
                            physical_loss = system_ode_loss(
                                coords,
                                model_output_phys,
                                ffun(self.tp[shuffle_idx_p]),
                                m,
                                c,
                                k,
                                max_displacement,
                            )

                            # Total loss
                            loss = collocation_loss + alpha * physical_loss

                            # Print out an update, save model, + log to file
                            if step % iterations_til_summary == 0:
                                print(
                                    "Iteration %d: L_c = %.5g -- L_p = %.5g -- E = [%.5g]"
                                    % (
                                        step,
                                        collocation_loss,
                                        physical_loss,
                                        E_train[0, 0] * 1e8,
                                    )
                                )

                                PE = (self.E - E_train) / self.E * 100
                                with open(csv_filename, "a") as csvfile:
                                    writer = csv.writer(csvfile)
                                    writer.writerow(
                                        [
                                            step,
                                            collocation_loss.item(),
                                            physical_loss.item(),
                                            loss.item(),
                                            alpha,
                                            E_train[0, 0] * 1e8,
                                            PE,
                                        ]
                                    )
                                    torch.save(model.state_dict(), model_save_file)

                                # Backpropagate
                                loss.backward()

                                # Update model parameters
                                optimizer.step()

                        # Training finished
                        end_time = timer.time()
                        print(
                            "Training completed in %.5g seconds."
                            % (end_time - start_time)
                        )

                        # Make plot of the results
                        u_pred, _ = model(self.tc.reshape(-1, 1))

                        _, ax = plt.subplots(
                            self.ndim // 2, 2, sharex=True, sharey=True, figsize=(10, 5)
                        )

                        for i in range(self.ndim // 2):
                            # Plot solution
                            ax[i, 0].plot(
                                self.tc.cpu(),
                                displacements[2 * i].cpu(),
                                color="gray",
                                alpha=0.4,
                                label="Solution",
                            )
                            ax[i, 1].plot(
                                self.tc.cpu(),
                                displacements[2 * i + 1].cpu(),
                                color="gray",
                                alpha=0.4,
                                label="Solution",
                            )

                            # Plot model prediction
                            ax[i, 0].plot(
                                self.tc.cpu(),
                                u_pred.detach()[:, 2 * i].cpu()
                                * max_displacement.cpu(),
                                color="green",
                                label="Prediction",
                            )
                            ax[i, 1].plot(
                                self.tc.cpu(),
                                u_pred.detach()[:, 2 * i + 1].cpu()
                                * max_displacement.cpu(),
                                color="green",
                                label="Prediction",
                            )

                            # Format plots
                            ax[i, 0].set_ylim(
                                displacements.cpu().min(), displacements.cpu().max()
                            )
                            ax[i, 1].set_ylim(
                                displacements.cpu().min(), displacements.cpu().max()
                            )
                            ax[i, 0].set_ylabel("Node %d" % (2 + i))

                            if i == 0:
                                ax[i, 0].set_title("X-Displacement (m)")
                                ax[i, 1].set_title("Y-Displacement (m)")

                            if 2 * i in node_list:
                                # Plot given data
                                ax[i, 0].plot(
                                    self.tc.cpu(),
                                    displacements[2 * i].cpu(),
                                    color="orange",
                                    label="Data",
                                    marker=".",
                                    markersize=1,
                                    linestyle="None",
                                )

                            if (2 * i + 1) in node_list:
                                ax[i, 1].plot(
                                    self.tc.cpu(),
                                    displacements[2 * i + 1].cpu(),
                                    color="orange",
                                    label="Data",
                                    marker=".",
                                    markersize=1,
                                    linestyle="None",
                                )

                            # Legend
                            if i == 0:
                                ax[i, 1].legend(
                                    loc="upper center", ncols=1, bbox_to_anchor=(1.2, 1)
                                )

                        plt.savefig(plot_save_file, bbox_inches="tight")
                        plt.close()

                        # Delete all unneeded variables
                        del (
                            ffun,
                            displacements,
                            folder_name,
                            csv_filename,
                            model_save_file,
                            plot_save_file,
                            start_time,
                            max_displacement,
                            model,
                            E_train,
                            optimizer,
                            zero,
                            writer,
                            shuffle_idx_c,
                            shuffle_idx_p,
                            model_output,
                            target,
                            zpred,
                            collocation_loss,
                            m,
                            c,
                            k,
                            model_output_phys,
                            coords,
                            physical_loss,
                            loss,
                            end_time,
                        )

                        torch.cuda.empty_cache()
