import torch
import numpy as np
import h5py
import os
from kmade.core.kmades import MGKMADE
from kmade.core.loss import loss_function
from kmade.core.saving import save_model, save_expr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
torch.set_default_dtype(torch.float64)

if __name__ == "__main__":
    # load the posterior samples
    with h5py.File("kmade/data/GW150914_posterior.h5", "r") as f:
        posterior0914 = (
            f["C01:IMRPhenomXPHM"]["posterior_samples"][
                "chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2"
            ]
            .view(np.float64)
            .reshape(-1, 6)
        )

    # preprocess the data
    # remove the cutoffs
    data = np.empty(posterior0914.shape)
    data[:, [0]] = np.log(posterior0914[:, [0]])
    data[:, [1, 2, 3]] = np.log(
        posterior0914[:, [1, 2, 3]] / (1 - posterior0914[:, [1, 2, 3]])
    )
    data[:, [4, 5]] = np.log(
        posterior0914[:, [4, 5]] / (np.pi - posterior0914[:, [4, 5]])
    )
    mean = np.mean(data[:, 0])
    std = np.std(data[:, 0])
    data[:, 0] = (data[:, 0] - mean) / std

    data = torch.tensor(data).to(device)
    dataset = {"train_data": data, "test_data": data}

    model = MGKMADE(
        n_comps=5,
        data_l=6,
        hidden_layers=[],
        grid=20,
        k=3,
        seed=2024,
        input_order="sequential",
        mode="sequential",
        device=device,
        ckpt_path="checkpoints/made",
    )

    model.fit(
        dataset=dataset,
        opt="Adam",
        steps=1000,
        lamb=1e-3,
        loss_fn=loss_function,
        lr=1e-2,
        batch=10000,
    )
    model.fit(
        dataset=dataset,
        opt="Adam",
        steps=500,
        lamb=1e-3,
        loss_fn=loss_function,
        lr=1e-3,
        batch=10000,
    )

    os.makedirs("outputs", exist_ok=True)
    save_model(model, "outputs/GW150914")

    model.auto_symbolic()
    model.fit(
        dataset=dataset,
        opt="Adam",
        steps=1000,
        loss_fn=loss_function,
        lr=1e-4,
        batch=10000,
    )

    save_expr(model, "outputs/GW150914")
