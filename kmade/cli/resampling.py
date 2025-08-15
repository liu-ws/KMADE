import torch
import numpy as np
import h5py
from kmade.core.expr import sampler
from kmade.core.saving import load_model
from kmade.core.utils import corner_plot, pp_plot
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(2024)

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

    # get std and mean
    data = np.empty(posterior0914.shape)
    data[:, [0]] = np.log(posterior0914[:, [0]])
    mean = np.mean(data[:, 0])
    std = np.std(data[:, 0])

    # load the saved model
    model = load_model("outputs/GW150914")

    # number of samples is too large, so we need to sample many times

    sample_nn = np.empty(posterior0914.shape)
    sample_ana = np.empty(posterior0914.shape)
    # Wrap the loop with tqdm for a progress bar, showing only percentage
    for i in tqdm(range(36), desc="Sampling", bar_format="{desc}: {percentage:3.0f}%"):
        sample_ana[i * 4000 : (i + 1) * 4000] = sampler(
            path="outputs/GW150914", n_samples=4000
        )
        sample_nn[i * 4000 : (i + 1) * 4000] = (
            model.sample(n_samples=4000).detach().cpu().numpy()
        )
    sample_ana[144000:] = sampler(path="outputs/GW150914", n_samples=3634)
    sample_nn[144000:] = model.sample(n_samples=3634).detach().cpu().numpy()

    # inverse transformation of model samples
    sample_nn[:, 0] = np.exp(sample_nn[:, 0] * std + mean)
    sample_nn[:, [1, 2, 3]] = 1 / (1 + np.exp(-sample_nn[:, [1, 2, 3]]))
    sample_nn[:, [4, 5]] = np.pi / (1 + np.exp(-sample_nn[:, [4, 5]]))

    sample_ana[:, 0] = np.exp(sample_ana[:, 0] * std + mean)
    sample_ana[:, [1, 2, 3]] = 1 / (1 + np.exp(-sample_ana[:, [1, 2, 3]]))
    sample_ana[:, [4, 5]] = np.pi / (1 + np.exp(-sample_ana[:, [4, 5]]))

    # plots
    # keywords of corner plot
    minm = np.min(sample_nn[:, 0])
    maxm = np.max(sample_nn[:, 0])

    contour_kwargs = {
        "colors": [["#c39d21", "#8d4e00"], ["#0072c1"]],
        "levels": [0.5, 0.9],
        "widths": [2, 2],
        "linestyles": ["-", "-"],
        "smooth": 1,
        "alphas": [0, 1],
    }
    contourf_kwargs = {
        "colors": [["white", "#ffedb2", "#edd3b2"], None],
        "fill_contours": [True, False],
    }
    hist_kwargs = {
        "colors": ["#fa8b00", "#0072c1"],
        "widths": [2, 2],
        "linestyles": ["-", "-"],
        "alphas": [1, 1],
        "bins": 50,
        "smooth1d": 0.8,
    }
    var_kwargs = {
        "names": [
            r"$\mathcal{M} \left[\mathrm{M}_{\odot}\right]$",
            r"$q$",
            r"$a_{1}$",
            r"$a_{2}$",
            r"$\theta_1$",
            r"$\theta_2$",
        ],
        "ranges": [
            (minm, maxm),
            (-0.2, 1.2),
            (-0.2, 1.2),
            (-0.2, 1.2),
            (-0.2, np.pi + 0.2),
            (-0.35, np.pi + 0.2),
        ],
        "size": 40,
        "pad": 0.18,
    }
    axes_kwargs_corner = {
        "width": 2,
        "length": 8,
        "size": 35,
        "color": "black",
        "rotationx": 50,
        "rotationy": 50,
        "max_n_ticks": 3,
    }
    legend_kwargs_corner = {
        "legends": ["Resampled (network weights)", "Raw samples"],
        "locs": [(0.40, 0.88), (0.40, 0.94)],
        "size": 40,
        "family": "Times New Roman",
        "weights": ["normal", "normal"],
        "ha": "left",
        "va": "top",
    }
    # keywords of pp plot
    axes_kwargs_pp = {
        "width": 2,
        "weight": "bold",
        "length": 8,
        "size": 25,
        "color": "black",
        "rotationx": 45,
        "rotationy": 45,
    }
    legend_kwargs_pp = {
        "loc": "lower right",
        "prop": {"family": "Times New Roman", "weight": "bold", "size": 28},
        "ls": ["-", "-", "-", "-", "-", "-"],
        "handlelength": 2,
        "color": "black",
    }
    title_kwargs = {
        "content": "",
        "size": 12,
        "color": "black",
        "family": "Times New Roman",
        "weight": "bold",
        "loc": "center",
        "pad": 0.05,
        "if_title": False,
    }

    label_kwargs = {
        "labels": ["Raw samples", "Resampled"],
        "size": 35,
        "family": "Times New Roman",
        "colors": ["black", "black"],
        "pad": 0.1,
    }
    ref_kwargs = {
        "width": 2,
        "color": "red",
        "alpha": 0.8,
        "linestyle": "--",
        "if_ref": False,
    }

    # corner plots for nn resmpling results
    corner_plot(
        datas=[sample_nn, posterior0914],
        path="outputs/150914_nn.pdf",
        contour_kwargs=contour_kwargs,
        contourf_kwargs=contourf_kwargs,
        hist_kwargs=hist_kwargs,
        var_kwargs=var_kwargs,
        axes_kwargs=axes_kwargs_corner,
        legend_kwargs=legend_kwargs_corner,
    )

    # pp plots for nn resampling results
    pp_plot(
        sample1=posterior0914[:140000, :],
        sample2=sample_nn,
        sample_lw=2,
        figure_size=(10, 10),
        color_bar=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f"],
        column_names=[
            r"$\mathcal{M}$",
            r"$q$",
            r"$a_{1}$",
            r"$a_{2}$",
            r"$\theta_1$",
            r"$\theta_2$",
        ],
        path="outputs/150914_pp.pdf",
        axes_kwargs=axes_kwargs_pp,
        legend_kwargs=legend_kwargs_pp,
        title_kwargs=title_kwargs,
        label_kwargs=label_kwargs,
        ref_kwargs=ref_kwargs,
    )

    # corner plots for analytic expression resmpling results
    legend_kwargs_corner["legends"][1] = "Resampled (analytic expressions)"
    corner_plot(
        datas=[sample_ana, posterior0914],
        path="outputs/150914_ana.pdf",
        contour_kwargs=contour_kwargs,
        contourf_kwargs=contourf_kwargs,
        hist_kwargs=hist_kwargs,
        var_kwargs=var_kwargs,
        axes_kwargs=axes_kwargs_corner,
        legend_kwargs=legend_kwargs_corner,
    )

    # pp plots for symbolic expression resmpling results
    pp_plot(
        sample1=posterior0914[:140000, :],
        sample2=sample_ana,
        sample_lw=2,
        figure_size=(10, 10),
        color_bar=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f"],
        column_names=[
            r"$\mathcal{M}$",
            r"$q$",
            r"$a_{1}$",
            r"$a_{2}$",
            r"$\theta_1$",
            r"$\theta_2$",
        ],
        path="outputs/150914_anapp.pdf",
        axes_kwargs=axes_kwargs_pp,
        legend_kwargs=legend_kwargs_pp,
        title_kwargs=title_kwargs,
        label_kwargs=label_kwargs,
        ref_kwargs=ref_kwargs,
    )
