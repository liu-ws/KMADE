import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import corner


def discrete_sample(
    p: torch.Tensor, num_samples: int, device: "str | torch.device" = "cuda:0"
) -> torch.Tensor:
    """
    Sample from a discrete distribution.

    Args:
        p : torch.Tensor
            Discrete distribution with N values, shape (batch, N).
        num_samples : int
            Number of samples to draw.
        device : str or torch.device, optional
            Device for computation. Default: "cuda:0".

    Returns:
        torch.Tensor
            Indices of sampled values, shape (num_samples,).
    """

    # Calculating the cumulative distribution
    c = torch.cumsum(p[:, :-1], dim=1).unsqueeze(0).to(device)

    # generate random numbers
    r = torch.rand(num_samples, 1).to(device)

    # Index the sample by comparing the random number r with the cumulative distribution c
    # For each random number, if it is greater than some value in the cumulative distribution,
    # the corresponding index is counted
    # The index corresponding to the final sample of results returned

    return torch.sum((r > c).int(), dim=2)[0]


# MMD
def gaussian_kernel_mmd(
    sample1: np.ndarray,
    sample2: np.ndarray,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float = None,
) -> np.ndarray:
    """
    Compute the Gaussian kernel matrix for MMD.

    Args:
        sample1 : np.ndarray
            First sample, shape (n1, d).
        sample2 : np.ndarray
            Second sample, shape (n2, d).
        kernel_mul : float, optional
            Multiplier for bandwidth. Default: 2.0.
        kernel_num : int, optional
            Number of bandwidths. Default: 5.
        fix_sigma : float, optional
            Fixed bandwidth. If None, use median heuristic.

    Returns:
        np.ndarray
            Kernel matrix, shape (n1+n2, n1+n2).
    """

    n_samples = sample1.shape[0] + sample2.shape[0]
    total = np.concatenate(
        [sample1, sample2], axis=0
    )  # Merge source, target in direction of column

    # For the total sample transformation format is (1,n1+n2,d), then the latter two dimensions data are copied to the newly expanded dimensions (n1+n2,n1+n2,d), [[total],[total]...]
    total0 = np.expand_dims(total, axis=0).repeat(total.shape[0], axis=0)

    # For the total sample transformation format is (n1+n2,1,d), then the latter two dimensions data are copied to the newly expanded dimensions (n1+n2,n1+n2,d), [[total[0]*(n1+n2)], [total[1]*(n1+n2)] ...]
    total1 = np.expand_dims(total, axis=1).repeat(total.shape[0], axis=1)

    # Compute the sum between any two data, the coordinates (i,j) in the resulting matrix represent the l2 distance between the ith and jth rows of data in total (0 for i==j)
    L2_distance = np.sum((total0 - total1) ** 2, axis=2)

    # Adjusting the sigma value of the Gaussian kernel function
    if fix_sigma:
        bandcenter = fix_sigma
    else:
        bandcenter = np.sum(L2_distance) / (n_samples**2 - n_samples)

    # Take fix_sigma as the median value and kernel_mul as the multiplier, and take kernel_num bandwidth values (e.g., when fix_sigma is 1, we get [0.25,0.5,1,2,4]).
    bandcenter /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandcenter * (kernel_mul**i) for i in range(kernel_num)]

    # Math expression of Gaussian kernel function
    kernel_val = [
        np.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list
    ]

    return sum(kernel_val)


def MMD(
    sample1: np.ndarray,
    sample2: np.ndarray,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two samples.

    Args:
        sample1 : np.ndarray
            First sample, shape (n1, d).
        sample2 : np.ndarray
            Second sample, shape (n2, d).
        kernel_mul : float, optional
            Multiplier for bandwidth. Default: 2.0.
        kernel_num : int, optional
            Number of bandwidths. Default: 5.
        fix_sigma : float, optional
            Fixed bandwidth. If None, use median heuristic.

    Returns:
        float
            MMD loss value.
    """

    sample1 = np.array(sample1)
    sample2 = np.array(sample2)

    s = sample1.shape[0]

    kernels = gaussian_kernel_mmd(
        sample1,
        sample2,
        kernel_mul=kernel_mul,
        kernel_num=kernel_num,
        fix_sigma=fix_sigma,
    )

    # Divide the kernel matrix to 4 parts
    XX = kernels[:s, :s]
    YY = kernels[s:, s:]
    XY = kernels[:s, s:]
    YX = kernels[s:, :s]
    loss = np.mean(XX + YY - XY - YX)

    return loss


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Jensen-Shannon divergence between two distributions using PyTorch.

    Args:
        p : torch.Tensor
            First distribution.
        q : torch.Tensor
            Second distribution.

    Returns:
        torch.Tensor
            JS divergence.
    """

    # Calculate the average distribution
    m = 0.5 * (p + q)

    # Calculate JS divergence
    jsd = 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))
    return jsd


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Kullback-Leibler divergence using PyTorch.

    Args:
        p : torch.Tensor
            First distribution.
        q : torch.Tensor
            Second distribution.

    Returns:
        torch.Tensor
            KL divergence.
    """

    # Avoid log of zero
    p = torch.clamp(p, min=1e-10)
    q = torch.clamp(q, min=1e-10)
    return torch.sum(p * torch.log(p / q), dim=-1)


def pp_plot(
    sample1: np.ndarray,
    sample2: np.ndarray,
    color_bar,
    column_names: list,
    sample_lw: float = 1,
    path: str = None,
    figure_size: tuple = (5, 5),
    axes_kwargs: dict = {
        "width": 2,
        "length": 4,
        "size": 12,
        "color": "black",
        "rotationx": 0,
        "rotationy": 0,
    },
    legend_kwargs: dict = {
        "loc": "best",
        "prop": {"size": 12},
        "ls": [],
        "handlelength": 1.5,
        "color": "black",
    },
    title_kwargs: dict = {
        "content": "",
        "size": 12,
        "loc": "center",
        "color": "black",
        "family": "Times New Roman",
        "weight": "bold",
        "if_title": False,
    },
    label_kwargs: dict = {
        "label1": "sample1",
        "label2": "sample2",
        "size": 12,
        "family": "Times New Roman",
        "color1": "black",
        "color2": "black",
    },
    ref_kwargs: dict = {
        "width": 2,
        "alpha": 0.5,
        "color": "red",
        "if_ref": False,
        "linestyle": "--",
    },
) -> None:
    """
    Plot the p-p plot for two samples on a single graph.

    Args:
        sample1 : np.ndarray
            First sample, shape (n, d).
        sample2 : np.ndarray
            Second sample, shape (n, d).
        color_bar : list or Colormap
            Color map or list of colors for each variable.
        column_names : list
            Names of variables.
        sample_lw : float, optional
            Line width of samples. Default: 1.
        path : str, optional
            Path to save the figure.
        figure_size : tuple, optional
            Size of the figure. Default: (5, 5).
        axes_kwargs : dict, optional
            Keyword arguments for axes.
        legend_kwargs : dict, optional
            Keyword arguments for legend.
        title_kwargs : dict, optional
            Keyword arguments for title.
        label_kwargs : dict, optional
            Keyword arguments for x/y labels.
        ref_kwargs : dict, optional
            Keyword arguments for reference line.

    Returns:
        None
    """

    num_columns = sample1.shape[1]  # Get the number of columns
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{bm}",
        }
    )
    # Define a color map for different columns
    if isinstance(color_bar, matplotlib.colors.Colormap):
        colors = color_bar(np.linspace(0, 1, num_columns))
    elif isinstance(color_bar, list):
        colors = color_bar
        assert (
            len(colors) == num_columns
        ), "The length of color_bar list must be equal to the number of columns"
    else:
        raise ValueError("color_bar must be a color map or a list of colors")

    fig = plt.figure(figsize=figure_size)

    for i in range(num_columns):

        sorted_sample1 = np.sort(sample1[:, i])
        sorted_sample2 = np.sort(sample2[:, i])
        min1 = sorted_sample1[0]
        min2 = sorted_sample2[0]
        max1 = sorted_sample1[-1]
        max2 = sorted_sample2[-1]

        min_val = min(min1, min2)
        max_val = max(max1, max2)

        nodes = np.linspace(min_val, max_val, 1000)
        cdf1 = np.zeros_like(nodes)
        cdf2 = np.zeros_like(nodes)

        # Calculate the cdf for each sample
        for j, node in enumerate(nodes):
            cdf1[j] = np.sum(sorted_sample1 <= node) / len(sorted_sample1)
            cdf2[j] = np.sum(sorted_sample2 <= node) / len(sorted_sample2)

        # Plot the p-p plot for each column with different colors
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        if len(legend_kwargs["ls"]) == 0:
            plt.plot(
                cdf1,
                cdf2,
                linestyle="-",
                color=colors[i],
                label=f"{column_names[i]}",
                linewidth=sample_lw,
            )
        else:
            plt.plot(
                cdf1,
                cdf2,
                linestyle=legend_kwargs["ls"][i],
                color=colors[i],
                label=f"{column_names[i]}",
                linewidth=sample_lw,
            )

    # Plot the 45-degree reference line
    if ref_kwargs["if_ref"]:
        plt.plot(
            [0, 1],
            [0, 1],
            color=ref_kwargs["color"],
            linestyle=ref_kwargs["linestyle"],
            linewidth=ref_kwargs["width"],
            label=r"$45^{\circ}$",
            alpha=ref_kwargs["alpha"],
        )  # 45-degree line

    legend = plt.legend(
        loc=legend_kwargs["loc"],
        labelcolor=legend_kwargs["color"],
        prop=legend_kwargs["prop"],
        handlelength=legend_kwargs["handlelength"],
    )

    for line in legend.get_lines():
        line.set_linewidth(sample_lw)

    # change the outlook of the axes and edges
    for ax in fig.get_axes():
        ax.spines["top"].set_linewidth(axes_kwargs["width"])
        ax.spines["right"].set_linewidth(axes_kwargs["width"])
        ax.spines["bottom"].set_linewidth(axes_kwargs["width"])
        ax.spines["left"].set_linewidth(axes_kwargs["width"])
        ax.tick_params(width=axes_kwargs["width"], length=axes_kwargs["length"])

    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(axes_kwargs["size"])
            tick.label1.set_color(axes_kwargs["color"])
            tick.label1.set_rotation(axes_kwargs["rotationx"])
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(axes_kwargs["size"])
            tick.label1.set_color(axes_kwargs["color"])
            tick.label1.set_rotation(axes_kwargs["rotationy"])

    # title
    plt.rcParams.update({"text.usetex": False, "font.family": title_kwargs["family"]})
    if title_kwargs["if_title"]:
        plt.rcParams.update(
            {"text.usetex": False, "font.family": title_kwargs["family"]}
        )
        plt.title(
            title_kwargs["content"],
            fontsize=title_kwargs["size"],
            loc=title_kwargs["loc"],
            color=title_kwargs["color"],
            family=title_kwargs["family"],
            weight=title_kwargs["weight"],
        )

    # label
    plt.rcParams.update({"text.usetex": False, "font.family": label_kwargs["family"]})
    plt.xlabel(
        label_kwargs["label1"],
        fontsize=label_kwargs["size"],
        color=label_kwargs["color1"],
    )
    plt.ylabel(
        label_kwargs["label2"],
        fontsize=label_kwargs["size"],
        color=label_kwargs["color2"],
    )

    if path is not None:
        plt.savefig(path, bbox_inches="tight")


# better corner plot
def corner_plot(
    data1: np.ndarray,
    data2: np.ndarray = None,
    path: str = None,
    pad_inches: float = 0.1,
    contour_kwargs: dict = {
        "colors1": [],
        "colors2": [],
        "levels": [0.5, 0.9],
        "width": [2, 2.5],
        "linestyles": ["-", "--"],
        "smooth": 1,
        "alpha": 0.5,
    },
    contourf_kwargs: dict = {"colors": []},
    hist_kwargs: dict = {
        "color": [],
        "width": [2, 2.5],
        "alpha": 1,
        "bins": 20,
        "smooth1d": 0.1,
    },
    var_kwargs: dict = {"names": [], "ranges": [], "size": 12, "pad": -0.2},
    axes_kwargs: dict = {
        "width": 2,
        "length": 4,
        "size": 12,
        "color": "black",
        "rotationx": 0,
        "rotationy": 0,
        "max_n_ticks": 3,
    },
    legend_kwargs: dict = {
        "legend1": "sample1",
        "legend2": "sample2",
        "loc1": (0, 0),
        "loc2": (0, 0),
        "size": 12,
        "family": "Times New Roman",
        "weight1": "bold",
        "weight2": "bold",
    },
) -> None:
    """
    Plot a corner plot for one or two datasets.

    Args:
        data1 : np.ndarray
            First dataset, shape (n, d).
        data2 : np.ndarray, optional
            Second dataset, shape (n, d).
        path : str, optional
            Path to save the figure.
        pad_inches : float, optional
            Padding for saving figure.
        contour_kwargs : dict, optional
            Keyword arguments for contour plot.
        contourf_kwargs : dict, optional
            Keyword arguments for contourf plot.
        hist_kwargs : dict, optional
            Keyword arguments for histograms.
        var_kwargs : dict, optional
            Variable names, ranges, etc.
        axes_kwargs : dict, optional
            Keyword arguments for axes.
        legend_kwargs : dict, optional
            Keyword arguments for legend.

    Returns:
        None
    """

    plt.rcdefaults()

    plt.rcParams.update(
        {
            "text.usetex": True,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "text.latex.preamble": r"\usepackage{bm}",
        }
    )

    fig = corner.corner(
        data=data1,
        range=var_kwargs["ranges"],
        bins=hist_kwargs["bins"],
        show_titles=False,
        labels=var_kwargs["names"],
        label_kwargs={"fontsize": var_kwargs["size"]},
        labelpad=var_kwargs["pad"],
        # quantiles=[0.05, 0.5, 0.95],
        hist_kwargs={
            "color": hist_kwargs["color"][0],
            "lw": hist_kwargs["width"][0],
            "ls": contour_kwargs["linestyles"][0],
            "alpha": hist_kwargs["alpha"],
        },
        smooth1d=hist_kwargs["smooth1d"],
        levels=contour_kwargs["levels"],
        smooth=contour_kwargs["smooth"],
        plot_density=False,
        plot_datapoints=False,
        no_fill_contours=(data2 is not None),
        fill_contours=(data2 is None),
        max_n_ticks=axes_kwargs["max_n_ticks"],
        contour_kwargs={
            "linewidths": contour_kwargs["width"][0],
            "linestyles": contour_kwargs["linestyles"][0],
            "colors": contour_kwargs["colors1"],
        },
        contourf_kwargs=contourf_kwargs,
    )
    if data2 is not None:
        corner.corner(
            data=data2,
            fig=fig,
            range=var_kwargs["ranges"],
            bins=hist_kwargs["bins"],
            # quantiles=[0.05, 0.5, 0.95],
            hist_kwargs={
                "color": hist_kwargs["color"][1],
                "lw": hist_kwargs["width"][1],
                "ls": contour_kwargs["linestyles"][1],
                "alpha": hist_kwargs["alpha"],
            },
            smooth1d=hist_kwargs["smooth1d"],
            levels=contour_kwargs["levels"],
            smooth=contour_kwargs["smooth"],
            plot_density=False,
            plot_datapoints=False,
            fill_contours=True,
            max_n_ticks=axes_kwargs["max_n_ticks"],
            contour_kwargs={
                "linewidths": contour_kwargs["width"][1],
                "linestyles": contour_kwargs["linestyles"][1],
                "colors": contour_kwargs["colors2"],
                "alpha": contour_kwargs["alpha"],
            },
            contourf_kwargs=contourf_kwargs,
        )

    # remove the space between subplots
    plt.subplots_adjust(hspace=0, wspace=0)

    # change the outlook of the axes and edges
    for ax in fig.get_axes():
        ax.spines["top"].set_linewidth(axes_kwargs["width"])
        ax.spines["right"].set_linewidth(axes_kwargs["width"])
        ax.spines["bottom"].set_linewidth(axes_kwargs["width"])
        ax.spines["left"].set_linewidth(axes_kwargs["width"])
        ax.tick_params(width=axes_kwargs["width"], length=axes_kwargs["length"])

    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(axes_kwargs["size"])
            tick.label1.set_color(axes_kwargs["color"])
            tick.label1.set_rotation(axes_kwargs["rotationx"])

        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(axes_kwargs["size"])
            tick.label1.set_color(axes_kwargs["color"])
            tick.label1.set_rotation(axes_kwargs["rotationy"])

    # legend
    if data2 is not None:
        plt.rcParams.update(
            {"text.usetex": False, "font.family": legend_kwargs["family"]}
        )
        fig.text(
            *legend_kwargs["loc1"],
            legend_kwargs["legend1"],
            ha="center",
            va="center",
            fontsize=legend_kwargs["size"],
            color=hist_kwargs["color"][0],
            weight=legend_kwargs["weight1"],
        )
        fig.text(
            *legend_kwargs["loc2"],
            legend_kwargs["legend2"],
            ha="center",
            va="center",
            fontsize=legend_kwargs["size"],
            color=hist_kwargs["color"][1],
            weight=legend_kwargs["weight2"],
        )

    if path is not None:
        plt.savefig(path, bbox_inches="tight", pad_inches=pad_inches)
