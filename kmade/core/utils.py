import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import corner


def discrete_sample(p, num_samples, device="cuda:0"):
    """
    sample from a discrete distribution
    p:            discrete distribution with N values
    num_samples:  number of samples
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
def gaussian_kernel_mmd(sample1, sample2, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

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


def MMD(sample1, sample2, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

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


def js_divergence(p, q):
    """
    Calculate the Jensen-Shannon divergence between two distributions using PyTorch.

    p: First distribution (torch tensor)
    q: Second distribution (torch tensor)
    """
    # Calculate the average distribution
    m = 0.5 * (p + q)

    # Calculate JS divergence
    jsd = 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))
    return jsd


def kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler divergence using PyTorch.

    p: First distribution (torch tensor)
    q: Second distribution (torch tensor)
    """
    # Avoid log of zero
    p = torch.clamp(p, min=1e-10)
    q = torch.clamp(q, min=1e-10)
    return torch.sum(p * torch.log(p / q), dim=-1)


def pp_plot(
    sample1,
    sample2,
    color_bar,
    column_names,
    sample_lw=1,
    path=None,
    figure_size=(5, 5),
    axes_kwargs={
        "width": 2,
        "length": 4,
        "size": 12,
        "color": "black",
        "rotationx": 0,
        "rotationy": 0,
    },
    legend_kwargs={
        "loc": "best",
        "prop": {"size": 12},
        "ls": [],
        "handlelength": 1.5,
        "color": "black",
    },
    title_kwargs={
        "content": "",
        "size": 12,
        "loc": "center",
        "color": "black",
        "family": "Times New Roman",
        "weight": "bold",
        "if_title": False,
    },
    label_kwargs={
        "label1": "sample1",
        "label2": "sample2",
        "size": 12,
        "family": "Times New Roman",
        "color1": "black",
        "color2": "black",
    },
    ref_kwargs={
        "width": 2,
        "alpha": 0.5,
        "color": "red",
        "if_ref": False,
        "linestyle": "--",
    },
):
    """
    Plot the p-p plot for two samples on a single graph

    column_names: names of variables
    color_bar: color map or list of colors for each variable
    sample_lw: line width of samples

    ref_kwargs: keyword arguments for reference line
                if_ref: whether to show reference line or not
                color: color of reference line
                width: line width of reference line
                alpha: transparency of reference line

    axes_kwargs: keyword arguments for axes
                width: width of egdes of the plot
                length: length of the axes ticks
                size: font size
                color: color of the axes numbers and edges of the plot
                rotationx: x-axis number rotation
                rotationy: y-axis number rotation

    legend_kwargs: keyword arguments for legend
                loc: location of the legend
                prop: font properties of the legend
                    size: font size
                ls: list of line styles for each column
                handlelength: length of the legend handles
                color: color of legend

    title_kwargs: keyword arguments for title
                content: content of the title
                size: font size of the title
                loc: location of the title
                color: color of the title
                family: font family of the title
                weight: thickness of the title
                if_title: whether to show title or not

    label_kwargs: keyword arguments for x y labels
                label1: label of sample1
                label2: label of sample2
                size: font size of the labels
                family: font family of the labels
                color1: color of label1
                color2: color of label2

    path: path to save the figure
    figure_size: size of the figure

    """
    num_columns = sample1.shape[1]  # Get the number of columns
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "text.latex.preamble": [r"\usepackage{bm}"],
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


def qq_plot(
    sample1,
    sample2,
    column_names,
    rows,
    cols,
    figure_size,
    path=None,
    ref_color="red",
    ref_lw=2,
    axes_kwargs={
        "width": 2,
        "size": 12,
        "color": "black",
        "rotationx": 0,
        "rotationy": 0,
    },
    legend_kwargs={"loc": "best", "prop": {"size": 12}, "color": "black"},
    title_kwargs={
        "content": "",
        "size": 12,
        "color": "black",
        "family": "Times New Roman",
        "weight": "bold",
        "pad": 0.05,
        "if_title": False,
    },
    label_kwargs={
        "label1": "sample1",
        "label2": "sample2",
        "size": 12,
        "family": "Times New Roman",
        "color1": "black",
        "color2": "black",
        "weight1": "bold",
        "weight2": "bold",
        "pad": 0,
    },
):
    """
    sample1:
    sample2:
    column_names: names of variables
    rows: number of rows of subplots
    cols: number of columns of subplots
    figure_size: size of the figure
    path: path to save the figure
    ref_color: color of 45 degree reference line
    ref_lw: line width of 45 degree reference line

    axes_kwargs: keyword arguments for axes
                width: width of egdes of the plot
                family: font family for axes numbers
                size: font size of axes numbers
                color: color of the axes numbers and edges of the plot
                rotationx: x-axis number rotation
                rotationy: y-axis number rotation

    legend_kwargs: keyword arguments for legend
                loc: location of the legend
                prop: font properties of the legend
                    size: font size of the legend
                color: color of legend

    title_kwargs: keyword arguments for title
                content: content of the title
                size: font size of the title
                color: color of the title
                family: font family of the title
                weight: thickness of the title
                pad: distance of the title from the top of the figure
                if_title: whether to show title or not

    label_kwargs: keyword arguments for x y labels
                label1: label of sample1
                label2: label of sample2
                size: font size of the labels
                family: font family of the labels
                color1: color of label1
                color2: color of label2
                weight1: thickness of label1
                weight2: thickness of label2
                pad: distance of the labels from the edges of the figure

    """
    # Ensure the samples are of the same shape
    assert sample1.shape == sample2.shape, "Samples must have the same shape"
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "text.usetex": True,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "text.latex.preamble": [r"\usepackage{bm}"],
        }
    )

    num_features = sample1.shape[1]  # Number of physical quantities
    total_plots = rows * cols  # Total number of subplots
    fig, axes = plt.subplots(
        rows, cols, figsize=(figure_size[0] * cols, figure_size[1] * rows)
    )  # Create subplots
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i in range(num_features):
        # Get the i-th column from both samples
        data1 = sample1[:, i]
        data2 = sample2[:, i]

        # Sort the data
        sorted_data1 = np.sort(data1)
        sorted_data2 = np.sort(data2)

        # Calculate the quantiles
        quantiles1 = np.percentile(sorted_data1, np.linspace(0, 100, len(sorted_data1)))
        quantiles2 = np.percentile(sorted_data2, np.linspace(0, 100, len(sorted_data2)))

        # Create Q-Q plot
        axes[i].scatter(quantiles1, quantiles2, s=5, label=f"{column_names[i]}")
        axes[i].plot(
            [min(quantiles1), max(quantiles1)],
            [min(quantiles1), max(quantiles1)],
            color=ref_color,
            linestyle="--",
            linewidth=ref_lw,
            label="45 Degree",
        )  # Reference line
        axes[i].legend(
            loc=legend_kwargs["loc"],
            prop=legend_kwargs["prop"],
            labelcolor=legend_kwargs["color"],
        )

    # Hide any unused subplots if num_features < total_plots
    for j in range(num_features, total_plots):
        fig.delaxes(axes[j])
    fig.tight_layout()  # Adjust layout to prevent overlap

    for ax in fig.get_axes():
        ax.spines["top"].set_linewidth(axes_kwargs["width"])
        ax.spines["right"].set_linewidth(axes_kwargs["width"])
        ax.spines["bottom"].set_linewidth(axes_kwargs["width"])
        ax.spines["left"].set_linewidth(axes_kwargs["width"])
        ax.tick_params(width=axes_kwargs["width"])

    for ax in fig.get_axes():
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontproperties(axes_kwargs["family"])
            tick.label1.set_fontsize(axes_kwargs["size"])
            tick.label1.set_fontweight(axes_kwargs["weight"])
            tick.label1.set_color(axes_kwargs["color"])
            tick.label1.set_rotation(axes_kwargs["rotationx"])
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontproperties(axes_kwargs["family"])
            tick.label1.set_fontsize(axes_kwargs["size"])
            tick.label1.set_fontweight(axes_kwargs["weight"])
            tick.label1.set_color(axes_kwargs["color"])
            tick.label1.set_rotation(axes_kwargs["rotationy"])

    plt.rcParams.update({"text.usetex": False, "font.family": title_kwargs["family"]})
    if title_kwargs["if_title"]:
        fig.text(
            0.5,
            1 + title_kwargs["pad"],
            title_kwargs["content"],
            ha="center",
            va="top",
            fontsize=title_kwargs["size"],
            color=title_kwargs["color"],
            weight=title_kwargs["weight"],
        )

    plt.rcParams.update({"text.usetex": False, "font.family": label_kwargs["family"]})
    fig.text(
        0.5,
        0 - label_kwargs["pad"],
        label_kwargs["label1"],
        fontsize=label_kwargs["size"],
        color=label_kwargs["color1"],
        weight=label_kwargs["weight1"],
        ha="center",
    )
    fig.text(
        0 - label_kwargs["pad"],
        0.5,
        label_kwargs["label2"],
        fontsize=label_kwargs["size"],
        color=label_kwargs["color2"],
        weight=label_kwargs["weight2"],
        va="center",
        rotation="vertical",
    )

    if path is not None:
        plt.savefig(path, bbox_inches="tight")


# better corner plot
def corner_plot(
    data1,
    data2=None,
    path=None,
    pad_inches=0.1,
    contour_kwargs={
        "colors1": [],
        "colors2": [],
        "levels": [0.5, 0.9],
        "width": [2, 2.5],
        "linestyles": ["-", "--"],
        "smooth": 1,
        "alpha": 0.5,
    },
    contourf_kwargs={"colors": []},
    hist_kwargs={
        "color": [],
        "width": [2, 2.5],
        "alpha": 1,
        "bins": 20,
        "smooth1d": 0.1,
    },
    var_kwargs={"names": [], "ranges": [], "size": 12, "pad": -0.2},
    axes_kwargs={
        "width": 2,
        "length": 4,
        "size": 12,
        "color": "black",
        "rotationx": 0,
        "rotationy": 0,
        "max_n_ticks": 3,
    },
    legend_kwargs={
        "legend1": "sample1",
        "legend2": "sample2",
        "loc1": (0, 0),
        "loc2": (0, 0),
        "size": 12,
        "family": "Times New Roman",
        "weight1": "bold",
        "weight2": "bold",
    },
):
    """
    counter_kwargs: keyword arguments for contour plot
                colors1: counter colors of data1
                colors2: counter colors of data2
                levels: levels of the contours
                width: width of the contours
                linestyles: linestyles of the contours
                smooth: smoothing factor of the contours
                alpha: transparency of the contours

    contourf_kwargs:keyword arguments for contourf plot
                colors: colors between counters

    hist_kwargs: keyword arguments for hists of data1 and data2
                color: color of the hists
                width: width of the hists
                alpha: transparency of the hists
                bins: number of bins of the hists
                smooth1d: smoothing factor of the hists

    axes_kwargs: keyword arguments for axes
                width: width of the edges of the figure
                length: length of the ticks
                size: font size of the axes numbers
                color: color of the axes numbers
                rotationx: rotation of the x-axis numbers
                rotationy: rotation of the y-axis numbers
                max_n_ticks: maximum number of ticks on the axes

    legend_kwargs: keyword arguments for legend
                legend1: legend of data1
                legend2: legend of data2
                loc1: location of legend1
                loc2: location of legend2
                size: font size of the legend
                family: font family of the legend
                weight1: thickness of legend1
                weight2: thickness of legend2
    """
    plt.rcdefaults()

    plt.rcParams.update(
        {
            "text.usetex": True,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "text.latex.preamble": [r"\usepackage{bm}"],
        }
    )

    fig = corner.corner(
        data=data1,
        range=var_kwargs["ranges"],
        bins=hist_kwargs["bins"],
        show_titles=False,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
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
