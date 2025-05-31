import yaml
from .utils import discrete_sample
import sympy as sp
from sympy import symbols
import numpy as np
import torch
import os


# forward function of symbolic expressions
def forward(funcs, input):
    output = np.zeros((input.shape[0], len(funcs)))
    for i, func in enumerate(funcs):
        output[:, i] = func(*input.T)
    return output


# sample with symbolic expressions of kmades
def sampler(path, n_samples, para=[], u=None):

    path = path + "_expr.yml"

    if not isinstance(para, list):
        raise TypeError("para must be a list")
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)

    expressions = config["expressions"]
    for i in range(len(expressions)):
        expressions[i] = sp.sympify(expressions[i])
    data_l, para_l, n_comps = config["data_l"], config["para_l"], config["n_comps"]
    input_degrees = np.array(config["input_degrees"])

    if len(para) != para_l:
        raise ValueError(f"para's length must equals {para_l}")

    para_data = symbols(f"x_1:{data_l + para_l + 1}")
    funcs = []
    for expr in expressions:
        f = sp.lambdify(para_data, expr, "numpy")
        funcs.append(f)

    u = (
        np.random.randn(n_samples, data_l) if u is None else u
    )  # samples of standard Gaussian distribution variable

    if n_comps == 1:
        para = np.repeat([para], repeats=n_samples, axis=0)
        samples = np.zeros([n_samples, data_l])
        input = np.column_stack((samples, para))

        for i in range(1, data_l + 1):
            idx = np.argwhere(input_degrees == i)[0, 0]

            m = forward([funcs[idx]], input).flatten()
            logp = forward([funcs[idx + data_l]], input).flatten()

            input[:, idx] = m + np.exp(-0.5 * logp) * u[:, idx]

        return input[:, 0:data_l]
    else:
        para = np.repeat([para], repeats=n_samples, axis=0)
        samples = np.zeros([n_samples, data_l])
        input = np.column_stack((samples, para))
        """
        Each component of the data corresponds to multiple u's and therefore one of these u's needs to be picked at random
        This is actually equivalent to picking different m and logp for the same u
        The probability of this selection is determined by loga
        """
        for i in range(1, data_l + 1):

            idx = np.argwhere(input_degrees == i)[0, 0]

            # we just need to compute the parameters of the idx^th conditional
            m = forward([funcs[idx + i * data_l] for i in range(n_comps)], input)
            logp = forward(
                [funcs[idx + i * data_l + data_l * n_comps] for i in range(n_comps)],
                input,
            )
            loga = forward(
                [
                    funcs[idx + i * data_l + data_l * n_comps * 2]
                    for i in range(n_comps)
                ],
                input,
            )

            loga -= np.log(np.sum(np.exp(loga), axis=1, keepdims=True))  # normalization
            z = (
                discrete_sample(torch.tensor(np.exp(loga)), num_samples=n_samples)
                .detach()
                .cpu()
                .numpy()
            )

            input[:, idx] = (
                m[np.arange(n_samples), z]
                + np.exp(-0.5 * logp[np.arange(n_samples), z]) * u[:, idx]
            )

        return input[:, 0:data_l]


# sample with symbolic expressions of kmafs
def sampler_kmaf(path, n_samples=1, n_mades=1, para=[]):
    u = None
    for i in range(n_mades):
        u = sampler(
            path=os.path.join(path, f"made{i+1}"), n_samples=n_samples, u=u, para=para
        )

    return u


# compute the u of the given data x using symbolic expressions of kmades
def compute_u(path, x):

    with open(path, "r") as stream:
        config = yaml.safe_load(stream)

    expressions = config["expressions"]
    for i in range(len(expressions)):
        expressions[i] = sp.sympify(expressions[i])

    data_l, para_l, n_comps = config["data_l"], config["para_l"], config["n_comps"]

    para_data = symbols(f"x_1:{data_l + para_l + 1}")
    funcs = []
    for expr in expressions:
        f = sp.lambdify(para_data, expr, "numpy")
        funcs.append(f)

    if n_comps == 1:
        # single gaussian
        m = forward(funcs[0:data_l], x)
        logp = forward(funcs[data_l : data_l * 2], x)
        u = (x[:, 0:data_l] - m) * np.exp(0.5 * logp)
        return u, m, logp

    else:
        m = forward(funcs[0 : data_l * n_comps], x).reshape(-1, n_comps, data_l)
        logp = forward(funcs[data_l * n_comps : data_l * n_comps * 2], x).reshape(
            -1, n_comps, data_l
        )
        loga = forward(funcs[data_l * n_comps * 2 : data_l * n_comps * 3], x).reshape(
            -1, n_comps, data_l
        )
        loga -= np.log(np.sum(np.exp(loga), axis=1, keepdims=True))  # normalization
        u = torch.exp(0.5 * logp) * (x[:, 0:data_l] - m)
        return u, m, logp, loga


# compute the likelihood of the given data x using symbolic expressions of mgkmade and cmgkmade
def likelihood(path, x, log=True):
    u, m, logp, loga = compute_u(path, x)
    data_l = u.shape[2]
    L = torch.log(torch.sum(torch.exp(loga - 0.5 * u**2 + 0.5 * logp), axis=1))
    L = -0.5 * data_l * torch.log(torch.tensor(2 * torch.pi)) + torch.sum(L, axis=1)
    if log:
        return L
    else:
        return torch.exp(L)


# compute the probability density function of the given data x using symbolic expressions of kmafs
def compute_kmaf(x, path, n_mades=1, log=True):

    # read the information of the last made
    with open(path + f"made{n_mades}_expr.yml", "r") as stream:
        config = yaml.safe_load(stream)
    data_l, n_comps = config["data_l"], config["n_comps"]

    logdet_dudx = 0.0

    for i in range(n_mades - 1):
        u, m, logp = compute_u(path + f"made{i}_expr.yml", x)
        data_l = m.shape[1]
        x[:, 0:data_l] = u

        logdet_dudx += 0.5 * np.sum(logp, axis=1)

    if n_comps == 1:
        u, m, logp = compute_u(path + f"made{n_mades}_expr.yml", x)
        x[:, 0:data_l] = u
        logdet_dudx += -0.5 * data_l * np.log(2 * np.pi) - 0.5 * np.sum(
            x[:, 0:data_l] ** 2, axis=1
        )
    else:
        logdet_dudx += likelihood(path + f"made{n_mades}_expr.yml", x=x, log=True)

    return logdet_dudx if log else np.exp(logdet_dudx)
