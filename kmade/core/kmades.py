from .saving import save_model, load_model
from .utils import discrete_sample
from .fitting import modified_fit
import torch
import numpy as np
from kan import MultKAN


def create_degrees(data_l, n_hiddens, input_order, mode, seed):
    """
    Create a degree for each node. Nodes in the hidden layer can only connect to nodes whose degree is less than or equal to their own degree in the lower layer
    The input para must be fully connected to the hidden-layers. Therefore, you only need to create degrees on the input data

    data_l:      data dimension
    n_hiddens:   list of number of nodes in hidden-layers
    input_order: arrangement of input-layer's degrees random/sequential/custom list
    mode:        arrangement of hidden-layers' degrees random/sequential
    """

    degrees = []
    torch.manual_seed(seed)  # set random seeds

    if isinstance(input_order, str):

        if input_order == "random":
            degrees_0 = torch.arange(1, data_l + 1)
            degrees_0 = degrees_0[torch.randperm(degrees_0.size(0))]

        elif input_order == "sequential":
            degrees_0 = torch.arange(1, data_l + 1)

        else:
            raise ValueError("invalid input order")

    else:
        input_order = torch.tensor(input_order)
        # to verify that input_order contains all numbers from 1 to n_inputs
        assert torch.all(
            torch.sort(input_order)[0] == torch.arange(1, data_l + 1)
        ), "invalid input order"
        degrees_0 = input_order
    degrees.append(degrees_0)

    # create degrees for hidden-layers
    if mode == "random":
        for N in n_hiddens:
            min_prev_degree = min(
                torch.min(degrees[-1]), data_l - 1
            )  # the smaller one in (the minimum value of previous layer,n_inputs-1)  to ensure that previous layer contains no useless node
            degrees_l = torch.randint(min_prev_degree, data_l, (N,))  # torch.randint
            degrees.append(degrees_l)

    elif mode == "sequential":
        for N in n_hiddens:
            degrees_l = torch.arange(N) % max(1, data_l - 1) + min(1, data_l - 1)
            # if n_inputs>=3,range is [1,n_inputs-1]
            # if n_inputs=2,all are 1
            # if n_inputs=1,all are 0
            degrees.append(degrees_l)

    else:
        raise ValueError("invalid mode")

    return degrees


def create_masks(degrees, n_comps, device):
    """
    degrees: list
    n_comps: The number of Gaussian distributions used to model the probability distributions

    In the single Gaussian case, the output dimension is data_l*2. The first half is m and the second half is logp
    In the mixed Gaussian case, the output dimension is data_l*n_comps*3. The first 1/3 is m, the middle 1/3 is logp, and the last 1/3 is loga

    """

    Ms = []  # mask matrixs

    for layer_idx, (d0, d1) in enumerate(
        zip(degrees[:-1], degrees[1:])
    ):  # lower layer and upper layer pairing
        M = (d0[:, None] <= d1).to(
            device, torch.float
        )  # Convert to column vectors, compare to generate a Boolean matrix, and convert to 0,1
        Ms.append(M)

    # create degrees of output-layer according to degrees of input-layer
    if n_comps == 1:  # single gaussian
        last_degrees = torch.cat((degrees[0], degrees[0]))
    else:  # mixture of gaussians
        last_degrees = torch.cat((degrees[0],) * (n_comps * 3))  # copy n_comps*3

    Mmp = (degrees[-1][:, None] < last_degrees).to(
        device, torch.float
    )  # the mask matrix from hidden-layer to output layer
    Ms.append(Mmp)

    return Ms, last_degrees


def create_masks_condition(model, device):
    Ms, last_degrees = create_masks(
        degrees=model.degrees, n_comps=model.n_comps, device=device
    )
    tMs = []
    hidden_layers = model.hidden_layers
    para_hidden_layers = model.para_hidden_layers
    data_l = model.data_l
    para_l = model.para_l
    """
    decouple the data side and para side
    ll length of hidden-layer in the data side of lower layer
    lr length of hidden-layer in the para side of lower layer
    hl length of hidden-layer in the data side of upper layer
    hr length of hidden-layer in the para side of upper layer
    remove the connections between ll-hr lr-hl
    """
    hl = data_l
    hr = para_l
    for h in range(len(hidden_layers)):
        if h == 0:
            ll = data_l
            lr = para_l
            hl = hidden_layers[h]
            hr = para_hidden_layers[h]
        else:
            ll = hidden_layers[h - 1]
            lr = para_hidden_layers[h - 1]
            hl = hidden_layers[h]
            hr = para_hidden_layers[h]
        M = torch.zeros(ll + lr, hl + hr)
        M[0:ll, 0:hl] = Ms[h]
        M[ll:, hl:] = torch.ones(lr, hr)
        tMs.append(M.to(device, torch.float))
    M = torch.ones(hl + hr, model.width[-1][0])
    M[0:hl, :] = Ms[-1]
    tMs.append(M.to(device, torch.float))

    return tMs, last_degrees


class SGKMADE(MultKAN):
    """
    unconditional
    single gaussian
    """

    def __init__(
        self,
        data_l,
        hidden_layers=[],
        input_order="sequential",
        mode="sequential",
        ifperturb=False,
        bias_mean=0,
        bias_std=1,
        **kwargs,
    ):
        """
        data_l:        data dimension
        hidden_layers: list of number of nodes in hidden-layers
        input_order:   arrangement of input-layer's degrees
        mode:          arrangement of hidden-layer's degrees
        """
        self.n_comps = 1
        self.para_l = 0
        self.para_hidden_layers = []

        self.data_l = data_l
        self.hidden_layers = hidden_layers
        self.input_order = input_order
        self.mode = mode

        self.seed = kwargs.get("seed", 1)

        self.ifperturb = ifperturb
        self.bias_mean = bias_mean
        self.bias_std = bias_std

        super().__init__(width=self.creat_width(), affine_trainable=True, **kwargs)

        # Create degrees and masks
        self.degrees = create_degrees(
            data_l=self.data_l,
            n_hiddens=self.hidden_layers,
            input_order=self.input_order,
            mode=self.mode,
            seed=self.seed,
        )
        self.Ms, last_degrees = create_masks(
            degrees=self.degrees, n_comps=self.n_comps, device=self.device
        )
        self.degrees.append(last_degrees)

        # perturb the bias of the last layer
        if ifperturb:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            exec(
                f"self.node_bias_{len(hidden_layers)} = torch.nn.Parameter(torch.normal({bias_mean},{bias_std},(data_l*2,)).to(self.device)).requires_grad_(True)"
            )
            exec(f"self.node_bias[-1]=self.node_bias_{len(hidden_layers)}")

        # Apply masks to activation functions
        for layer_idx, M in enumerate(self.Ms):
            self.act_fun[layer_idx].mask = torch.nn.Parameter(M).requires_grad_(False)

    def creat_width(self):
        # create the 'width' parameter of KAN network [list of number of nodes in all layers]
        width = []
        width.append(self.data_l)
        for h in self.hidden_layers:
            width.append(h)
        width.append(self.data_l * 2)
        return width

    def compute_u(self, data):
        """
        Used to calculate the standard Gaussian distribution variable corresponding to the given data
        data: torch.tensor
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be torch.Tensor")

        if data.shape[1] != self.data_l:
            raise ValueError(f"data's column number must equals {self.data_l}")

        data = data.to(device=self.device)
        pred = self.forward(data)

        m = pred[:, 0 : self.data_l]
        logp = pred[:, self.data_l :]
        u = torch.exp(0.5 * logp) * (data - m)

        return u, m, logp

    def likelihood(self, data, log=True):
        """
        Used to calculate the likelihood of given data
        data torch.tensor
        log  whether the output is log-likelihood
        """

        u, m, logp = self.compute_u(data)
        L = -0.5 * (
            self.data_l * torch.log(torch.tensor(2 * torch.pi))
            + torch.sum(u**2 - logp, axis=1)
        )
        if log:
            return L
        else:
            return torch.exp(L)

    def sample(self, n_samples, u=None):
        """
        Used to generate samples
        n_samples: number of samples
        u:         samples of standard Gaussian distribution variable [n_samples,data_l]

        step iterations
        generate the elements of x one by one according to u in order of degrees of input-layer
        """
        if u is not None:
            if not isinstance(u, torch.Tensor):
                raise TypeError("u must be torch.Tensor")

            if u.shape[1] != self.data_l or u.shape[0] != n_samples:
                raise ValueError(f"u's shape must be {[n_samples,self.data_l]}")
            u = u.to(device=self.device)

        input_degree = self.degrees[0].detach().cpu().numpy()

        x = torch.zeros(n_samples, self.data_l, device=self.device)

        u = (
            torch.randn(n_samples, self.data_l) if u is None else u
        )  # samples of standard Gaussian distribution variable
        u = u.to(device=self.device)

        for i in range(1, self.data_l + 1):
            pred = self.forward(x)
            m = pred[:, 0 : self.data_l]
            logp = pred[:, self.data_l :]

            idx = np.argwhere(input_degree == i)[
                0, 0
            ]  # the index of the first element in input_order whose degree equals i
            x[:, idx] = m[:, idx] + torch.exp(-0.5 * logp[:, idx]) * u[:, idx]

            # x[:, idx] = m[:, idx] + torch.exp(torch.minimum(-0.5 * logp[:, idx], torch.tensor(10.0))) * u[:, idx]

        return x

    def saveckpt(self, path=None, dtype=torch.float32):

        save_model(self, path=path, dtype=dtype)

    def loadckpt(self, path, dtype=torch.float32):

        loaded_model = load_model(path=path, dtype=dtype)
        return loaded_model

    def fit(self, **kwargs):

        results = modified_fit(self, **kwargs)

        return results


class MGKMADE(MultKAN):
    """
    unconditional
    mixture of  gaussians
    """

    def __init__(
        self,
        data_l,
        n_comps=2,
        hidden_layers=[],
        input_order="sequential",
        mode="sequential",
        ifperturb=False,
        bias_mean=0,
        bias_std=1,
        **kwargs,
    ):
        """
        data_l:        data dimension
        n_comps:       number of Gaussian distributions used to model the probability distributions
        hidden_layers: list of number of nodes in hidden-layers
        input_order:   arrangement of input-layer's degrees
        mode:          arrangement of hidden-layer's degrees
        """

        self.data_l = data_l
        self.para_l = 0
        self.n_comps = n_comps
        self.hidden_layers = hidden_layers
        self.para_hidden_layers = []

        self.input_order = input_order
        self.mode = mode

        self.seed = kwargs.get("seed", 1)

        self.ifperturb = ifperturb
        self.bias_mean = bias_mean
        self.bias_std = bias_std

        super().__init__(width=self.creat_width(), affine_trainable=True, **kwargs)

        self.degrees = create_degrees(
            data_l=self.data_l,
            n_hiddens=self.hidden_layers,
            input_order=self.input_order,
            mode=self.mode,
            seed=self.seed,
        )
        self.Ms, last_degrees = create_masks(
            degrees=self.degrees, n_comps=self.n_comps, device=self.device
        )
        self.degrees.append(last_degrees)

        # perturb the bias of the last layer
        if ifperturb:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            exec(
                f"self.node_bias_{len(hidden_layers)} = torch.nn.Parameter(torch.normal({bias_mean},{bias_std},(data_l*3*n_comps,)).to(self.device)).requires_grad_(True)"
            )
            exec(f"self.node_bias[-1]=self.node_bias_{len(hidden_layers)}")

        # mask
        for layer_idx in range(len(self.Ms)):  # different layers
            M = torch.nn.Parameter(self.Ms[layer_idx]).requires_grad_(False)
            self.act_fun[layer_idx].mask = M

    def creat_width(self):

        width = []
        width.append(self.data_l)
        for h in range(len(self.hidden_layers)):
            width.append(self.hidden_layers[h])
        width.append(self.data_l * 3 * self.n_comps)
        return width

    def compute_u(self, data):

        if not isinstance(data, torch.Tensor):
            raise TypeError("data must be torch.Tensor")

        if data.shape[1] != self.data_l:
            raise ValueError(f"data's column number must equals {self.data_l}")

        data = data.to(device=self.device)
        pred = self.forward(data)
        data = data.unsqueeze(1).repeat(1, self.n_comps, 1)

        data_l = self.data_l
        n_comps = self.n_comps

        m = pred[:, 0 : data_l * n_comps].reshape(-1, n_comps, data_l)
        logp = pred[:, data_l * n_comps : 2 * data_l * n_comps].reshape(
            -1, n_comps, data_l
        )
        loga = pred[:, 2 * data_l * n_comps :].reshape(-1, n_comps, data_l)

        # normalization
        loga -= torch.log(torch.sum(torch.exp(loga), axis=1, keepdim=True))
        u = torch.exp(0.5 * logp) * (data - m)

        return u, m, logp, loga

    def likelihood(self, data, log=True):

        u, m, logp, loga = self.compute_u(data)

        L = torch.log(torch.sum(torch.exp(loga - 0.5 * u**2 + 0.5 * logp), axis=1))
        L = -0.5 * self.data_l * torch.log(torch.tensor(2 * torch.pi)) + torch.sum(
            L, axis=1
        )
        if log:
            return L
        else:
            return torch.exp(L)

    def sample(self, n_samples, u=None):

        if u is not None:
            if not isinstance(u, torch.Tensor):
                raise TypeError("u must be torch.Tensor")

            if u.shape[1] != self.data_l or u.shape[0] != n_samples:
                raise ValueError(f"u's shape must be {[n_samples,self.data_l]}")
            u = u.to(device=self.device)

        input_degree = self.degrees[0].detach().cpu().numpy()
        data_l = self.data_l
        n_comps = self.n_comps

        u = (
            torch.randn(n_samples, data_l) if u is None else u
        )  # samples of standard Gaussian distribution variable
        u = u.to(device=self.device)
        x = torch.zeros(n_samples, data_l, device=self.device)
        """
        Each component of the data corresponds to multiple u's and therefore one of these u's needs to be picked at random
        This is actually equivalent to picking different m and logp for the same u
        The probability of this selection is determined by loga
        """
        for i in range(1, data_l + 1):

            pred = self.forward(x)
            m = pred[:, 0 : data_l * n_comps].reshape(-1, n_comps, data_l)
            logp = pred[:, data_l * n_comps : data_l * n_comps * 2].reshape(
                -1, n_comps, data_l
            )

            loga = pred[:, data_l * n_comps * 2 :].reshape(-1, n_comps, data_l)
            loga -= torch.log(
                torch.sum(torch.exp(loga), axis=1, keepdims=True)
            )  # normalization

            idx = np.argwhere(input_degree == i)[0, 0]

            z = discrete_sample(
                torch.exp(loga[:, :, idx]), num_samples=n_samples, device=self.device
            )
            x[:, idx] = (
                m[torch.arange(n_samples), z, idx]
                + torch.exp(-0.5 * logp[torch.arange(n_samples), z, idx]) * u[:, idx]
            )

        return x

    def saveckpt(self, path=None, dtype=torch.float32):

        save_model(self, path=path, dtype=dtype)

    def loadckpt(self, path, dtype=torch.float32):

        loaded_model = load_model(path=path, dtype=dtype)
        return loaded_model

    def fit(self, **kwargs):
        results = modified_fit(self, **kwargs)
        return results


class CSGKMADE(MultKAN):
    """
    Conditional
    Single gaussian
    """

    def __init__(
        self,
        data_l,
        para_l,
        hidden_layers=[],
        para_hidden_layers=[],
        input_order="sequential",
        mode="sequential",
        ifperturb=False,
        bias_mean=0,
        bias_std=1,
        **kwargs,
    ):
        """
        data_l:        data dimension
        para_l:        para dimension
        n_comps:       number of Gaussian distributions used to model the probability distributions
        hidden_layers: list of number of nodes in hidden-layers
        para_hidden_layers: list of number of nodes in para side hidden-layers
        input_order:   arrangement of input-layer's degrees
        mode:          arrangement of hidden-layer's degrees
        """

        self.data_l = data_l
        self.para_l = para_l
        self.n_comps = 1

        if len(hidden_layers) != len(para_hidden_layers):
            raise ValueError(
                "hidden_layers and para_hidden_layers must have same length"
            )
        self.hidden_layers = hidden_layers
        self.para_hidden_layers = para_hidden_layers

        self.input_order = input_order
        self.mode = mode
        self.seed = kwargs.get("seed", 1)

        self.ifperturb = ifperturb
        self.bias_mean = bias_mean
        self.bias_std = bias_std

        super().__init__(width=self.creat_width(), affine_trainable=True, **kwargs)

        self.degrees = create_degrees(
            data_l=self.data_l,
            n_hiddens=self.hidden_layers,
            input_order=self.input_order,
            mode=self.mode,
            seed=self.seed,
        )
        self.Ms, last_degrees = create_masks_condition(self, device=self.device)
        self.degrees.append(last_degrees)

        # perturb the bias of the last layer
        if ifperturb:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            exec(
                f"self.node_bias_{len(hidden_layers)} = torch.nn.Parameter(torch.normal({bias_mean},{bias_std},(data_l*2,)).to(self.device)).requires_grad_(True)"
            )
            exec(f"self.node_bias[-1]=self.node_bias_{len(hidden_layers)}")

        # mask
        for layer_idx in range(len(self.Ms)):  # different layers
            M = torch.nn.Parameter(self.Ms[layer_idx]).requires_grad_(False)
            self.act_fun[layer_idx].mask = M

    def creat_width(self):
        width = []
        width.append(self.data_l + self.para_l)
        for h in range(len(self.hidden_layers)):
            width.append(self.hidden_layers[h] + self.para_hidden_layers[h])
        width.append(self.data_l * 2)
        return width

    def compute_u(self, data_para):
        """
        data_para: torch.Tensor data-parameter
        """
        if not isinstance(data_para, torch.Tensor):
            raise TypeError("data_para must be torch.Tensor")

        if data_para.shape[1] != self.data_l + self.para_l:
            raise ValueError(
                f"data_para's column number must equals {self.data_l+self.para_l}"
            )

        data_para = data_para.to(device=self.device)
        pred = self.forward(data_para)

        data = data_para[:, 0 : self.data_l]
        m = pred[:, 0 : self.data_l]
        logp = pred[:, self.data_l :]
        u = torch.exp(0.5 * logp) * (data - m)

        return u, m, logp

    def likelihood(self, data, log=True):

        u, m, logp = self.compute_u(data)
        L = -0.5 * (
            self.data_l * torch.log(torch.tensor(2 * torch.pi))
            + torch.sum(u**2 - logp, axis=1)
        )
        if log:
            return L
        else:
            return torch.exp(L)

    def sample(self, n_samples, para, u=None):
        """
        para: a list to decide the parameters
        """
        if u is not None:
            if not isinstance(u, torch.Tensor):
                raise TypeError("u must be torch.Tensor")

            if u.shape[1] != self.data_l or u.shape[0] != n_samples:
                raise ValueError(f"u's shape must be {[n_samples,self.data_l]}")
            u = u.to(device=self.device)

        if len(para) != self.para_l:
            raise ValueError(
                f"para's length {len(para)} don't equals parameter dimension {self.para_l}"
            )

        input_degree = self.degrees[0].detach().cpu().numpy()

        para = torch.tensor(para).repeat(n_samples, 1)
        samples = torch.zeros(n_samples, self.data_l)
        x = torch.column_stack((samples, para)).to(device=self.device)

        u = (
            torch.randn(n_samples, self.data_l) if u is None else u
        )  # samples of standard Gaussian distribution variable
        u = u.to(device=self.device)

        for i in range(1, self.data_l + 1):
            pred = self.forward(x)
            m = pred[:, 0 : self.data_l]
            logp = pred[:, self.data_l :]

            idx = np.argwhere(input_degree == i)[0, 0]
            # the index of the first element in input_order whose degree equals i
            x[:, idx] = m[:, idx] + torch.exp(-0.5 * logp[:, idx]) * u[:, idx]

            # x[:, idx] = m[:, idx] + torch.exp(torch.minimum(-0.5 * logp[:, idx], torch.tensor(10.0))) * u[:, idx]

        return x[:, 0 : self.data_l]

    def saveckpt(self, path=None, dtype=torch.float32):

        save_model(self, path=path, dtype=dtype)

    def loadckpt(self, path, dtype=torch.float32):

        loaded_model = load_model(path=path, dtype=dtype)
        return loaded_model

    def fit(self, **kwargs):
        results = modified_fit(self, **kwargs)
        return results


class CMGKMADE(MultKAN):
    """
    Conditional
    Mixture of gaussians
    """

    def __init__(
        self,
        data_l,
        para_l,
        n_comps,
        hidden_layers,
        para_hidden_layers,
        input_order="sequential",
        mode="sequential",
        ifperturb=False,
        bias_mean=0,
        bias_std=1,
        **kwargs,
    ):
        """
        data_l:             data dimension
        para_l:             para dimension
        n_comps:            number of Gaussian distributions used to model the probability distributions
        hidden_layers:      list of number of nodes in data side hidden-layers
        para_hidden_layers: list of number of nodes in para side hidden-layers

        input_order:   arrangement of input-layer's degrees
        mode:          arrangement of hidden-layer's degrees
        """

        self.data_l = data_l
        self.para_l = para_l
        self.n_comps = n_comps

        if len(hidden_layers) != len(para_hidden_layers):
            raise ValueError(
                "hidden_layers and para_hidden_layers must have same length"
            )
        self.hidden_layers = hidden_layers
        self.para_hidden_layers = para_hidden_layers

        self.input_order = input_order
        self.mode = mode
        self.seed = kwargs.get("seed", 1)

        self.ifperturb = ifperturb
        self.bias_mean = bias_mean
        self.bias_std = bias_std

        super().__init__(width=self.creat_width(), affine_trainable=True, **kwargs)

        self.degrees = create_degrees(
            data_l=self.data_l,
            n_hiddens=self.hidden_layers,
            input_order=self.input_order,
            mode=self.mode,
            seed=self.seed,
        )
        self.Ms, last_degrees = create_masks_condition(
            degrees=self.degrees, n_comps=self.n_comps, device=self.device
        )
        self.degrees.append(last_degrees)

        # perturb the bias of the last layer
        if ifperturb:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            exec(
                f"self.node_bias_{len(hidden_layers)} = torch.nn.Parameter(torch.normal({bias_mean},{bias_std},(data_l*3*n_comps,)).to(self.device)).requires_grad_(True)"
            )
            exec(f"self.node_bias[-1]=self.node_bias_{len(hidden_layers)}")

        # mask
        for layer_idx in range(len(self.Ms)):  # different layers
            M = torch.nn.Parameter(self.Ms[layer_idx]).requires_grad_(False)
            self.act_fun[layer_idx].mask = M

    def creat_width(self):

        width = []
        width.append(self.data_l + self.para_l)
        for h in range(len(self.hidden_layers)):
            width.append(self.hidden_layers[h] + self.para_hidden_layers[h])
        width.append(self.data_l * 3 * self.n_comps)
        return width

    def compute_u(self, data_para):
        """
        data_para torch.Tensor data-parameter
        """
        if not isinstance(data_para, torch.Tensor):
            raise TypeError("data_para must be torch.Tensor")

        if data_para.shape[1] != self.data_l + self.para_l:
            raise ValueError(
                f"data_para's column number must equals {self.data_l+self.para_l}"
            )

        data_para = data_para.to(device=self.device)
        pred = self.forward(data_para)
        data_l = self.data_l
        n_comps = self.n_comps

        data = data_para[:, 0 : self.data_l]
        data = data.unsqueeze(1).repeat(1, self.n_comps, 1)

        m = pred[:, 0 : data_l * n_comps].reshape(-1, n_comps, data_l)
        logp = pred[:, data_l * n_comps : 2 * data_l * n_comps].reshape(
            -1, n_comps, data_l
        )
        loga = pred[:, 2 * data_l * n_comps :].reshape(-1, n_comps, data_l)

        # normalization
        loga -= torch.log(torch.sum(torch.exp(loga), axis=1, keepdim=True))
        u = torch.exp(0.5 * logp) * (data - m)

        return u, m, logp, loga

    def likelihood(self, data, log=True):
        """
        data: torch.tensor data-parameters
        """
        u, m, logp, loga = self.compute_u(data)

        L = torch.log(torch.sum(torch.exp(loga - 0.5 * u**2 + 0.5 * logp), axis=1))
        L = -0.5 * self.data_l * torch.log(torch.tensor(2 * torch.pi)) + torch.sum(
            L, axis=1
        )
        if log:
            return L
        else:
            return torch.exp(L)

    def sample(self, n_samples, para, u=None):
        """
        para: a list to decide the parameters
        """
        if u is not None:
            if not isinstance(u, torch.Tensor):
                raise TypeError("u must be torch.Tensor")

            if u.shape[1] != self.data_l or u.shape[0] != n_samples:
                raise ValueError(f"u's shape must be {[n_samples,self.data_l]}")
            u = u.to(device=self.device)

        if len(para) != self.para_l:
            raise ValueError(
                f"para's length ({len(para)}) doesn't equal parameter dimension ({self.para_l})"
            )

        input_degree = self.degrees[0].detach().cpu().numpy()
        data_l = self.data_l
        n_comps = self.n_comps

        para = torch.tensor(para).repeat(n_samples, 1)
        samples = torch.zeros(n_samples, data_l)
        x = torch.column_stack((samples, para)).to(device=self.device)
        u = (
            torch.randn(n_samples, data_l) if u is None else u
        )  # samples of standard Gaussian distribution variable
        u = u.to(device=self.device)
        """
        Each component of the data corresponds to multiple u's and therefore one of these u's needs to be picked at random
        This is actually equivalent to picking different m and logp for the same u
        The probability of this selection is determined by loga
        """
        for i in range(1, data_l + 1):

            pred = self.forward(x)
            m = pred[:, 0 : data_l * n_comps].reshape(-1, n_comps, data_l)
            logp = pred[:, data_l * n_comps : data_l * n_comps * 2].reshape(
                -1, n_comps, data_l
            )

            loga = pred[:, data_l * n_comps * 2 :].reshape(-1, n_comps, data_l)
            loga -= torch.log(
                torch.sum(torch.exp(loga), axis=1, keepdims=True)
            )  # normalization

            idx = np.argwhere(input_degree == i)[0, 0]

            z = discrete_sample(
                torch.exp(loga[:, :, idx]), num_samples=n_samples, device=self.device
            )
            x[:, idx] = (
                m[torch.arange(n_samples), z, idx]
                + torch.exp(-0.5 * logp[torch.arange(n_samples), z, idx]) * u[:, idx]
            )

        return x[:, 0:data_l]

    def saveckpt(self, path=None, dtype=torch.float32):

        save_model(self, path=path, dtype=dtype)

    def load_ckpt(self, path, dtype=torch.float32):

        self = load_model(path=path, dtype=dtype)

        return self

    def fit(self, **kwargs):
        results = modified_fit(self, **kwargs)
        return results
