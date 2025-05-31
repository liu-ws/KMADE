import torch


def loss_function(pred, data):

    i, j = pred.size()
    _, data_l = data.size()

    if j // data_l == 2:
        m = pred[:, 0 : j // 2]
        logp = pred[:, j // 2 :]
        u = torch.exp(0.5 * logp) * (data - m)
        L = -torch.mean(
            -0.5
            * (
                j // 2 * torch.log(torch.tensor(2 * torch.pi, device=pred.device))
                + torch.sum(u**2 - logp, axis=1)
            )
        )
    else:
        n_comps = j // data_l // 3
        """
        The arrangement of the output is set as follows: the first 1/3 is m, the middle 1/3 is logp, and the last 1/3 is a
        Each 1/3 requires each n_comps output to be divided into a group corresponding to the same p_i
        The output needs to be reformatted into three three-dimensional matrices
        The first dimension represents different samples
        Corresponding to markdown, the second dimension represents j and the third dimension represents i
        """
        data = data.unsqueeze(1).repeat(1, n_comps, 1)

        m = pred[:, 0 : j // 3].reshape(-1, n_comps, j // 3 // n_comps)
        logp = pred[:, j // 3 : 2 * j // 3].reshape(-1, n_comps, j // 3 // n_comps)

        loga = pred[:, 2 * j // 3 :].reshape(-1, n_comps, j // 3 // n_comps)
        loga -= torch.log(
            torch.sum(torch.exp(loga), axis=1, keepdims=True)
        )  # normalization

        u = torch.exp(0.5 * logp) * (data - m)
        L = torch.log(torch.sum(torch.exp(loga - 0.5 * u**2 + 0.5 * logp), axis=1))
        L = -0.5 * j // 3 // n_comps * torch.log(
            torch.tensor(2 * torch.pi, device=pred.device)
        ) + torch.sum(L, axis=1)
        L = -torch.mean(L)
    return L
