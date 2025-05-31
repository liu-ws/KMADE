import torch
import torch.nn as nn
from kan import KAN


class MGKVAE(nn.Module):
    def __init__(
        self,
        data_l,
        latent_l,
        latent_n,
        n_comps=1,
        encoder_hidden=[],
        decoder_hidden=[],
        ckpt_path="checkpoints/vae",
        **kwargs
    ):
        super(MGKVAE, self).__init__()

        encoder_width = [data_l] + encoder_hidden + [latent_l * n_comps * 3]
        decoder_width = [latent_l] + decoder_hidden + [data_l * n_comps * 3]
        self.encoder = KAN(
            width=encoder_width, ckpt_path=ckpt_path + "/encoder", **kwargs
        )
        self.decoder = KAN(
            width=decoder_width, ckpt_path=ckpt_path + "/decoder", **kwargs
        )
        self.vae = nn.ModuleList([self.encoder, self.decoder])

        self.data_l = data_l
        self.latent_l = latent_l
        self.latent_n = latent_n
        self.n_comps = n_comps
        self.device = kwargs["device"]

    def discrete_sample(self, p):
        c = torch.cumsum(p, dim=1)

        r = (
            torch.rand(p.shape[0], p.shape[2], device=self.device)
            .unsqueeze(1)
            .repeat(1, p.shape[1], 1)
        )

        index = torch.sum((r > c).int(), dim=1)

        return index

    def loss_fn(self, x, update_grid=False):

        data_l = self.data_l
        latent_l = self.latent_l
        latent_n = self.latent_n
        n_comps = self.n_comps

        if update_grid:
            self.encoder.update_grid(x)

        # repeat latent_n
        x = x.repeat(latent_n, 1)
        pred_e = self.encoder(x)
        m_e = pred_e[:, 0 : latent_l * n_comps].reshape(-1, n_comps, latent_l)  # mean
        logp_e = pred_e[:, latent_l * n_comps : latent_l * n_comps * 2].reshape(
            -1, n_comps, latent_l
        )  # standard deviation

        loga_e = pred_e[:, latent_l * n_comps * 2 :].reshape(
            -1, n_comps, latent_l
        )  # combination coefficient
        loga_e = loga_e - torch.logsumexp(loga_e, dim=1, keepdim=True)  # normalization

        # sample from p(h|x)
        u_e = torch.randn(
            x.shape[0], latent_l, device=self.device
        )  # reparameterization

        # sample indexes
        idx = self.discrete_sample(torch.exp(loga_e))
        idx_0 = torch.arange(x.shape[0]).unsqueeze(1).repeat(1, latent_l).flatten()
        idx_1 = idx.flatten()
        idx_2 = torch.arange(latent_l).repeat(x.shape[0])

        h = (
            m_e[idx_0, idx_1, idx_2].reshape(x.shape[0], latent_l)
            + torch.exp(-0.5 * logp_e[idx_0, idx_1, idx_2]).reshape(
                x.shape[0], latent_l
            )
            * u_e
        )

        # compute logp(h|x)
        uh = torch.exp(0.5 * logp_e) * (h.unsqueeze(1).repeat(1, n_comps, 1) - m_e)
        L_e = torch.logsumexp(loga_e - 0.5 * uh**2 + 0.5 * logp_e, dim=1)
        L_e = -0.5 * self.latent_l * torch.log(
            torch.tensor(2 * torch.pi, device=self.device)
        ) + torch.sum(L_e, axis=1)

        # compute logq(h)
        L_q = -0.5 * self.latent_l * torch.log(
            torch.tensor(2 * torch.pi, device=self.device)
        ) - 0.5 * torch.sum(h**2, axis=1)

        # compute logp(x|h)
        if update_grid:
            self.decoder.update_grid(h)
        pred_d = self.decoder(h)
        m_d = pred_d[:, 0 : data_l * n_comps].reshape(-1, n_comps, data_l)  # mean
        logp_d = pred_d[:, data_l * n_comps : data_l * n_comps * 2].reshape(
            -1, n_comps, data_l
        )  # standard deviation

        loga_d = pred_d[:, data_l * n_comps * 2 :].reshape(
            -1, n_comps, data_l
        )  # combination coefficient
        loga_d = loga_d - torch.logsumexp(loga_d, dim=1, keepdim=True)  # normalization

        ux = torch.exp(0.5 * logp_d) * (x.unsqueeze(1).repeat(1, n_comps, 1) - m_d)
        L_d = torch.logsumexp(loga_d - 0.5 * ux**2 + 0.5 * logp_d, dim=1)
        L_d = -0.5 * data_l * torch.log(
            torch.tensor(2 * torch.pi, device=self.device)
        ) + torch.sum(L_d, axis=1)

        return -torch.mean(L_d + L_q - L_e)

    def sample(self, num_samples):
        data_l = self.data_l
        latent_l = self.latent_l
        n_comps = self.n_comps

        # sample from prior q(h)
        h = torch.randn(num_samples, latent_l, device=self.device)

        # decode
        pred_d = self.decoder(h)

        m_d = pred_d[:, 0 : data_l * n_comps].reshape(-1, n_comps, data_l)  # mean
        logp_d = pred_d[:, data_l * n_comps : data_l * n_comps * 2].reshape(
            -1, n_comps, data_l
        )  # standard deviation

        loga_d = pred_d[:, data_l * n_comps * 2 :].reshape(
            -1, n_comps, data_l
        )  # combination coefficient
        loga_d -= torch.logsumexp(loga_d, dim=1, keepdim=True)  # normalization

        idx = self.discrete_sample(torch.exp(loga_d))
        idx_0 = torch.arange(num_samples).unsqueeze(1).repeat(1, data_l).flatten()
        idx_1 = idx.flatten()
        idx_2 = torch.arange(data_l).repeat(num_samples)

        u_d = torch.randn(num_samples, data_l, device=self.device)
        x_d = (
            m_d[idx_0, idx_1, idx_2].reshape(num_samples, data_l)
            + torch.exp(-0.5 * logp_d[idx_0, idx_1, idx_2]).reshape(num_samples, data_l)
            * u_d
        )

        return x_d
