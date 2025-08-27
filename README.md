# KMADE

This repository provides several density estimation methods based on the Kolmogorov-Arnold Network (KAN), capable of fitting symbolic expressions for probability density functions using samples from corresponding distributions. We hope to explore efficient GW catalog construction in future GW observations through these density estimation methods. Implemented models include Masked Autoencoder Density Estimator (MADE), Masked Autoregressive Flow (MAF), and Variational Autoencoder (VAE). A detailed description of the KAN-MADE model is available in [our paper](https://arxiv.org/abs/2508.18698).

---

## Requirements

```bash
conda create -n kmade python=3.10
conda activate kmade
pip install -r requirements.txt
```

---

## Quick Start

Apply our method to the GW150914 event posterior data and reproduce the results in our paper with the following commands:

```bash
# Obtain data
python -m kmade.cli.data

# Train neural network and perform symbolification
python -m kmade.cli.train

# Resample from neural network and symbolic expression
python -m kmade.cli.resampling
```

Results will be saved in the `outputs/` directory.

---

## Public Data

The data used to generate the figures in [our paper](https://arxiv.org/abs/2508.18698) can be found in the `public_data/` directory.


## Reference

If you use this code, please cite:

```bibtex
@misc{liu2025light,
      title={Lightweight posterior construction for gravitational-wave catalogs with the Kolmogorov-Arnold network}, 
      author={Wenshuai Liu and Yiming Dong and Ziming Wang and Lijing Shao},
      year={2025},
      eprint={2508.18698},
      archivePrefix={arXiv},
      primaryClass={gr-qc},
      url={https://arxiv.org/abs/2508.18698}, 
}
```

---

For questions or feedback, please open an issue or contact the authors.