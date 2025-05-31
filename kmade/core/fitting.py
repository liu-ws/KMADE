import torch
import numpy as np
import matplotlib.pyplot as plt
from kan import LBFGS
from tqdm import tqdm
import os


def modified_fit(
    self,
    dataset,
    opt="LBFGS",
    steps=100,
    log=1,
    lamb=0.0,
    lamb_l1=1.0,
    lamb_entropy=2.0,
    lamb_coef=0.0,
    lamb_coefdiff=0.0,
    update_grid=True,
    grid_update_num=10,
    loss_fn=None,
    lr=1.0,
    start_grid_update_step=-1,
    stop_grid_update_step=50,
    batch=-1,
    metrics=None,
    save_fig=False,
    in_vars=None,
    out_vars=None,
    beta=3,
    save_fig_freq=1,
    img_folder="./video",
    singularity_avoiding=False,
    y_th=1000.0,
    reg_metric="edge_forward_spline_n",
    display_metrics=None,
):
    """
    training

    Args:
    -----
        dataset : dic
            contains dataset['train_data'],dataset['test_data']
        opt : str
            "LBFGS" or "Adam"
        steps : int
            training steps
        log : int
            logging frequency
        lamb : float
            overall penalty strength
        lamb_l1 : float
            l1 penalty strength
        lamb_entropy : float
            entropy penalty strength
        lamb_coef : float
            coefficient magnitude penalty strength
        lamb_coefdiff : float
            difference of nearby coefficits (smoothness) penalty strength
        update_grid : bool
            If True, update grid regularly before stop_grid_update_step
        grid_update_num : int
            the number of grid updates before stop_grid_update_step
        start_grid_update_step : int
            no grid updates before this training step
        stop_grid_update_step : int
            no grid updates after this training step
        loss_fn : function
            loss function
        lr : float
            learning rate
        batch : int
            batch size, if -1 then full.
        save_fig_freq : int
            save figure every (save_fig_freq) steps
        singularity_avoiding : bool
            indicate whether to avoid singularity for the symbolic part
        y_th : float
            singularity threshold (anything above the threshold is considered singular and is softened in some ways)
        reg_metric : str
            regularization metric. Choose from {'edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'}
        metrics : a list of metrics (as functions)
            the metrics to be computed in training
        display_metrics : a list of functions
            the metric to be displayed in tqdm progress bar

    Returns:
    --------
        results : dic
            results['train_loss'], 1D array of training losses (RMSE)
            results['test_loss'], 1D array of test losses (RMSE)
            results['reg'], 1D array of regularization
            other metrics specified in metrics
    """
    data_l = self.data_l
    if lamb > 0.0 and not self.save_act:
        print("setting lamb=0. If you want to set lamb > 0, set self.save_act=True")

    old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

    pbar = tqdm(range(steps), desc="description", ncols=100)

    if loss_fn is None:
        loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
    else:
        loss_fn = loss_fn_eval = loss_fn

    grid_update_freq = int(stop_grid_update_step / grid_update_num)

    if opt == "Adam":
        optimizer = torch.optim.Adam(self.get_params(), lr=lr)
    elif opt == "LBFGS":
        optimizer = LBFGS(
            self.get_params(),
            lr=lr,
            history_size=10,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-32,
            tolerance_change=1e-32,
            tolerance_ys=1e-32,
        )

    results = {}
    results["train_loss"] = []
    results["test_loss"] = []
    results["reg"] = []
    if metrics is not None:
        for i in range(len(metrics)):
            results[metrics[i].__name__] = []

    if batch == -1 or batch > dataset["train_data"].shape[0]:
        batch_size = dataset["train_data"].shape[0]
        batch_size_test = dataset["test_data"].shape[0]
    else:
        batch_size = batch
        batch_size_test = batch

    global train_loss, reg_

    def closure():
        global train_loss, reg_
        optimizer.zero_grad()
        pred = self.forward(
            dataset["train_data"][train_id],
            singularity_avoiding=singularity_avoiding,
            y_th=y_th,
        )
        train_loss = loss_fn(pred, dataset["train_data"][train_id, :data_l])
        if self.save_act:
            if reg_metric == "edge_backward":
                self.attribute()
            if reg_metric == "node_backward":
                self.node_attribute()
            reg_ = self.get_reg(
                reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
            )
        else:
            reg_ = torch.tensor(0.0)
        objective = train_loss + lamb * reg_
        objective.backward()
        return objective

    if save_fig:
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

    for _ in pbar:

        if _ == steps - 1 and old_save_act:
            self.save_act = True

        if save_fig and _ % save_fig_freq == 0:
            save_act = self.save_act
            self.save_act = True

        train_id = np.random.choice(
            dataset["train_data"].shape[0], batch_size, replace=False
        )
        test_id = np.random.choice(
            dataset["test_data"].shape[0], batch_size_test, replace=False
        )

        if (
            _ % grid_update_freq == 0
            and _ < stop_grid_update_step
            and update_grid
            and _ >= start_grid_update_step
        ):
            self.update_grid(dataset["train_data"][train_id])

        if opt == "LBFGS":
            optimizer.step(closure)

        if opt == "Adam":
            pred = self.forward(
                dataset["train_data"][train_id],
                singularity_avoiding=singularity_avoiding,
                y_th=y_th,
            )
            train_loss = loss_fn(pred, dataset["train_data"][train_id, :data_l])
            if self.save_act:
                if reg_metric == "edge_backward":
                    self.attribute()
                if reg_metric == "node_backward":
                    self.node_attribute()
                reg_ = self.get_reg(
                    reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff
                )
            else:
                reg_ = torch.tensor(0.0)
            loss = train_loss + lamb * reg_
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_loss = loss_fn_eval(
            self.forward(dataset["test_data"][test_id]),
            dataset["test_data"][test_id, :data_l],
        )

        if metrics is not None:
            for i in range(len(metrics)):
                results[metrics[i].__name__].append(metrics[i]().item())

        results["train_loss"].append(train_loss.cpu().detach().numpy())
        results["test_loss"].append(test_loss.cpu().detach().numpy())
        results["reg"].append(reg_.cpu().detach().numpy())

        if _ % log == 0:
            if display_metrics is None:
                pbar.set_description(
                    "| train_loss: %.2e | test_loss: %.2e | reg: %.2e | "
                    % (
                        train_loss.cpu().detach().numpy(),
                        test_loss.cpu().detach().numpy(),
                        reg_.cpu().detach().numpy(),
                    )
                )
            else:
                string = ""
                data = ()
                for metric in display_metrics:
                    string += f" {metric}: %.2e |"
                    try:
                        results[metric]
                    except Exception:
                        raise Exception(f"{metric} not recognized")
                    data += (results[metric][-1],)
                pbar.set_description(string % data)

        if save_fig and _ % save_fig_freq == 0:
            self.plot(
                folder=img_folder,
                in_vars=in_vars,
                out_vars=out_vars,
                title="Step {}".format(_),
                beta=beta,
            )
            plt.savefig(
                img_folder + "/" + str(_) + ".jpg", bbox_inches="tight", dpi=200
            )
            plt.close()
            self.save_act = save_act

    self.log_history("fit")
    # revert back to original state
    self.symbolic_enabled = old_symbolic_enabled
    return results
