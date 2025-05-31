import torch
import sympy as sp
from sympy import symbols, ex_round
import yaml
from kan import SYMBOLIC_LIB


# Modify the saving method of KAN for KMADEs
def save_model(model, path, dataset=None, dtype=torch.float32):

    model_name = type(model).__name__

    # if data is inputed, save the data too
    if dataset is None:
        ifsavedata = False
    else:
        ifsavedata = True

    # record the information of KMADEs
    dic = dict(
        grid=model.grid,
        k=model.k,
        mult_arity=model.mult_arity,
        base_fun_name=model.base_fun_name,
        symbolic_enabled=model.symbolic_enabled,
        affine_trainable=model.affine_trainable,
        grid_eps=model.grid_eps,
        grid_range=model.grid_range,
        sp_trainable=model.sp_trainable,
        sb_trainable=model.sb_trainable,
        state_id=model.state_id,
        auto_save=model.auto_save,
        ckpt_path=model.ckpt_path,
        round=model.round,
        device=str(model.device),
        name=model_name,
        n_comps=model.n_comps,
        para_l=model.para_l,
        data_l=model.data_l,
        hidden_layers=model.hidden_layers,
        para_hidden_layers=model.para_hidden_layers,
        seed=model.seed,
        input_order=model.input_order,
        mode=model.mode,
        ifsavedata=ifsavedata,
        ifperturb=model.ifperturb,
        bias_mean=model.bias_mean,
        bias_std=model.bias_std,
    )

    for i in range(model.depth):
        dic[f"symbolic.funs_name.{i}"] = model.symbolic_fun[i].funs_name

    with open(f"{path}_config.yml", "w") as outfile:
        yaml.dump(dic, outfile, default_flow_style=False)

    converted_state = {
        key: value.to(dtype=dtype) for key, value in model.state_dict().items()
    }

    torch.save(converted_state, f"{path}_weights.pth")
    if ifsavedata:
        train_data = dataset["train_input"]
        test_data = dataset["test_input"]
        torch.save({"train_data": train_data, "test_data": test_data}, f"{path}_data")


# Modify the loading method of KAN for KMADEs
def load_model(path, ifloaddata=False, dtype=torch.float32):

    with open(f"{path}_config.yml", "r") as stream:
        config = yaml.safe_load(stream)

    state = torch.load(f"{path}_weights.pth", weights_only=True)
    converted_state = {key: value.to(dtype=dtype) for key, value in state.items()}
    base_args = {
        "grid": config["grid"],
        "k": config["k"],
        "mult_arity": config["mult_arity"],
        "base_fun": config["base_fun_name"],
        "symbolic_enabled": config["symbolic_enabled"],
        "affine_trainable": config["affine_trainable"],
        "grid_eps": config["grid_eps"],
        "grid_range": config["grid_range"],
        "sp_trainable": config["sp_trainable"],
        "sb_trainable": config["sb_trainable"],
        "state_id": config["state_id"],
        "auto_save": config["auto_save"],
        "first_init": False,
        "ckpt_path": config["ckpt_path"],
        "round": config["round"] + 1,
        "device": config["device"],
        "seed": config["seed"],
        "data_l": config["data_l"],
        "hidden_layers": config["hidden_layers"],
        "input_order": config["input_order"],
        "mode": config["mode"],
        "ifperturb": config["ifperturb"],
        "bias_mean": config["bias_mean"],
        "bias_std": config["bias_std"],
    }

    if config["para_l"] != 0:
        base_args["para_l"] = config["para_l"]
    if config["para_hidden_layers"] != []:
        base_args["para_hidden_layers"] = config["para_hidden_layers"]
    if config["n_comps"] != 1:
        base_args["n_comps"] = config["n_comps"]

    from kmade.core import kmades

    model_class = getattr(kmades, config["name"])
    model_load = model_class(**base_args)

    model_load.load_state_dict(converted_state)
    # model_load.cache_data = None

    depth = len(model_load.width) - 1
    for layer_idx in range(depth):
        out_dim = model_load.symbolic_fun[layer_idx].out_dim
        in_dim = model_load.symbolic_fun[layer_idx].in_dim
        funs_name = config[f"symbolic.funs_name.{layer_idx}"]
        for j in range(out_dim):
            for i in range(in_dim):
                fun_name = funs_name[j][i]
                model_load.symbolic_fun[layer_idx].funs_name[j][i] = fun_name
                model_load.symbolic_fun[layer_idx].funs[j][i] = SYMBOLIC_LIB[fun_name][
                    0
                ]
                model_load.symbolic_fun[layer_idx].funs_sympy[j][i] = SYMBOLIC_LIB[
                    fun_name
                ][1]
                model_load.symbolic_fun[layer_idx].funs_avoid_singularity[j][i] = (
                    SYMBOLIC_LIB[fun_name][3]
                )

    if ifloaddata:
        if config["ifsavedata"]:
            train_data = torch.load(f"{path}_data", weights_only=True)["train_data"]
            test_data = torch.load(f"{path}_data", weights_only=True)["test_data"]
            dataset = {
                "train_input": train_data,
                "train_label": train_data[:, : config["data_l"]],
                "test_input": test_data,
                "test_label": test_data[:, : config["data_l"]],
            }

            return model_load, dataset
        else:
            print("there is no data saved!")
            return model_load
    else:
        return model_load


# save and read expressions of the distribution got by model
def save_expr(model, path, tolerance=None):
    # save every output expr
    # tolerance: the tolerance of the rounding

    path = path + "_expr.yml"
    expressions = []
    for e in model.symbolic_formula()[0]:
        if tolerance is None:
            expressions.append(str(sp.sympify(e)))
        else:
            expressions.append(str(ex_round(sp.sympify(e), tolerance)))
    data_l = model.data_l
    para_l = model.para_l
    n_comps = model.n_comps
    input_degrees = model.degrees[0].detach().cpu().numpy().tolist()

    dic = dict(
        expressions=expressions,
        data_l=data_l,
        para_l=para_l,
        n_comps=n_comps,
        input_degrees=input_degrees,
    )
    with open(path, "w") as outfile:
        yaml.dump(dic, outfile, default_flow_style=False)


def read_expr(path, simplify=False):
    # reture the expreession of the probability density function of the distribution got by model
    # load the expressions from the file
    path = path + "_expr.yml"
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)

    expressions = config["expressions"]
    for i in range(len(expressions)):
        expressions[i] = sp.sympify(expressions[i])
    data_l, n_comps = config["data_l"], config["n_comps"]

    vars_data = symbols(f"x_1:{data_l+1}")
    vars_data_matrix = sp.Matrix(n_comps, data_l, lambda i, j: vars_data[j])

    m = expressions[0 : data_l * n_comps]
    logp = expressions[data_l * n_comps : data_l * n_comps * 2]
    # convert to matrix
    m_matrix = sp.Matrix(n_comps, data_l, m)
    logp_matrix = sp.Matrix(n_comps, data_l, logp)

    # compute the likelihood
    if n_comps > 1:
        loga = expressions[data_l * n_comps * 2 :]
        loga_matrix = sp.Matrix(n_comps, data_l, loga)
        loga_matrix -= (
            sp.ones(n_comps, n_comps) * (loga_matrix.applyfunc(sp.exp))
        ).applyfunc(sp.log)
        u = ((0.5 * logp_matrix).applyfunc(sp.exp)).multiply_elementwise(
            vars_data_matrix - m_matrix
        )

        L = (
            sp.ones(1, n_comps)
            * (
                (
                    loga_matrix - 0.5 * u.applyfunc(lambda x: x**2) + 0.5 * logp_matrix
                ).applyfunc(sp.exp)
            )
        ).applyfunc(sp.log)
        L = (
            sp.sympify(-0.5 * data_l * sp.log(2 * sp.pi))
            + (L * sp.ones(data_l, 1))[0, 0]
        )
    else:
        u = ((0.5 * logp_matrix).applyfunc(sp.exp)).multiply_elementwise(
            vars_data_matrix - m_matrix
        )
        L = -0.5 * (
            sp.sympify(data_l * sp.log(2 * sp.pi))
            + ((u.applyfunc(lambda x: x**2) - logp_matrix) * sp.ones(data_l, 1))[0, 0]
        )

    L = sp.exp(L)
    if simplify:
        L = sp.simplify(L)
    return L
