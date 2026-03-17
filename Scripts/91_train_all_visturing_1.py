#!/usr/bin/env python
# coding: utf-8

import os
from tqdm.auto import tqdm

import numpy as np
import jax
from jax import random, numpy as jnp
import flax.linen as nn
from flax.core import pop
import matplotlib.pyplot as plt
from safetensors.numpy import load_file

from dn_exploration.model import Model
from dn_exploration.initialization import init_dn_gamma, init_cs, init_dn_cs, init_v1, init_dn_v1
from dn_exploration.utils import save_state


path = "/media/disk/vista/Papers/A_Perceptnet"


path_a = "/media/disk/vista/Papers/A_back_to_vista/A_modelo_parametrico"
path_visturing = "/media/disk/users/vitojor/visturing/Data"


key = random.PRNGKey(42)
x = jnp.ones((1,128,128,3))
model = Model()
variables = model.init(key, x)
state, params = pop(variables, "params")
_, state = model.apply({"params": params, **state}, x, train=True, mutable=list(state.keys()))


# ## Inits

params = init_dn_gamma(params)
params["CenterSurroundLogSigmaK_0"] = init_cs(params["CenterSurroundLogSigmaK_0"])
params, state = init_dn_cs(params, state)
params = init_v1(params)
params, state = init_dn_v1(params, state)


_, state = model.apply({"params": params, **state}, x, train=True, mutable=list(state.keys()))


# ## Visturing

from visturing.properties.jax import prop1 as prop
# from visturing.properties.jax import prop3_4 as prop
# from visturing.properties.jax import prop2 as prop


def train_step(params):
    def loss_fn(params):
        def calculate_diffs(a, b):
            a = a/255.
            b = b/255.
            pred_a, _ = model.apply({"params": params, **state}, a, train=True, mutable=list(state.keys()))
            pred_b, _ = model.apply({"params": params, **state}, b, train=True, mutable=list(state.keys()))
            return ((pred_a-pred_b)**2).mean(axis=(1,2,3))**(1/2)
        results = prop.evaluate(
            calculate_diffs,
            data_path=os.path.join(path_visturing, "Experiment_1"),
            gt_path=os.path.join(path_visturing, "ground_truth"),
        )
        return -results["correlations"]["pearson"]
        # return -results["correlations"]["pearson_achrom"]
        # return -results["correlations"]["kendall"]["achrom"]["kendall"]
        # return -results["correlations"]["kendall"]["kendall"]
    loss, grad = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda x,g: x-lr*g, params, grad)
    # Train only the first layer
    # params["GDNGamma_0"] = jax.tree_util.tree_map(lambda x,g: x-lr*g, params["GDNGamma_0"], grad["GDNGamma_0"])

    return params, loss, grad


# from jax import config
# config.update("jax_debug_nans", True)


lr = 0.01
epochs = 100
losses = []
for epoch in range(epochs):
    params, loss, grad = train_step(params)
    losses.append(loss)
    print(loss)
    # break

## Save the trained model
save_state(params, "params_train_all_1")
print("Saved!")

# def calculate_diffs(a, b):
#     a = a/255.
#     b = b/255.
#     pred_a, _ = model.apply({"params": params, **state}, a, train=True, mutable=list(state.keys()))
#     pred_b, _ = model.apply({"params": params, **state}, b, train=True, mutable=list(state.keys()))
#     return ((pred_a-pred_b)**2).mean(axis=(1,2,3))**(1/2)

# results = prop.evaluate(
#     calculate_diffs,
#     data_path=os.path.join(path_visturing, "Experiment_1"),
#     gt_path=os.path.join(path_visturing, "ground_truth"),
# )

# from visturing.properties.utils import evaluate_all, build_evaluation_table
# path_visturing = "/media/disk/users/vitojor/visturing/Data"


# def calculate_diffs(a, b):
#     # a = a/255.
#     # b = b/255.
#     pred_a = model.apply({"params": params, **state}, a, train=False)
#     pred_b = model.apply({"params": params, **state}, b, train=False)
#     return ((pred_a-pred_b)**2).mean(axis=(1,2,3))**(1/2)


# results = evaluate_all(
#     calculate_diffs,
#     data_path=path_visturing,
#     gt_path=os.path.join(path_visturing, "ground_truth"),
# )


# table_results = build_evaluation_table(results)