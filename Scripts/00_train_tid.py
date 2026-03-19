import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import argparse
from absl import flags
from absl.flags import FLAGS

from typing import Any, Callable, Sequence, Union
import numpy as np


import jax
from jax import lax, random, numpy as jnp
print(jax.devices())
import flax
from flax.core import freeze, unfreeze, FrozenDict
from flax import linen as nn
from flax import struct
from flax.training import train_state
from flax.training import orbax_utils

import tensorflow as tf
tf.config.set_visible_devices([], device_type="GPU")

import optax
import orbax.checkpoint

from clu import metrics
from ml_collections import ConfigDict, config_flags

import wandb
from iqadatasets.datasets import *
from JaxPlayground.utils.wandb import *
from paramperceptnet.constraints import *
from paramperceptnet.training import *
from paramperceptnet.configs import param_config as config

from dn_exploration.model import Model as PerceptNet
from dn_exploration.initialization import init_dn_gamma, init_cs, init_dn_cs, init_v1, init_dn_v1
from config import config
from dn_exploration.utils import save_state

# _CONFIG = config_flags.DEFINE_config_file("config")
# flags.FLAGS(sys.argv)
# config = _CONFIG.value
print(config)
# %%
# dst_train = TID2008("/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2008/", exclude_imgs=[25])
# dst_train = KADIK10K("/lustre/ific.uv.es/ml/uv075/Databases/IQA/KADIK10K/")
# dst_val = TID2013( "/lustre/ific.uv.es/ml/uv075/Databases/IQA//TID/TID2013/", exclude_imgs=[25])
dst_train = TID2008("/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2008/", exclude_imgs=[25])
dst_val = TID2013("/media/disk/vista/BBDD_video_image/Image_Quality//TID/TID2013/", exclude_imgs=[25])
# dst_train = TID2008("/media/databases/IQA/TID/TID2008/", exclude_imgs=[25])
# dst_val = TID2013("/media/databases/IQA/TID/TID2013/", exclude_imgs=[25])

# %%
img, img_dist, mos = next(iter(dst_train.dataset))
img.shape, img_dist.shape, mos.shape

# %%
img, img_dist, mos = next(iter(dst_val.dataset))
img.shape, img_dist.shape, mos.shape

# %%
wandb.init(
    project="PNet-Tubin",
    name="BioInit-TrainCorr",
    job_type="training",
    config=config,
    mode="online",
)
config = config
config

# %%
dst_train_rdy = dst_train.dataset.shuffle(
    buffer_size=100, reshuffle_each_iteration=True, seed=config.SEED
).batch(config.BATCH_SIZE, drop_remainder=True)
dst_val_rdy = dst_val.dataset.batch(config.BATCH_SIZE, drop_remainder=True)

if hasattr(config, "LEARNING_RATE"):
    tx = optax.adam(config.LEARNING_RATE)
else:
    tx = optax.adam(config.PEAK_LR)
state = create_train_state(
    PerceptNet(), random.PRNGKey(config.SEED), tx, input_shape=(1, 384, 512, 3)
)
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
state = state.replace(params=clip_param(state.params, "A", a_min=0))

# %%
state.params.keys()

# %%
pred, _ = state.apply_fn(
    {"params": state.params, **state.state},
    jnp.ones(shape=(1, 384, 512, 3)),
    train=True,
    mutable=list(state.state.keys()),
)
state = state.replace(state=_)

tx = optax.adam(learning_rate=config.LEARNING_RATE)

# %%
state = create_train_state(
    PerceptNet(), random.PRNGKey(config.SEED), tx, input_shape=(1, 384, 512, 3)
)
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))

# %%
param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
print(param_count)

wandb.run.summary["total_parameters"] = param_count

## Initialization
params, state_ = state.params.copy(), state.state.copy()

params = unfreeze(params)
state_ = unfreeze(state_)

params = init_dn_gamma(params)
params["CenterSurroundLogSigmaK_0"] = init_cs(params["CenterSurroundLogSigmaK_0"])
params, state_ = init_dn_cs(params, state_)
params = init_v1(params)
params, state_ = init_dn_v1(params, state_)

# params = freeze(params)
# state_ = freeze(state_)

state = state.replace(params=params,
                      state=state_)

## Recalculate parametric filters
pred, _ = state.apply_fn(
    {"params": state.params, **state.state},
    jnp.ones(shape=(1, 384, 512, 3)),
    train=True,
    mutable=list(state.state.keys()),
)
state = state.replace(state=_)

# %%
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(state)

# %%
orbax_checkpointer.save(
    os.path.join(wandb.run.dir, "model-0"), state, save_args=save_args, force=True
)  # force=True means allow overwritting.
save_state(state, os.path.join(wandb.run.dir, "model-0"))
wandb.save(os.path.join(wandb.run.dir, "model-0"))

# %%
metrics_history = {
    "train_loss": [],
    "val_loss": [],
}

# %%
batch = next(iter(dst_train_rdy.as_numpy_iterator()))

# %%
from functools import partial


# %%
def forward(state, inputs):
    return state.apply_fn({"params": state.params, **state.state}, inputs, train=False)


@partial(jax.jit, static_argnums=2)
def train_step(state, batch, return_grads=False):
    img, img_dist, mos = batch

    def loss_fn(params):
        pred, updated_state = state.apply_fn({"params": params, **state.state}, img, train=True, mutable=list(state.state.keys()))
        pred_dist, updated_state = state.apply_fn({"params": params, **state.state}, img_dist, train=True, mutable=list(state.state.keys()))
        dist = ((pred-pred_dist)**2).mean(axis=(1,2,3))**(1/2)
        return pearson_correlation(dist, mos), updated_state

    (loss, updated_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics_updated = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metrics_updated)
    state = state.replace(metrics=metrics,
                          state=updated_state)
    if return_grads:
        return state, grads
    else:
        return state


# %%
s1, grads = train_step(state, batch, return_grads=True)
print("TRAINED STEPPED ONCE")

# %%
# jax.config.update("jax_debug_nans", True)

# %%
step = 0
for epoch in range(config.EPOCHS):
    ## Training
    for batch in dst_train_rdy.as_numpy_iterator():
        state, grads = train_step(state, batch, return_grads=True)
        state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
        state = state.replace(params=clip_param(state.params, "A", a_min=0))
        state = state.replace(params=clip_param(state.params, "K", a_min=1 + 1e-5))
        state = state.replace(params=unfreeze(state.params))
        wandb.log(
            {f"{k}_grad": wandb.Histogram(v) for k, v in flatten_params(grads).items()},
            commit=False,
        )
        step += 1
        # state = compute_metrics(state=state, batch=batch)
        # break

    ## Log the metrics
    for name, value in state.metrics.compute().items():
        metrics_history[f"train_{name}"].append(value)

    ## Empty the metrics
    state = state.replace(metrics=state.metrics.empty())

    ## Evaluation
    for batch in dst_val_rdy.as_numpy_iterator():
        state = compute_metrics(state=state, batch=batch)
        # break
    for name, value in state.metrics.compute().items():
        metrics_history[f"val_{name}"].append(value)
    state = state.replace(metrics=state.metrics.empty())

    ## Checkpointing
    if metrics_history["val_loss"][-1] <= min(metrics_history["val_loss"]):
        orbax_checkpointer.save(
            save_path:=os.path.join(wandb.run.dir, "model-best"),
            state,
            save_args=save_args,
            force=True,
        )  # force=True means allow overwritting.
    # orbax_checkpointer.save(os.path.join(wandb.run.dir, f"model-{epoch+1}"), state, save_args=save_args, force=False) # force=True means allow overwritting.
        save_state(state, save_path)
        wandb.save(os.path.join(wandb.run.dir, "model-best.msgpack"))

    wandb.log(
        {f"{k}": wandb.Histogram(v) for k, v in flatten_params(state.params).items()},
        commit=False,
    )
    if hasattr(config, "LEARNING_RATE"):
        wandb.log(
            {
                "epoch": epoch + 1,
                "learning_rate": config.LEARNING_RATE,
                **{name: values[-1] for name, values in metrics_history.items()},
            }
        )
    else:
        wandb.log(
            {
                "epoch": epoch + 1,
                "learning_rate": schedule_lr(step),
                **{name: values[-1] for name, values in metrics_history.items()},
            }
        )
    print(
        f'Epoch {epoch} -> [Train] Loss: {metrics_history["train_loss"][-1]} [Val] Loss: {metrics_history["val_loss"][-1]}'
    )
    # break

orbax_checkpointer.save(
    os.path.join(wandb.run.dir, "model-final"), state, save_args=save_args
)

wandb.finish()
