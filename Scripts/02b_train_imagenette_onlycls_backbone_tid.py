import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import argparse
from absl import flags
from absl.flags import FLAGS

from typing import Any, Callable, Sequence, Union
import numpy as np

import tensorflow as tf

tf.config.set_visible_devices([], device_type="GPU")
import tensorflow_datasets as tfds

import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze, FrozenDict
from flax import linen as nn
from flax import struct
from flax.training import train_state
from flax.training import orbax_utils

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

from dn_exploration.model import ModelCls as PerceptNet
from dn_exploration.initialization import init_dn_gamma, init_cs, init_dn_cs, init_v1, init_dn_v1
from config_imagenette import config
from dn_exploration.utils import save_state, load_state

@struct.dataclass
class Metrics(metrics.Collection):
    """Collection of metrics to be tracked during training."""

    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Accuracy

class TrainState(train_state.TrainState):
    metrics: Metrics
    state: FrozenDict

def create_train_state(module, key, tx, input_shape):
    """Creates the initial `TrainState`."""
    variables = module.init(key, jnp.ones(input_shape))
    state, params = pop(variables, "params")
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        state=state,
        tx=tx,
        metrics=Metrics.empty(),
    )


# _CONFIG = config_flags.DEFINE_config_file("config")
# flags.FLAGS(sys.argv)
# config = _CONFIG.value
# %%


## Download TID weights
id = "62c3kfxj"
api = wandb.Api()
run = api.run(f"Jorgvt/PNet-Tubin/{id}")
for file in run.files():
    if ".msgpack" in file.name:
        path = file.download(replace=True)
state_ = load_state(path.name)
params_pnet = state_["params"]
state_pnet = state_["state"]

# %%
# %%
wandb.init(
    project="PNet-Tubin",
    name="BioInit-TrainCls",
    job_type="training-cls",
    config=dict(config),
    mode="disabled",
)
config = wandb.config
print(config)

## Load the dataset
dst = tfds.load("imagenette/320px-v2")
dst_train = dst["train"]
dst_val = dst["validation"]
print(f'Train: {len(dst["train"])} | Validation: {len(dst["validation"])}')

def preprocess(sample):
    img, label = sample.values()
    img = tf.image.resize(img, size=(256,256))
    return img/255., label

dst_train_rdy = dst_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(buffer_size=512, reshuffle_each_iteration=True).batch(64, num_parallel_calls=tf.data.AUTOTUNE).prefetch(1)
dst_val_rdy = dst_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(64, num_parallel_calls=tf.data.AUTOTUNE).prefetch(1)
print(f"Train (B): {len(dst_train_rdy)} | Validation (B): {len(dst_val_rdy)}")

if hasattr(config, "LEARNING_RATE"):
    tx = optax.adam(config.LEARNING_RATE)
else:
    tx = optax.adam(config.PEAK_LR)
state = create_train_state(
    PerceptNet(config), random.PRNGKey(config.SEED), tx, input_shape=(1, 256, 256, 3)
)
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))
state = state.replace(params=clip_param(state.params, "A", a_min=0))

# %%
pred, _ = state.apply_fn(
    {"params": state.params, **state.state},
    jnp.ones(shape=(1, 256, 256, 3)),
    train=True,
    mutable=list(state.state.keys()),
)
state = state.replace(state=_)

freeze_mask = flax.traverse_util.path_aware_map(lambda path, x: True if "perceptnet" in path else False, state.params)
tx = optax.chain(
    optax.adam(learning_rate=config.LEARNING_RATE), optax.transforms.freeze(freeze_mask)
)

# %%
state = create_train_state(
    PerceptNet(config), random.PRNGKey(config.SEED), tx, input_shape=(1, 256, 256, 3)
)
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))

# %%
param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
trainable_params = sum(x.size for x, mask in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(freeze_mask)) if mask)
print(f"Total params: {param_count}")
print(f"Trainable params: {trainable_params}")

wandb.run.summary["total_parameters"] = param_count
wandb.run.summary["trainable_parameters"] = trainable_params

params, state_ = state.params.copy(), state.state.copy()
print(f"STATE: {state_.keys()}")

params = unfreeze(params)
state_ = unfreeze(state_)

print(f"STATE PNET: {state_pnet.keys()}")
params["perceptnet"] = params_pnet
state_["batch_stats"]["perceptnet"] = state_pnet["batch_stats"]
state_["precalc_filter"]["perceptnet"] = state_pnet["precalc_filter"]
# params = freeze(params)
# state_ = freeze(state_)

state = state.replace(params=params,
                      state=state_)

## Recalculate parametric filters
pred, _ = state.apply_fn(
    {"params": state.params, **state.state},
    jnp.ones(shape=(1, 256, 256, 3)),
    train=True,
    mutable=list(state.state.keys()),
)
state = state.replace(state=_)
print(f"PRED: {pred.shape}")

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
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
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
    img, label = batch

    def loss_fn(params):
        pred, updated_state = state.apply_fn({"params": params, **state.state}, img, train=True, mutable=list(state.state.keys()))
        return optax.losses.softmax_cross_entropy_with_integer_labels(pred, label).mean(), (pred, updated_state)

    (loss, (logits, updated_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics_updated = state.metrics.single_from_model_output(loss=loss,
                                                             logits=logits,
                                                             labels=label)
    metrics = state.metrics.merge(metrics_updated)
    state = state.replace(metrics=metrics,
                          state=updated_state)
    if return_grads:
        return state, grads
    else:
        return state

@jax.jit
def compute_metrics(*, state, batch):
    """Obtaining the metrics for a given batch."""
    img, label = batch

    def loss_fn(params):
        pred = state.apply_fn({"params": params, **state.state}, img, train=False)
        return optax.losses.softmax_cross_entropy_with_integer_labels(pred, label).mean(), pred

    loss, logits = loss_fn(state.params)
    metrics_updates = state.metrics.single_from_model_output(loss=loss,
                                                             logits=logits,
                                                             labels=label)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
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

    # wandb.log(
        # {f"{k}": wandb.Histogram(v) for k, v in flatten_params(state.params).items()},
        # commit=False,
    # )
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
        f'Epoch {epoch} -> [Train] Loss: {metrics_history["train_loss"][-1]} | Accuracy: {metrics_history["train_accuracy"][-1]} [Val] Loss: {metrics_history["val_loss"][-1]} | Accuracy: {metrics_history["val_accuracy"][-1]}'
    )
    # break

orbax_checkpointer.save(
    os.path.join(wandb.run.dir, "model-final"), state, save_args=save_args
)

wandb.finish()
