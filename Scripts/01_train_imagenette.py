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
from dn_exploration.utils import save_state

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


# %%

# %%
# %%
wandb.init(
    project="PNet-Tubin",
    name="BioInit-TrainCls",
    job_type="training-cls",
    config=dict(config),
    mode="online",
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

tx = optax.adam(learning_rate=config.LEARNING_RATE)

# %%
state = create_train_state(
    PerceptNet(config), random.PRNGKey(config.SEED), tx, input_shape=(1, 256, 256, 3)
)
state = state.replace(params=clip_layer(state.params, "GDN", a_min=0))

# %%
param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
print(param_count)

wandb.run.summary["total_parameters"] = param_count

## Initialization
def init_dn_gamma(params):
    params_ = params.copy()
    params_["perceptnet"]["GDNGamma_0"]["bias"] = jnp.ones_like(params_["perceptnet"]["GDNGamma_0"]["bias"]) * 0.1
    params_["perceptnet"]["GDNGamma_0"]["kernel"] = (
        jnp.ones_like(params_["perceptnet"]["GDNGamma_0"]["kernel"]) * 0.5
    )
    # params_["GDNGamma_0"]["bias"] = jnp.ones_like(params_["GDNGamma_0"]["bias"]) * (-0.04)
    # params_["GDNGamma_0"]["kernel"] = (
    #     jnp.ones_like(params_["GDNGamma_0"]["kernel"]) * 0.216
    # )
    return params

def init_dn_cs(params, state):
    """
    K = cs_q
    a_star = cs_q
    use_noise = False
    sigma = 0.1
    mean_lh = False
    b = (cs_q**2)/10
    """
    params_ = params.copy()
    state_ = state.copy()
    a_star_cs = jnp.load("a_star_gdn_cs.npy")
    params_["perceptnet"]["DN_0"]["GDNGaussian_0"]["GaussianLayerGamma_0"]["gamma"] = (1/(0.1/jnp.sqrt(2)))*jnp.ones_like(params["perceptnet"]["DN_0"]["GDNGaussian_0"]["GaussianLayerGamma_0"]["gamma"])
    params_["perceptnet"]["DN_0"]["GDNGaussian_0"]["GaussianLayerGamma_0"]["bias"] = (a_star_cs.squeeze()**2)/10
    state_["batch_stats"]["perceptnet"]["DN_0"]["K"] = a_star_cs
    state_["batch_stats"]["perceptnet"]["DN_0"]["inputs_star"] = a_star_cs
    return params, state

def init_v1(params):
    params_ = params.copy()
    params_["perceptnet"]["GaborLayerGammaHumanLike__0"]["freq_a"] = jnp.array([2.0, 4., 8., 16.])
    params_["perceptnet"]["GaborLayerGammaHumanLike__0"]["freq_t"] = jnp.array([2., 4.])
    params_["perceptnet"]["GaborLayerGammaHumanLike__0"]["freq_d"] = jnp.array([2., 4.])

    A_a = jnp.zeros(shape=(3, 64), dtype=jnp.float32)
    A_a = A_a.at[0, :].set(1.0)
    A_t = jnp.zeros(shape=(3, 33), dtype=jnp.float32) # Add 1 to account for the f=0
    A_t = A_t.at[1, :].set(1.0)
    A_t = A_t.at[1,0].set(5.0)
    A_t = A_t/(2*1.2)
    A_d = jnp.zeros(shape=(3, 33), dtype=jnp.float32) # Add 1 to account for the f=0
    A_d = A_d.at[2, :].set(1.0)
    A_d = A_d.at[2,0].set(4.0)
    A_d = A_d/(1.5*1.2)
    params_["perceptnet"]["GaborLayerGammaHumanLike__0"]["A"] = jnp.concatenate(
        [A_a, A_t, A_d], axis=-1
    )
    params_["perceptnet"]["GaborLayerGammaHumanLike__0"]["gammax_a"] = 1/jnp.array([0.16, 0.08, 0.06, 0.04])
    params_["perceptnet"]["GaborLayerGammaHumanLike__0"]["gammay_a"] = 1/jnp.array([0.16, 0.08, 0.06, 0.04])
    return params_

def init_dn_v1(params, state):
    params_ = params.copy()
    state_ = state.copy()

    params_["perceptnet"]["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["H_cc"] = jnp.eye(3,3)


    params_["perceptnet"]["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_f_a"] = 1/(0.25*jnp.array([1., 1., 1., 1.])/jnp.sqrt(2))
    params_["perceptnet"]["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_f_t"] = 1/(0.25*jnp.array([1., 1.,])/jnp.sqrt(2))
    params_["perceptnet"]["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_f_d"] = 1/(0.25*jnp.array([1., 1.,])/jnp.sqrt(2))

    params_["perceptnet"]["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_a"] = 1/(jnp.pi/180*jnp.array([15.]*8)/jnp.sqrt(2))
    params_["perceptnet"]["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_t"] = 1/(jnp.pi/180*jnp.array([15.]*8)/jnp.sqrt(2))
    params_["perceptnet"]["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["ChromaFreqOrientGaussianGamma_0"]["gamma_theta_d"] = 1/(jnp.pi/180*jnp.array([15.]*8)/jnp.sqrt(2))


    params_["perceptnet"]["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["GaussianLayerGamma_0"]["gamma"] = jnp.concatenate([
        jnp.tile(jnp.tile((1/2)*params_["perceptnet"]["GaborLayerGammaHumanLike__0"]["gammax_a"], reps=8), reps=2),
        jnp.tile(jnp.tile((1/2)*params_["perceptnet"]["GaborLayerGammaHumanLike__0"]["gammax_t"], reps=8), reps=2),
        jnp.tile(jnp.tile((1/2)*params_["perceptnet"]["GaborLayerGammaHumanLike__0"]["gammax_d"], reps=8), reps=2),
    ])
    params_["perceptnet"]["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["GaussianLayerGamma_0"]["gamma"] = jnp.insert(params_["perceptnet"]["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["GaussianLayerGamma_0"]["gamma"],
                                                                                                         jnp.array([64, 64+32]),
                                                                                                         jnp.array([
                                                                                                            params_["perceptnet"]["GaborLayerGammaHumanLike__0"]["gammax_t"][0],
                                                                                                            params_["perceptnet"]["GaborLayerGammaHumanLike__0"]["gammax_d"][0],
                                                                                                         ]))

    inputs_star = jnp.load("a_star_gdn_v1.npy")
    params_["perceptnet"]["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["bias"] = (inputs_star.squeeze()**2)/1000
    # params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["bias"] = params_["GDNControl_0"]["GDNSpatioChromaFreqOrient_0"]["bias"]**(1/3)

    state_["batch_stats"]["perceptnet"]["GDNControl_0"]["inputs_star"] = inputs_star

    state_["batch_stats"]["perceptnet"]["GDNControl_0"]["K"] = inputs_star
    coef = 2
    Wr = jnp.concatenate([
        jnp.tile(
            jnp.tile(
                jnp.array([1/(coef**3), 1/(coef**2), 1/coef, 1]),
                reps=8
            )
            , reps=2,
        ),
        jnp.tile(
            jnp.tile(
                jnp.array([1/(coef**3), 1/(coef**2)]),
                reps=8
            )
            , reps=2,
        ),
        jnp.tile(
            jnp.tile(
                jnp.array([1/(coef**3), 1/(coef**2)]),
                reps=8
            )
            , reps=2,
        )
    ])
    Wr = jnp.insert(Wr, jnp.array([64, 64+32]), jnp.array([1/coef**3, 1/coef**3]))
    state_["batch_stats"]["perceptnet"]["GDNControl_0"]["K"] = state_["batch_stats"]["perceptnet"]["GDNControl_0"]["K"]*Wr
    return params, state

params, state_ = state.params.copy(), state.state.copy()
print(f"STATE: {state_.keys()}")

params = unfreeze(params)
state_ = unfreeze(state_)

params = init_dn_gamma(params)
params["perceptnet"]["CenterSurroundLogSigmaK_0"] = init_cs(params["perceptnet"]["CenterSurroundLogSigmaK_0"])
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
        f'Epoch {epoch} -> [Train] Loss: {metrics_history["train_loss"][-1]} | Accuracy: {metrics_history["train_accuracy"][-1]} [Val] Loss: {metrics_history["val_loss"][-1]} | Accuracy: {metrics_history["val_accuracy"][-1]}'
    )
    # break

orbax_checkpointer.save(
    os.path.join(wandb.run.dir, "model-final"), state, save_args=save_args
)

wandb.finish()
