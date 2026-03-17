#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tqdm.auto import tqdm

import numpy as np
import jax
from jax import random, numpy as jnp
import flax.linen as nn
from flax.core import pop
import matplotlib.pyplot as plt
from safetensors.numpy import load_file
from iqadatasets.datasets import TID2008, TID2013

from dn_exploration.model import Model
from dn_exploration.utils import load_state


key = random.PRNGKey(42)
x = jnp.ones((1,128,128,3))
model = Model()
variables = model.init(key, x)
state, params = pop(variables, "params")
_, state = model.apply({"params": params, **state}, x, train=True, mutable=list(state.keys()))


# ## Load trained

path_pretrained = "../Scripts/params_train_all_1.msgpack"
params = load_state(path_pretrained)
params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)


_, state = model.apply({"params": params, **state}, x, train=True, mutable=list(state.keys()))


dst = TID2008("/media/disk/vista/BBDD_video_image/Image_Quality/TID/TID2008")


dst_rdy = dst.dataset.batch(64).prefetch(1)


@jax.jit
def calculate_distance(ref, dist):
    pred_ref = model.apply({"params": params, **state}, ref, train=False)
    pred_dist = model.apply({"params": params, **state}, dist, train=False)
    dist = ((pred_ref-pred_dist)**2).mean(axis=(1,2,3))**(1/2)
    return dist


preds = []
moses = []
for batch in tqdm(dst_rdy.as_numpy_iterator(), total=len(dst_rdy)):
    ref, dist, mos = batch
    dist = calculate_distance(ref, dist)
    preds.extend(dist)
    moses.extend(mos)
    # break


from scipy.stats import pearsonr


pearsonr(preds, moses)


dst = TID2013("/media/disk/vista/BBDD_video_image/Image_Quality/TID/TID2013")


dst_rdy = dst.dataset.batch(64).prefetch(1)


preds = []
moses = []
for batch in tqdm(dst_rdy.as_numpy_iterator(), total=len(dst_rdy)):
    ref, dist, mos = batch
    dist = calculate_distance(ref, dist)
    preds.extend(dist)
    moses.extend(mos)
    # break


pearsonr(preds, moses)

