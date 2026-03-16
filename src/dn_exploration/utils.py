import os
from datasets import load_dataset
from PIL import Image
import numpy as np

from perceptualtests.color_matrices import Mng2xyz, Mxyz2atd, gamma
from flax.serialization import to_state_dict
from orbax.checkpoint.msgpack_utils import msgpack_serialize, msgpack_restore

def rgb2atd(img):
    return img**(1/gamma) @ Mng2xyz.T @ Mxyz2atd.T

def download_imagenet_subset(num_images=64, output_dir="imagenet_samples"):
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Streaming {num_images} images from Hugging Face (timm/mini-imagenet)...")
    
    # "streaming=True" allows us to load data without downloading the whole dataset
    dataset = load_dataset("timm/mini-imagenet", split="train", streaming=True)
    
    count = 0
    # Iterate through the stream
    imgs = []
    for sample in dataset:
        if count >= num_images:
            break
            
        image = sample['image']
        label = sample['label']
        imgs.append(image.resize((256,256)))
        count += 1
    return imgs

def get_imagenet_ready(num_images):
    """Downloads num_images imagenet images and returns them together
    with their atd transformation."""

    imgs = download_imagenet_subset(num_images=64)
    imgs = np.stack([np.array(img) for img in imgs])/255.
    atd = np.stack([rgb2atd(i) for i in imgs])

    return imgs, atd


def save_state(state, path):
    """Saves the state as .msgpack"""
    if path.split(".")[-1] != "msgpack":
        path = path + ".msgpack"
    with open(path, "wb") as f:
        f.write(msgpack_serialize(
            to_state_dict(state)
            ))

def load_state(path):
    """Loads the state in .msgpack"""
    
    with open(path, "rb") as f:
        state = msgpack_restore(f.read())

    return state
