import random
from .common import Base_randoms, get_config_randoms
from nodes import LoraLoader, CheckpointLoaderSimple
from PIL import Image, ImageOps
import numpy as np
import torch
import os
  
class SeedContext():
    """
    Context Manager to allow one or more random numbers to be generated, optionally using a specified seed, 
    without changing the random number sequence for other code.
    """
    def __init__(self, seed=None):
        self.seed = seed
    def __enter__(self):
        self.state = random.getstate()
        if self.seed:
            random.seed(self.seed)
    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.state)

class RandomBase(Base_randoms):
    CATEGORY = "randoms"
    def IS_CHANGED(self, **kwargs):
        return random.random()

def SEED_INPUT():
    with SeedContext(None):
        return ("INT",{"default": random.randint(1,999999999), "min": 0, "max": 0xffffffffffffffff})

class RandomFloat(RandomBase):
    RETURN_NAMES = ("random_float",)
    REQUIRED = { 
                "minimum": ("FLOAT", {"default": 0.0}), 
                "maximum": ("FLOAT", {"default": 1.0}), 
                "seed": SEED_INPUT(),
    }
    OPTIONAL = { "decimal_places": ("INT", {"default": 10, "min":1, "max":20}), }
    RETURN_TYPES = ("FLOAT",)
    def func(self, minimum, maximum, seed, decimal_places=10):
        with SeedContext(seed):
            rand = round(random.uniform(minimum, maximum), decimal_places)
        return (rand,)

class RandomInt(RandomBase):
    RETURN_NAMES = ("random_int",)
    REQUIRED = { 
                "minimum": ("INT", {"default": 0}), 
                "maximum": ("INT", {"default": 99999999}), 
                "seed": SEED_INPUT(),
            }
    RETURN_TYPES = ("INT",)
    def func(self, minimum, maximum, seed):
        with SeedContext(seed):
            rand = random.randint(minimum, maximum)
        return (rand,)

def from_list(seed, list, index):
    if seed!=0:
        with SeedContext(seed):
            return (random.choice(list), index)
    else:
        index = (index+1)%len(list)
        return (list[index], index)

class LoadRandomLora(RandomBase, LoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        i = LoraLoader.INPUT_TYPES()
        i['required'].pop('lora_name')
        i['required']['seed'] = SEED_INPUT()
        i['optional'] = s.OPTIONAL
        return i
    RETURN_TYPES = ("MODEL", "CLIP", "STRING",)
    RETURN_NAMES = ("model", "clip", "lora_name",)
    OPTIONAL = {"lora_name": ("STRING", {"default":""})}

    def __init__(self):
        self.systematic_index = -1
        LoraLoader.__init__(self)

    def func(self, model, clip, strength_model, strength_clip, seed, lora_name=""):
        loras = get_config_randoms('lora_names', exception_if_missing_or_empty=True)
        if lora_name=="":
            lora_name, self.systematic_index = from_list(seed, loras, self.systematic_index)
        lora_name = lora_name if '.' in lora_name else lora_name + ".safetensors" 
        return self.load_lora(model, clip, lora_name, strength_model, strength_clip) + (lora_name,)

class LoadRandomCheckpoint(RandomBase, CheckpointLoaderSimple):
    systematic_index = -1
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "seed": SEED_INPUT() }, "optional": {"ckpt_name": ("STRING", {"default":""})} }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING",)
    RETURN_NAMES = ("model", "CLIP", "VAE", "ckpt_name",)

    def func(self, seed, ckpt_name=""):
        checkpoints = get_config_randoms('checkpoint_names', exception_if_missing_or_empty=True)
        if ckpt_name=="":
            ckpt_name, self.systematic_index = from_list(seed, checkpoints, self.systematic_index)
        ckpt_name = ckpt_name if '.' in ckpt_name else ckpt_name + '.safetensors'
        return self.load_checkpoint(ckpt_name) + (ckpt_name,)

class LoadRandomImage(RandomBase):
    def __init__(self):
        self.systematic_index = -1
    REQUIRED = { "folder": ("STRING", {} ), "seed": SEED_INPUT() }
    RETURN_TYPES = ("IMAGE","STRING",)
    RETURN_NAMES = ("image","filename",)
    OPTIONAL = {"filename": ("STRING", {"default":""})}

    def get_filenames(self, folder):
        image_extensions = get_config_randoms('image_extensions', exception_if_missing_or_empty=True)
        def is_image_filename(filename):
            split = os.path.splitext(filename)
            return len(split)>0 and split[1] in image_extensions
        files = [file for file in os.listdir(folder) if is_image_filename(file)]
        if len(files)==0:
            raise Exception(f"No files matching {image_extensions} in {folder}")
        return files

    def func(self, folder, seed, filename=""):
        if filename=="":
            filename, self.systematic_index = from_list(seed, self.get_filenames(folder), self.systematic_index)
        filepath = os.path.join(folder, filename)
        i = Image.open(filepath)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image, filename, )
