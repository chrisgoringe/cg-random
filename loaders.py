from .common import SeedContext, SEED_INPUT
import random
from .randoms import RandomBase

from nodes import LoraLoader, CheckpointLoaderSimple
from PIL import Image, ImageOps
import numpy as np
import torch
import os

from folder_paths import folder_names_and_paths, get_folder_paths
from comfy.sd import load_checkpoint_guess_config

class RandomLoaderException(Exception):
    pass

class KeepForRandomBase(RandomBase):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "seed": SEED_INPUT(), 
                             "keep_for": ("INT", {"default": 1, "min":1, "max":100}), 
                             "mode": ( ["random", "systematic"], {}), 
                             "subfolder": ("STRING", {"default":"random"}) } }
    
    @classmethod
    def add_input_types(cls, it):
        add = KeepForRandomBase.INPUT_TYPES()
        for x in add['required']: it['required'][x] = add['required'][x]
        return it

    def __init__(self):
        self.since_last_change = 0
        self.last_systematic = -1
        self.result = None
        self.systematic = False

    def func(self, seed, keep_for, mode, subfolder, **kwargs):
        self.subfolder = subfolder
        self.since_last_change += 1
        self.systematic = (mode=="systematic")
        if self.since_last_change >= keep_for or self.result is None:
            self.since_last_change = 0
            with SeedContext(seed):
                self.result = self.func_(**kwargs)
        return self.result
    
    def _get_list(self, category):
        fnap = folder_names_and_paths[category]
        options = set()
        for folder in fnap[0]:
            random_folder = os.path.join(folder, self.subfolder)
            if os.path.exists(random_folder):
                for file in os.listdir(random_folder):
                    if os.path.splitext(file)[1] in fnap[1]:
                        options.add(os.path.join(random_folder,file))
        return list(options)
    
    def choose_from(self, category_or_list):
        lst = self._get_list(category_or_list) if isinstance(category_or_list,str) else category_or_list
        if not len(lst): raise RandomLoaderException(f"Nothing in list to choose from")
        if self.systematic:
            self.last_systematic = (1+self.last_systematic) % len(lst)
            return lst[self.last_systematic]
        else:
            return random.choice(lst)

class LoadRandomCheckpoint(KeepForRandomBase, CheckpointLoaderSimple):
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING",)
    RETURN_NAMES = ("model", "CLIP", "VAE", "ckpt_name",)

    def func_(self):
        ckpt_path = self.choose_from("checkpoints")
        out = load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=get_folder_paths("embeddings"))
        return out[:3] + (os.path.splitext(os.path.split(ckpt_path)[1])[0],)
    
class LoadRandomLora(KeepForRandomBase, LoraLoader):
    RETURN_TYPES = ("MODEL", "CLIP", "STRING",)
    RETURN_NAMES = ("model", "clip", "lora_name",)

    def __init__(self):
        LoraLoader.__init__(self)
        KeepForRandomBase.__init__(self)

    @classmethod
    def INPUT_TYPES(cls):
        it = LoraLoader.INPUT_TYPES()
        it['required'].pop('lora_name')
        return cls.add_input_types(it)

    def func_(self, **kwargs):
        lora_name = self.choose_from("loras")
        lora_name = os.path.join(self.subfolder, os.path.split(lora_name)[1])
        return self.load_lora(lora_name=lora_name, **kwargs) + (os.path.splitext(os.path.split(lora_name)[1])[0],)
    
    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("NaN")

class LoadRandomImage(KeepForRandomBase):
    @classmethod
    def INPUT_TYPES(cls):
        it = {'required': {"folder":("string", {"default":""}), "extensions":("string", {"default":".png, .jpg, .jpeg"})}}
        return cls.add_input_types(it)

    RETURN_TYPES = ("IMAGE","STRING",)
    RETURN_NAMES = ("image","filepath",)

    def get_filenames(self, folder:str, extensions:str):
        image_extensions = (e.strip() for e in extensions.split(","))
        is_image_filename = lambda a : os.path.split(a)[1] in image_extensions
        return [file for file in os.listdir(folder) if is_image_filename(file)]

    def func_(self, folder:str, extensions:str):
        filename = self.choose_from(self.get_filenames(folder, extensions))
        filepath = os.path.join(folder, filename)
        i = Image.open(filepath)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image, filepath, )
