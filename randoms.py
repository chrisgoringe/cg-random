import random
from common import *
from nodes import LoraLoader
  
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
    RETURN_NAMES = ("rand",)
    CATEGORY = "randoms"
    def IS_CHANGED(self, minimum, maximum, seed):
        return random.random()
    def func(self, minimum, maximum, seed, **kwargs):
        with SeedContext(seed):
            rand = self.gen(minimum, maximum)
            if 'decimal_places' in kwargs:
                rand = round(rand, kwargs['decimal_places'])
        return (rand,)

def SEED_INPUT():
    with SeedContext(None):
        return ("INT",{"default": random.randint(1,999999999), "min": 0, "max": 0xffffffffffffffff})

class RandomFloat(Base_randoms):
    RETURN_NAMES = ("random_float",)
    CATEGORY = "randoms"
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

class RandomInt(Base_randoms):
    RETURN_NAMES = ("random_int",)
    CATEGORY = "randoms"
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

class LoadRandomLora(Base_randoms, LoraLoader):
    CATEGORY = "randoms"
    systematic_index = -1
    @classmethod
    def INPUT_TYPES(s):
        i = LoraLoader.INPUT_TYPES()
        i['required'].pop('lora_name')
        i['required']['seed'] = SEED_INPUT()
        return i
    
    RETURN_TYPES = ("MODEL", "CLIP", "STRING",)
    RETURN_NAMES = ("model", "clip", "lora_name",)

    def __init__(self):
        LoraLoader.__init__(self)

    def func(self, model, clip, strength_model, strength_clip, seed):
        loras = get_config_randoms('lora_names', exception_if_missing_or_empty=True)
        if seed!=0:
            with SeedContext(seed):
                lora_name = random.choice(loras)
                lora_name = lora_name if '.' in lora_name else lora_name + ".safetensors" 
        else:
            lora_name = loras[self.systematic_index]
            self.systematic_index = (self.systematic_index + 1) % len(loras)
        return self.load_lora(model, clip, lora_name, strength_model, strength_clip) + (lora_name,)