import random
from .common import BaseNode, SeedContext, SEED_INPUT

class RandomBase(BaseNode):
    CATEGORY = "randoms"
    def IS_CHANGED(self, **kwargs):
        return random.random()

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

