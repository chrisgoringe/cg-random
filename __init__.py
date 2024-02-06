import sys, os

sys.path.insert(0,os.path.dirname(os.path.realpath(__file__)))
from .randoms import *
from .systematics import *
from .loaders import *

NODE_CLASS_MAPPINGS = { "Random Int" : RandomInt,
                        "Systematic Int" : SystematicInt,
                        "Random Float" : RandomFloat,
                        "Systematic Float" : SystematicFloat,
                        "Load Random Lora" : LoadRandomLora,
                        "Load Random Image" : LoadRandomImage,
                        "Load Random Checkpoint" : LoadRandomCheckpoint,
                      }

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

version = "1.1"