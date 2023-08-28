import sys, os, importlib, re
sys.path.insert(0,os.path.dirname(os.path.realpath(__file__)))
from randoms import *

NODE_CLASS_MAPPINGS = { "Random Int" : RandomInt,
                        "Random Float" : RandomFloat,
                        "Load Random Lora" : LoadRandomLora,
                      }

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

