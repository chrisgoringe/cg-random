import sys, os, git
import folder_paths
try:
    import custom_nodes.cg_custom_core
except:
    print("Installing cg_custom_nodes")
    repo_path = os.path.join(os.path.dirname(folder_paths.__file__), 'custom_nodes', 'cg_custom_core')  
    repo = git.Repo.clone_from('https://github.com/chrisgoringe/cg-custom-core.git/', repo_path)
    repo.git.clear_cache()
    repo.close()

sys.path.insert(0,os.path.dirname(os.path.realpath(__file__)))
from .randoms import *

NODE_CLASS_MAPPINGS = { "Random Int" : RandomInt,
                        "Random Float" : RandomFloat,
                        "Load Random Lora" : LoadRandomLora,
                        "Load Random Image" : LoadRandomImage,
                      }

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

