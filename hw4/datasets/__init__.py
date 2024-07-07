# from .blender import BlenderDataset
# from .llff import LLFFDataset
# dataset_dict = {'blender': BlenderDataset,
#                 'llff': LLFFDataset}

from .dataset import KlevrDataset
dataset_dict = {'Klevr': KlevrDataset}