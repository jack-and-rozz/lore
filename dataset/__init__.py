from .base import *
from occult.dataset.multiturn_dialog import MultiTurnDialogDataset


import os, importlib

extension_root = os.environ['OCCULT_EXTENSIONS']
if extension_root:
  m = importlib.import_module(extension_root + '.dataset')
  print(dir(m))

