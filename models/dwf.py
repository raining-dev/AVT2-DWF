import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math

from transformers.activations import ACT2FN
from transformers.pytorch_utils import apply_chunking_to_forward

class DwfFusion(nn.Module):
    #coming soon