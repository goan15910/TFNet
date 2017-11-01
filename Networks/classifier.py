import os, sys
import numpy as np
import math
from math import ceil

# modules
from NN_base.nn_base import NN_BASE


class CLASSIFIER(NN_BASE):
  """Classifier structure"""
  def __init__(self):
    NN_BASE.__init__(self):

  
  def build(self):
    raise NotImplementedError


  def loss(self, logits, labels):
    raise NotImplementedError

  # TODO: 
  #  1. conv
  #  2. fc
  #  3. 
