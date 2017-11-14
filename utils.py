import numpy as np
from easydict import EasyDict as edict


def join_keys_mapping(d1, d2):
  """
  Mapping values from join keys of two dict.
  Args:
    d1: dict 1
    d2: dict 2
  Return:
    A dictionary with items of v1: v2,
    where v1, v2 is the value from join keys of d1, d2
  """
  #join_keys = set().union(d1.keys(), d2.keys())
  join_keys = set(d1.keys()) & set(d2.keys())
  join_keys = list(join_keys)
  value_dict = {}
  for k in join_keys:
    v1 = d1[k]
    v2 = d2[k]
    value_dict[v1] = v2
  return value_dict
