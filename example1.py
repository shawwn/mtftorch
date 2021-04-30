from functools import partial

import numpy as np; import tensorflow as tf; import mtftorch as torch; mtf = torch.mtf; from mtftorch.nn import parameter; c = torch.ones(dict(N=1, H=16,W=4,C=3)) 

torch.reset(); v = torch.randn("H=2 W=4", seed=(0,3))

v = mtf.gather(v, v.cwise(partial(tf.argsort, direction="DESCENDING", axis=-1)).long().view("H=2 ?W=-1"), v.size("W")).view("H=2 W=4")

v = mtf.gather(v, v.cwise(partial(tf.argsort, direction="ASCENDING", axis=0)).long().view("?H=2 W=-1"), v.size("H")).view("H=2 W=4") 

def gather(x, indices, dim):
  dim = indices.size(dim)
  shape = indices.shape.rename_dimension(dim.name, "?" + dim.name)
  v = mtf.gather(x, indices.view(shape).long(), dim)
  return v.view(indices.shape)
