from pprint import pprint as pp
import mtftorch as torch


from collections import OrderedDict

def listify(x):
  if x is None:
    return []
  if not isinstance(x, (list, tuple)):
    return [x]
  return list(x)

def grad_downstream(ys, xs, operations=None):
  ys = listify(ys)
  # if xs is None:
  #   xs = backward_tensors(group(ys) if len(ys) != 1 else ys)
  if operations is None:
    operations = ys[0].graph.operations
  # figure out what Tensors are downstream of xs
  downstream = set(xs)
  for op in operations:
    if op.has_gradient:
      if set(op.inputs) & downstream:
        downstream |= set(op.outputs)
  return downstream

def traverse_ops(tensor, fn, seen=None, inputs=True, outputs=True):
  if seen is None:
    seen = []
  op = as_operation(op)
  if op in seen:
    return
  seen.append(op)
  if fn(tensor) is False:
    return
  if inputs:
    for i in op.inputs:
      traverse_ops(i, fn, seen=seen, inputs=inputs, outputs=outputs)
  if outputs:
    for o in op.outputs:
      traverse_ops(o, fn, seen=seen, inputs=inputs, outputs=outputs)

def backward_slots(tensors, operations=None):
  slots = []
  def traverse(t):
    if has_grad_slot(t):
      slots.append(t)
  seen = []
  tensors = listify(tensors)
  for t in tensors:
    traverse_ops(t, traverse, seen=seen)
  return slots

torch.TensorMixin.backward_slots = backward_slots

import tensorflow.compat.v1 as tf


def grad(ys, xs, grad_ys=None, operations=None):
  if operations is None:
    operations = ys[0].graph.operations
  if grad_ys is None:
    grad_ys = [torch.ones_like(y) for y in ys]
    # grad_ys = [i+1 for i, y in enumerate(ys)]
  downstream = grad_downstream(ys, xs, operations=operations)
  tensor_to_gradient = {y: g for y, g in zip(ys, grad_ys) if g is not None}
  with tf.variable_scope(ys[0].graph.captured_variable_scope):
    for op in operations[::-1]:
      grad_outputs = [tensor_to_gradient.get(out) for out in op.outputs]
      if op.has_gradient and any(grad_outputs) and (set(op.inputs) & set(downstream)):
        # grad_outputs = [torch.ones_like(ys[g-1]) if isinstance(g, int) else g for g in grad_outputs]
        with tf.variable_scope(op.name + "/gradients"):
          input_grads = op.gradient(grad_outputs)
          for inp, grad in zip(op.inputs, input_grads):
            if inp in downstream and grad is not None:
              if inp in tensor_to_gradient:
                tensor_to_gradient[inp] += grad
              else:
                tensor_to_gradient[inp] = grad
  # return [tensor_to_gradient.get(x, None) for x in xs]
  return OrderedDict([(x, tensor_to_gradient.get(x, None)) for x in xs if x in tensor_to_gradient]), tensor_to_gradient

def grads(ys, xs=None, grad_ys=None, operations=None):
  if not isinstance(ys, (tuple, list)):
    ys = [ys]
    if grad_ys is not None and not isinstance(grad_ys, (tuple, list)):
      grad_ys = [grad_ys]
    if xs is not None and not isinstance(xs, (tuple, list)):
      xs = [xs]
  if xs is None:
    xs = backward_tensors()
  gradients, _ = grad(ys, xs, grad_ys=grad_ys, operations=operations)
  return gradients

def as_operation(x):
  if hasattr(x, 'operation'):
    x = x.operation
  assert isinstance(x, torch.mtf.Operation)
  return x

def backward_tensors(start=None, operations=None):
  if operations is None:
    operations = torch.get_graph().operations
  if start is not None:
    start = max([operations.index(as_operation(op)) for op in listify(start)])
    operations = operations[:start]
  ts = []
  for op in operations[::-1]:
    for o in op.inputs:
      if o not in ts and has_grad_slot(o):
        ts.append(o)
  for op in operations:
    for o in op.outputs:
      if o not in ts and has_grad_slot(o):
        ts.append(o)
  return list(ts)

def get_variable(data, dtype=None, *, shape=None, name=None, initializer=None, trainable=True):
  # dtype = torch.get_dtype(dtype, torch.get_dtype(data, tf.float32) or tf.float32)
  if torch.is_tf_tensor(data) or torch.is_numpy(data):
    data = torch.tensor(data, shape=shape)
  if not torch.is_tensor(data):
    data = torch.zeros(data, dtype=dtype)
  # if shape is not None:
  #   shape = torch.shapelist(data, shape)
  # else:
  #   shape = torch.size(data)
  shape = torch.shapelist(data, shape)
  if initializer is None:
    initializer = tf.zeros_initializer()
  if name is None:
    #variables = torch.get_mesh().graph.all_variables
    #name = 'variable_%d' % len(variables)
    name = data.operation.name + '/value'
  t = torch.mtf.get_variable(torch.get_mesh(), name, shape=shape, dtype=data.dtype, trainable=trainable, initializer=initializer)
  return parameter.Parameter(t, requires_grad=trainable)
  # if trainable and t.dtype.is_floating_point:
  #   t._requires_grad = True
  # return t

torch.get_variable = get_variable

def get_parameter(data, dtype=None, *, name=None, initializer=None, trainable=True):
  return get_buffer(data=data, dtype=dtype, name=name, initializer=initializer, trainable=trainable)

torch.get_parameter = get_parameter

def has_grad_slot(tensor):
  return tensor._requires_grad and tensor.dtype.is_floating_point

def get_slot(tensor, slot="grad", initializer=None):
  assert has_grad_slot(tensor)
  variable = tensor.operation
  #assert isinstance(variable, torch.mtf.Variable)
  get_buffer(tensor.operation.name + '/slots/' + slot, shape=tensor.shape, dtype=tensor.dtype)
  return torch.mtf.get_variable(tensor.mesh, tensor.operation.name + '/slots/' + slot, shape=tensor.shape, trainable=False, initializer=initializer)

torch.get_slot = get_slot

def grad_slot(tensor):
  if isinstance(tensor, (list, tuple)):
    return [grad_slot(x) for x in tensor]
  assert has_grad_slot(tensor)
  grad = getattr(tensor, 'grad', None)
  if grad is None:
    grad = get_slot(tensor, 'grad')
    tensor.grad = grad
  return grad

def backward_ops(tensor, *, update_ops=None):
  xs = backward_tensors(tensor)
  tensor_to_gradient = grads(tensor, xs)
  if update_ops is None:
    update_ops = []
  for tensor, gradient in tensor_to_gradient.items():
    g = grad_slot(tensor)
    op = torch.mtf.assign_add(g.operation, gradient)
    update_ops.append(op)
  return update_ops

def zero_grad_ops(xs=None, *, update_ops=None):
  if update_ops is None:
    update_ops = []
  for tensor in backward_tensors(xs):
    g = grad_slot(tensor)
    op = torch.mtf.assign(g.operation, torch.mtf.zeros_like(g))
    update_ops.append(op)
  return update_ops

def no_op(dependencies=None):
  op = torch.tensor([], "anonymous")
  if dependencies is not None:
    op = torch.mtf.depend(op, dependencies)
  return op

torch.no_op = no_op

def group(ops):
  if ops is None:
    ops = []
  if not isinstance(ops, (tuple, list)):
    ops = [ops]
  ops = list(ops)
  return no_op(ops)

torch.group = group

def backward(tensor):
  ops = backward_ops(tensor)
  return group(ops)

torch.TensorMixin.backward = backward

torch.TensorMixin.backward_tensors = backward_tensors

def zero_grad(model=None):
  ops = zero_grad_ops(model)
  return group(ops)

torch.zero_grad = zero_grad

def get_session():
  return torch.get_converter().get_session()

torch.get_session = get_session

def use_session():
  return get_session().as_default()

torch.use_session = use_session

if __name__ == '__main__':
  from mtftorch.testing._internal.common_utils import tensorflow_startup
  tensorflow_startup()

  x = torch.tensor([1.0], "C=1", requires_grad=True)

  with torch.no_grad():
    with torch.enable_grad():
      y = x * 2.0

  y.requires_grad

  # y.backward()

  # x.grad

  @torch.enable_grad()
  def doubler(x):
      return x * 2

  with torch.no_grad():
      z = doubler(x)
      w = z ** 2.0
      with torch.enable_grad():
        q1 = z * 42.0
        q2 = w * 42.0

  z.requires_grad

  channel = torch.ones("H=4 W=4")
  pixel = torch.tensor([0.1, 0.5, 0.9], "C")
  image_hwc = torch.tensor(channel * pixel, requires_grad=True)

  x2 = torch.tensor([42.0], "C=1", requires_grad=True)

  # torch.set_grad_enabled(True)

  with torch.enable_grad():
    loss = image_hwc.mean()
    loss += x2 ** 2

  train_op = loss.backward().tf()
  init_op = zero_grad().tf()

  #gs, tensor_to_gradient = grad([2.0 * z, x + y, x - y], [z, x, y, q1, q2])

  with torch.use_session():
    init_op.eval()
    for i in range(8):
      train_op.eval()

  tf.get_logger().setLevel("WARN")
