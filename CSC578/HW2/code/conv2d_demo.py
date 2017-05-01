"""You may use this demo to verify your implementation of the Con2D layer.

"""

import numpy as np
from layer import Conv2D

stride = 2
padding = 2
filter_size = (3, 2, 2, 2)

x = np.array(range(32)).reshape((2, 4, 4)).astype(np.float)
dv_y = np.array(range(48)).reshape(3, 4, 4).astype(np.float)
W = np.ones(filter_size)
W[1] *= 2
W[2] *= 3
b = np.zeros((filter_size[0], 1))

conv = Conv2D(filter_size, stride, padding)
conv.W = W
conv.b = b

# Forward test
y = conv.forward(x)
print(y)
"""Expected output of y
[[[   0.    0.    0.    0.]
  [   0.   84.  100.    0.]
  [   0.  148.  164.    0.]
  [   0.    0.    0.    0.]]

 [[   0.    0.    0.    0.]
  [   0.  168.  200.    0.]
  [   0.  296.  328.    0.]
  [   0.    0.    0.    0.]]

 [[   0.    0.    0.    0.]
  [   0.  252.  300.    0.]
  [   0.  444.  492.    0.]
  [   0.    0.    0.    0.]]]
"""

# Backward test
dv_x, dv_W, dv_b = conv.backward(x, dv_y)
print(dv_x)
"""Expected output of dv_x
[[[ 158.  158.  164.  164.]
  [ 158.  158.  164.  164.]
  [ 182.  182.  188.  188.]
  [ 182.  182.  188.  188.]]

 [[ 158.  158.  164.  164.]
  [ 158.  158.  164.  164.]
  [ 182.  182.  188.  188.]
  [ 182.  182.  188.  188.]]]
"""
print(dv_W)
"""Expected output of dv_W
[[[[  184.   214.]
   [  304.   334.]]

  [[  664.   694.]
   [  784.   814.]]]


 [[[  504.   598.]
   [  880.   974.]]

  [[ 2008.  2102.]
   [ 2384.  2478.]]]


 [[[  824.   982.]
   [ 1456.  1614.]]

  [[ 3352.  3510.]
   [ 3984.  4142.]]]]
"""
print(dv_b)
"""Expected output of dv_b
[[ 120.]
 [ 376.]
 [ 632.]]
"""
