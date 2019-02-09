# import numpy as np
#
# append_zeros = np.arange(784).reshape(28,28)
# append_zeros = append_zeros.reshape((append_zeros.shape[0]*3, 28, 28))
# print(append_zeros)

import numpy as np

data3 = np.arange(122304).reshape(52, 2352 )
print(data3.shape)
data3 = data3.reshape((data3.shape[0]*3, 28, 28))
print(data3)
print(data3[0].shape)

zeros = np.zeros(64*64*3,dtype="float32").reshape((64,64,3))
print(zeros)
print(zeros.shape)
#must reach 12288
