import numpy as np
Label = np.zeros((7236, 1))
for i in range(0, 590):
    Label[i] = 0
for i in range(590, 1427):
    Label[i] = 1
for i in range(1427, 1913):
    Label[i] = 2
for i in range(1913, 2019):
    Label[i] = 3
for i in range(2019, 3039):
    Label[i] = 4
for i in range(3039, 3927):
    Label[i] = 5
for i in range(3927, 4493):
    Label[i] = 6
for i in range(4493, 6485):
    Label[i] = 7
for i in range(6485, 6948):
    Label[i] = 8
for i in range(6948, 7236):
    Label[i] = 9


np.save('Label.npy', Label)
