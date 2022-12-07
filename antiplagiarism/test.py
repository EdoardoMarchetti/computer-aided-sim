import numpy as np
import pandas as pd


a = np.array(['aa', 'bb', 'cc', 'dd', 'ee', 'ff'], dtype=str)
a_strided = np.lib.stride_tricks.sliding_window_view(x=a, window_shape=(2,),).copy()
a_strided = pd.DataFrame(a_strided)
b = a_strided.apply(''.join, axis=1)
print(set(b))
