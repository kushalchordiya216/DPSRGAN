# %%
import keras 
from keras.models import load_model
import cv2
import os
import numpy as np 
# %% 
model = load_model('model2.h5')

#%%
'''Load datasets'''

test_set = []

path_test = ('./Linnaeus 5 32x32/test/')
path2_test = ('./Linnaeus 5 64x64/test/')

for r1,d1,f1 in os.walk(path_test):
    f1 = f1[:1]
    for files in f1:
        test_set.append(cv2.imread(r1 + '/' + files))

print(len(test_set))
test_set = np.array(test_set)
print(test_set.shape)
#%%
result = model.predict(test_set,batch_size = 100, verbose=1)
print(result.shape)

#%%
result_list = list(result)
len(result_list)

#%%
i = 0
for r in result_list:
    cv2.imwrite(str(i)+'.jpg',r)
    i += 1
#%%
#%%
