# %%
# %%
import cv2
import numpy as np
import os

stream = cv2.VideoCapture('Trainspotting.mp4')

count = 0
ret = True
while ret:
    ret, frame = stream.read()
    if count % 30 == 0:
        resized = cv2.resize(frame, dsize=(640,360), interpolation=cv2.INTER_LINEAR)
        # img_path = f'./data/720/{count}.jpg'
        img_path_resized = f'./data/480/{count}.jpg'
        # cv2.imwrite(img_path, frame)
        cv2.imwrite(img_path_resized,resized)
    count += 1

    k = cv2.waitKey(5) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()

