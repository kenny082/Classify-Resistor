import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os

masks = []
for file in os.listdir("masks/1"):
    masks.append(np.load(os.path.join("masks/1", file)))
masks = np.stack(masks)
plt.imshow(np.mean(masks, axis=0))
plt.show()