import numpy as np
from skimage import morphology, io
from scipy import ndimage as nd
import matplotlib.pyplot as plt

im3d = np.random.rand(2, 3, 6)
# show first plane
print(im3d[0])
plt.imshow(im3d[0], cmap="gray")
plt.show()
seeds = nd.label(im3d < 0.1)[0]
ws = morphology.watershed(im3d, seeds)
