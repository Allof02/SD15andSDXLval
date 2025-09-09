import numpy as np
np.savez("coco_val2014_256_stats.npz",
         mu=np.load("mu.npy"),
         sigma=np.load("sigma.npy"))
