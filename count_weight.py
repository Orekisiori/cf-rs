import time
import numpy as np
import scipy.io as scio
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numba as nb


data = scio.loadmat("data.mat")
Video = data['Video_Matrix']
Video_sparse = sparse.csr_matrix(Video)
Video_sparse = Video_sparse.transpose()
SSIM = cosine_similarity(Video_sparse)
print("相似度计算完成")


W = np.zeros((2505, 3000))

@nb.jit()
def count_weight(W):
    for i in range(2505):
        for j in range(3000):
            temp_w = 0
            for k in range(3000):
                if Video[i, k] != 0:
                    temp_w += SSIM[k, j]
            W[i, j] = temp_w


tt = time.time()
print('开始计算权重矩阵')
count_weight(W)
np.save("weight_for_video",W)
print('Time used: {} secs'.format(time.time() - tt))

