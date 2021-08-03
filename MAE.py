import numpy as np
import scipy.io as scio
from tqdm import tqdm
import random


def compute_MAE(P,META,dict,count,last_loss):
    loss = 0.000
    for i in range(1000):
        r = random.randint(0, 87500)
        m = dict[r]
        p = m[0]
        q = m[1]
        loss += abs(META[p, q] - P[p, q])
    count += 1
    print("当前迭代MAE为：%f\n" % ( loss / 1000))
    last_loss += loss/1000
    return last_loss

def normalize(M):
    for i in range(2505):
        for j in range(3000):
            if M[i, j] > 4.5:
                M[i, j] = 5
            elif 3.5 < M[i, j] < 4.5:
                M[i,j] = 4
            elif 2.5 < M[i, j] < 3.5:
                M[i,j] = 3
            elif 1.5 < M[i, j] < 2.5:
                M[i,j] = 2
            elif M[i, j] < 1.5:
                M[i, j] = 1

    np.save("final_D.npy", M)
    return M


# 获取有评分的数据，字典形式存储，用以估计误差
def get_marked(META):
    dict = {}
    count = 0
    for i in range(2505):
        for j in range(3000):
            if META[i,j] != 0:
                dict[count] = [i,j]
                count += 1
    return dict

if __name__ == '__main__':
    count = 0
    last_loss = 0
    P = np.load("mark_for_DVD.npy")
    data = scio.loadmat("data.mat")
    META = data['DVD_Matrix']
    P = normalize(P)
    dict = get_marked(META)

    for i in tqdm(range(10)):
        last_loss = compute_MAE(P, META, dict, count, last_loss)

    print("10次平均MAE：%f" %(last_loss/10))