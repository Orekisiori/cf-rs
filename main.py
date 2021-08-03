import numpy as np
import scipy.io as scio
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from tqdm import tqdm
import random
import logging


class collaborative_filtering:
    def __init__(self):
        self.count = 0
        self.SSIM = np.zeros((3000, 3000), dtype=np.float64)
        self.DVD = np.zeros((2505, 3000), dtype=np.float64)
        self.P = np.zeros((2505, 3000), dtype=np.float64)
        self.meta_data = np.zeros((2505, 3000), dtype=np.float64)
        self.W = np.zeros((2505, 3000), dtype=np.float64)
        self.marked = {}

    def get_data(self):
        data = scio.loadmat("data.mat")
        self.DVD = self.meta_data = data['DVD_Matrix']

    # 计算余弦相似度
    def compute_similarity(self):
        DVD_sparse = sparse.csr_matrix(self.DVD)
        DVD_sparse = DVD_sparse.transpose()
        self.SSIM = cosine_similarity(DVD_sparse)

    # 评分过程就是矩阵相乘的过程
    def compute_grade(self):
        # 矩阵相乘
        self.P = np.dot(self.DVD, self.SSIM)

    # 归一化，对每个物品的预测都除以其与其他物品的相似度之和
    def normalizer(self):
        self.P = self.P / np.sum(self.SSIM, axis=0)

    # 获取有评分的数据，字典形式存储，用以估计误差
    def get_marked(self):
        count = 0
        for i in range(2505):
            for j in range(3000):
                if self.DVD[i,j] != 0:
                    self.marked[count] = [i,j]
                    count += 1

    # 首次评分前，由于存在大量未打分数据，需要根据用户的评分情况构造权重矩阵，完成归一化，迭代时则不需要
    # 出现不可处理数据时用3代替
    def first_normalizer(self):
        self.W = np.load('weight_for_DVD.npy')
        self.P = self.P / self.W
        # print("打印评分矩阵\n",self.P)
        where_are_nan = np.isnan(self.P)
        where_are_inf = np.isinf(self.P)
        self.P[where_are_nan] = 3
        self.P[where_are_inf] = 3

        for i in range(2505):
            for j in range(3000):
                if self.P[i,j] > 5:
                    self.P[i,j] = 5
                elif self.P[i,j] < 1:
                    self.P[i,j] = 1


    # 用3填补稀疏数据
    def set_mark(self):
        for i in range(2505):
            for j in range(3000):
                if self.DVD[i,j] == 0:
                    self.DVD[i,j] = 3


    # 通过保存的标签字典，将原数据中的正确标签带回迭代矩阵
    def change(self):
        for _,tpl in self.marked.items():
            self.DVD[tpl[0],tpl[1]] = self.meta_data[tpl[0],tpl[1]]


    # 绝对误差均值，评价指标
    def compute_MAE(self):
        loss = 0.000
        for i in range(1000):
            r = random.randint(0, 92514)
            m = self.marked[r]
            p = m[0]
            q = m[1]
            loss += abs(self.meta_data[p,q] - self.P[p,q])
            # print(loss)
        self.count += 1
        print("迭代第%d次，MAE为：%f\n" %(self.count,loss/1000))

    # 迭代方法
    def iter(self):
        self.DVD = self.P
        self.change()
        self.compute_similarity()
        self.compute_grade()
        self.normalizer()
        self.compute_MAE()


def create_logger():
    """
    输出到日志文件
    """
    log_path = 'training.log'

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


if __name__ == '__main__':
    logger = create_logger()
    CF = collaborative_filtering()
    CF.get_data()
    CF.get_marked()
    CF.compute_similarity()
    CF.compute_grade()
    CF.first_normalizer()

    print('打印初次评分矩阵:\n {}\n'.format(CF.P))

    for i in tqdm(range(50)):
        CF.iter()
        logger.info('judge matrix：\n{}'.format(CF.P))

    print('打印并保存最终评分矩阵:\n {}\n'.format(CF.P))
    np.save("mark_for_DVD.npy", CF.P)