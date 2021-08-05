## 协同过滤

我们限定一下范围，假设现在给你一个用户-商品的评分矩阵(1~5)，矩阵中的每个数代表用户对该商品的喜爱程度，问如何设计一个推荐系统

这个问题的本质就是对用户对商品的评分进行预测，将预测评分高的商品推荐给用户，这里我们用到的算法就是协同过滤

协同过滤有两个维度：商品维度：将与该用户喜爱的商品相似的商品推荐给用户，也就是说与用户最喜欢的商品最相似的商品评分最高；用户维度：将与该用户相似的用户喜爱的商品推荐给用户。问题进一步转化成如何衡量商品或者用户的相似性的问题

用户的相似性：两名用户对同一件商品的打分越接近，两名用户越相似

商品的相似性：同一位用户对两件商品的打分越接近，两件商品越接近

这边深入研究一下商品相似性导向的推荐算法的做法，用户相似性其实就是转置一下的事

对商品的相似度度量有多种办法，这边我们选取余弦相似度

```python
DVD_sparse = sparse.csr_matrix(self.DVD)
DVD_sparse = DVD_sparse.transpose()
self.SSIM = cosine_similarity(DVD_sparse)
```

获取数据

```python
    def __init__(self):
        self.count = 0
        # 相似矩阵
        self.SSIM = np.zeros((3000, 3000), dtype=np.float64)
        self.DVD = np.zeros((2505, 3000), dtype=np.float64)
        # 评分矩阵
        self.P = np.zeros((2505, 3000), dtype=np.float64)
        # 原始数据
        self.meta_data = np.zeros((2505, 3000), dtype=np.float64)
        # 权重矩阵
        self.W = np.zeros((2505, 3000), dtype=np.float64)
        self.marked = {}
```

矩阵相乘就是评分

```python
self.P = np.dot(self.DVD, self.SSIM)
```

+ `def get_marked(self):`获取有评分的数据
+ `def first_normalizer(self)`：由于数据稀疏，根据用户的评分情况构造权重矩阵，将所有不能处理的数据用3代替
+ `def compute_MAE(self):`将正确标签回代后根据MAE（绝对平均误差）评价

权重计算，@nb.jit()是numba的修饰器，可以用来加速python的计算，在本函数中加速了600倍

```python
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
```





