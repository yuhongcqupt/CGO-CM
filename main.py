import random

from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import os
from cluster_trace import TraceBasedClustering, visualizeSelectedClusters
from cluster_trace import visualizeSelectedClusters_withoutlabel,visualizeSelectedClusters_withlabel,visualizeSelectedClusters_result_6
from scipy.stats import uniform
from sklearn import metrics
from sklearn import decomposition as dec
from sklearn.base import BaseEstimator
from trace import hcd_index
from DPC import DPC
from sklearn.cluster import KMeans
import gc
from scipy.optimize import linear_sum_assignment

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
np.set_printoptions(threshold=np.inf)

import numpy as np
from sklearn.neighbors import NearestNeighbors




def best_map(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 映射成连续整数（避免标签是{1,3,7}这种）
    _, y_true_i = np.unique(y_true, return_inverse=True)
    _, y_pred_i = np.unique(y_pred, return_inverse=True)

    D = max(y_true_i.max(), y_pred_i.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(len(y_true_i)):
        w[y_pred_i[i], y_true_i[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    mapped_pred_i = np.zeros_like(y_pred_i)
    for r, c in zip(row_ind, col_ind):
        mapped_pred_i[y_pred_i == r] = c

    # 把“连续整数标签”映射回“原始真实标签值”
    true_label_values = np.unique(y_true)
    mapped_pred = true_label_values[mapped_pred_i]
    return mapped_pred

# 自定义一个Estimator类，确保参数可以通过get_params()获取
class GridSearchEstimator(BaseEstimator):
    def __init__(self, kk=None ,belief_threshold=None, mean_greater_or_equal_than=None, median_greater_or_equal_than=None, T=None, k=None,d=None,threshold=None):
        self.belief_threshold = belief_threshold
        self.kk = kk
        self.mean_greater_or_equal_than = mean_greater_or_equal_than
        self.median_greater_or_equal_than = median_greater_or_equal_than
        self.T = T
        self.k = k
        self.d = d
        self.threshold = threshold

    def align_clusters(self,true_labels, predicted_labels):
        """自动调整 KMeans 聚类标签，使其匹配原始 0,1,2"""
        cm = confusion_matrix(true_labels, predicted_labels)
        row_ind, col_ind = linear_sum_assignment(-cm)  # 最大匹配
        mapping = {old: new for old, new in zip(col_ind, row_ind)}
        return np.array([mapping[label] for label in predicted_labels])
    def purity_score(self, y_true, y_pred):
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        # 对每一列（每个簇）取最大值之和
        return np.sum(np.amax(cm, axis=0)) / np.sum(cm)

    def fit(self, data, labels,times, T, threshold, k, d,nn):

        # 确保参数是浮点数类型
        self.belief_threshold = float(self.belief_threshold)
        self.mean_greater_or_equal_than = float(self.mean_greater_or_equal_than)
        self.median_greater_or_equal_than = float(self.median_greater_or_equal_than)
        self.kk = self.kk

        print('k,T,d,threshold参数2为：',k,T,d,threshold)
        # ---------------------归一化数据,改变了原始分布--------------------------
        for j in range(data.shape[1]):
            max_ = max(data[:, j])
            min_ = min(data[:, j])
            if max_ == min_:
                continue
            for i in range(data.shape[0]):
                data[i][j] = (data[i][j] - min_) / (max_ - min_)
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        #---------------------------------------------------------------
        # 调用 PCA
        # 第一次可视化，
        visualizeSelectedClusters_withoutlabel(data, nn)
        visualizeSelectedClusters_withlabel(data, labels)
        # 调用 HIAC 处理数据


        save_dir = "."  # Define save directory
        distanceTGP = TGP(data, k, os.path.join(save_dir, "decision_graph.png"), threshold)
        neighbor_index = prune(data, k, threshold, distanceTGP)

        # 调用 hcd_index
        hcd_df,ff = hcd_index(data,  k=self.kk, belief_threshold=self.belief_threshold,
                           mean_greater_or_equal_than=self.mean_greater_or_equal_than,
                           median_greater_or_equal_than=self.median_greater_or_equal_than)
        # if ff==1:
        #     return None
        # 获取标签
        label = hcd_df['cluster_label'].values
        idx_0 = np.where(label == 0)[0]
        idx_1 = np.where(label == 1)[0]

        # 修改邻接矩阵
        for i in idx_1:
            for j in idx_0:
                neighbor_index[i, j] = -1  # 类别0到类别1的连接置为-1
                neighbor_index[j, i] = -1  # 类别1到类别0的连接置为-1
        # Ameliorate the dataset by d time-segments
        ori_data = data.copy()
        for i in range(d):
            bata = shrink(data, k, T, neighbor_index)
            data = bata
# 第五次展示被优化了分布的数据（带标签）
        visualizeSelectedClusters(data, labels,'ameliorated',vv=1)
        # 调用 DPC 进行聚类并计算 NMI 分数
        cluster_num = max(labels) - min(labels) + 1  # Example value; replace with your cluster number
#------------------------------使用的聚类方法----------------------------------------------------
        kmeans = KMeans(n_clusters=cluster_num, random_state=42)
        kmeans.fit(data)


        res = kmeans.labels_

        res = self.align_clusters(labels, res)
# ------------------------------使用的聚类方法----------------------------------------------------

        # 最终聚类结果可视化
        # visualizeSelectedClusters_result_6(ori_data, res)

        nmi = adjusted_mutual_info_score(labels, res, average_method='max')
        ari = adjusted_rand_score(labels, res)
        purity = self.purity_score(labels, res)
        if nmi==1:
            print(labels)
            print(res)
        return nmi,ari,purity

    def score(self, data, labels=None):
        # 返回模型的得分，这里使用 NMI 作为评价标准
        return self.fit(data, labels)

def manual_random_search(data, labels, n_iter):#param_dist,

    best_nmi = -1
    best_ari = -1
    best_purity = -1
    best_params = None
    times=0
    for nn in range(n_iter):
        print(nn)
        times=times+1

        # ---------------------------------指定参数---------------------------------
        belief_threshold = -490
        mean_threshold = 0.9005671193425526          #这个参数会产生变化
        median_threshold = 0.9947794349518033  #这个参数也会发生变化
        kk = 26 #这个参数有变化，变化不大
        #
        print("参数1为", "belief_threshold:", belief_threshold, "mean_threshold", mean_threshold, 'median_threshold',
              median_threshold, 'kk', kk)
        #
        k = 7 #从两个区间内随机采样
        T = 0.522154232562239 #uniform.rvs(loc=a, scale=b)，表示：从区间[a,a+b] 上，均匀随机取一个实数
        d = 9
        threshold = 0.9349821030028543#1.4116150073348097

# #--------------------------------------------------------------------------
        # 使用选择的超参数
        estimator = GridSearchEstimator(
            belief_threshold=belief_threshold,
            mean_greater_or_equal_than=mean_threshold,
            median_greater_or_equal_than=median_threshold,
            kk=kk,
            T=T,
            threshold=threshold,
            k=k,
            d=d
        )


        # 计算 NMI
        nmi,ari,purity = estimator.fit(data, labels,times, T, threshold, k, d,nn)
        print(purity,ari,nmi)
        if nmi==None:
            print('Finish this')
            continue
        # 更新最佳结果
        if (nmi > best_nmi and ari > best_ari):
        # if nmi > best_nmiand ari > best_ari and purity > best_purity:
            best_nmi = nmi
            best_ari = ari
            best_purity = purity
            best_params = {
                'belief_threshold': belief_threshold,
                'mean_greater_or_equal_than': mean_threshold,
                'median_greater_or_equal_than': median_threshold,
                'kk':kk
            }
        del estimator  # 删除不再需要的对象
        gc.collect()  # 手动触发垃圾回收
    return best_params, best_nmi,best_ari,best_purity


# 读取数据
#__________________________----------------------------------------
filePath = "data-sets/real-datasets/糖尿病-女性.txt"  # 数据集路径

data = np.loadtxt(filePath)

labels = data[:, -1]
labels = np.array(labels, dtype=np.int32)
data = data[:, :-1]



cluster_num = max(labels) - min(labels) + 1



# 使用随机搜索进行超参数优化
best_params, best_nmi,ari,purity = manual_random_search(data, labels,  n_iter=1) #开始循环param_dist,

# 执行随机搜索
print("Best Parameters:", best_params)
print("Best Purity、ARI、NMI:")
print(purity)
print(ari)
print(best_nmi)



gc.collect()