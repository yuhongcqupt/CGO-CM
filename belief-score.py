import numpy as np
from scipy.stats import norm
from scipy.linalg import LinAlgError
from sklearn.cluster import KMeans
from ecm import ecm
from utils import extractMass, makeF
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import multivariate_normal
from sklearn.neighbors import KernelDensity


def calculate_belief(sample, mean, cov, eps=1e-4):
    cov = np.asarray(cov, dtype=float)
    if not np.isfinite(cov).all():
        d = mean.shape[0]
        cov = np.eye(d)


    d = cov.shape[0]
    cov_reg = cov + eps * np.eye(d)

    return multivariate_normal.logpdf(sample, mean=mean, cov=cov_reg, allow_singular=True)


def update_labels(X_0, mean_1, cov_1, belief_threshold):
    updated_labels = []
    for sample in X_0:
        belief = calculate_belief(sample, mean_1, cov_1)
        # print(belief,'---------------------')
        if belief > belief_threshold:
            updated_labels.append(1)
        else:
            updated_labels.append(0)
    return updated_labels




def update_labels_kde(X_0, kde_model, belief_threshold):
    updated_labels = []
    for sample in X_0:
        belief = calculate_belief_kde(sample, kde_model)

        if belief > belief_threshold:
            updated_labels.append(1)
        else:
            updated_labels.append(0)
    return updated_labels

