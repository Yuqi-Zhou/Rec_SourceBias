import numpy as np

def fast_dcg_score(y_true, y_score, k=[10]):
    result = []
    order = np.argsort(y_score)[::-1]
    for i in k:
        y_true_k = np.take(y_true, order[:i])
        gains = 2**y_true_k - 1
        discounts = np.log2(np.arange(len(y_true_k)) + 2)
        result.append(np.sum(gains / discounts))
    return result

def fast_ndcg_score(y_true, y_score, k=[10]):
    best = fast_dcg_score(y_true, y_true, k)
    actual = fast_dcg_score(y_true, y_score, k)
    return [a/b for b,a in zip(best, actual)]

def fast_map_score(y_true, y_score, k=[10]):
    result = []
    rel_count = np.sum(y_true)
    order = np.argsort(y_score)[::-1]
    for i in k:
        y_true_k = np.take(y_true, order[:i])
        AP = np.cumsum(y_true_k)*y_true_k/(np.arange(len(y_true_k)) + 1)
        result.append(np.sum(AP) / rel_count)
    return result