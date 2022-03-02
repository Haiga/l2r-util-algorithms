import numpy as np


def dcg(true_relevance, pred_relevance, k=5, use_numpy=False):
    np_true_relevance = np.array(true_relevance)
    if not use_numpy:
        args_pred = [i[0] for i in sorted(enumerate(pred_relevance), key=lambda p: p[1], reverse=True)]
    else:
        args_pred = np.argsort(pred_relevance)[::-1]
    if np_true_relevance.shape[0] < k:
        k = np_true_relevance.shape[0]
    order_true_relevance = np.take(np_true_relevance, args_pred[:k])

    gains = 2 ** order_true_relevance - 1
    discounts = np.log2(np.arange(k) + 2)
    return np.sum(gains / discounts)


def ndcg_one_query(true_relevance, pred_relevance, k=5, no_relevant=False, use_numpy=False):
    dcg_atk = dcg(true_relevance, pred_relevance, k, use_numpy)
    idcg_atk = dcg(true_relevance, true_relevance, k, use_numpy)
    if idcg_atk == 0 and no_relevant: return 1.0
    if idcg_atk == 0 and not no_relevant: return 0.0
    return dcg_atk / idcg_atk


def ndcg(true_relevance, pred_relevance, k=5, no_relevant=False, use_numpy=False):
    np_true_relevance = np.array(true_relevance)
    num_queries = np_true_relevance.shape[0]
    return [ndcg_one_query(true_relevance[i], pred_relevance[i], k, no_relevant, use_numpy) for i in range(num_queries)]
