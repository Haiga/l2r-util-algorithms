# This is a sample Python script.

import numpy as np
import lightgbm as gbm
from sklearn.datasets import load_svmlight_file
from utils.metrics import ndcg


def get_data(path, role='train', fold='1'):
    X, y, qid = load_svmlight_file(f"{path}/Fold{fold}/{role}.txt", query_id=True)
    X = X.toarray()
    _, counts = np.unique(qid, return_counts=True)
    return X, y, qid, counts


if __name__ == '__main__':
    home = "toy_data/"
    fold = '1'

    X, y, qid, qid_counts = get_data(home, "train")
    X_test, y_test, qid_test, qid_test_counts = get_data(home, "test")
    X_vali, y_vali, qid_vali, qid_vali_counts = get_data(home, "vali")

    model = gbm.LGBMRanker(objective="lambdarank", metric="ndcg", learning_rate=0.0005, n_estimators=50)

    model.fit(X=X, y=y, group=qid_counts, eval_set=[(X_vali, y_vali)], eval_group=[qid_vali_counts], eval_at=10,
              verbose=10)

    predictions = model.predict(
        X=X_test,
        group=qid_test_counts
    )

    predictions_by_query = np.split(predictions, np.cumsum(qid_test_counts))
    y_test_by_query = np.split(y_test, np.cumsum(qid_test_counts))

    ndcg_eval = ndcg(y_test_by_query, predictions_by_query, k=10)
    print(np.mean(ndcg_eval))
