# This is a sample Python script.
import json

import numpy as np
import lightgbm as gbm
from allrank.data.dataset_loading import load_libsvm_dataset_role, load_libsvm_role
from sklearn.datasets import load_svmlight_file
from torch.utils.data import DataLoader

from utils.metrics import ndcg
from utils.call_self_attention import run
from utils.train_self_attention import compute_test


def get_data(path, role='train', fold='1'):
    X, y, qid = load_svmlight_file(f"{path}/Fold{fold}/{role}.txt", query_id=True)
    X = X.toarray()
    _, counts = np.unique(qid, return_counts=True)
    return X, y, qid, counts


if __name__ == '__main__':
    base_path = "toy_data/"
    fold = '1'

    home = f"{base_path}/Fold{fold}/"

    with open('local.json') as config_file:
        data = json.load(config_file)
        data['training']['epochs'] = 100
        data['data']['path'] = home
    with open('local.json', 'w') as config_file:
        json.dump(data, config_file)

    model, dev = run()

    test_ds = load_libsvm_role(home, "test")
    test_dl = DataLoader(test_ds, batch_size=2, num_workers=1, shuffle=False)

    predictions = compute_test(model, test_dl, dev)

    X_test, y_test, qid_test, qid_test_counts = get_data(base_path, "test")
    predictions_by_query = np.split(predictions, np.cumsum(qid_test_counts))
    y_test_by_query = np.split(y_test, np.cumsum(qid_test_counts))

    ndcg_eval = ndcg(y_test_by_query, predictions_by_query, k=10)
    print(np.mean(ndcg_eval))
