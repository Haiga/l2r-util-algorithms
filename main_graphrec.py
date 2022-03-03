# This is a sample Python script.
import os
import logging
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').disabled = True

import numpy as np
from utils.graphrec import GraphRec, read_process
from utils.metrics import ndcg

if __name__ == '__main__':
    df = read_process("toy_data/ml100k/u.data", sep="\t")
    rows = len(df)
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    split_index_train = int(rows * 0.6)

    df_train = df[0:int(rows * 0.6)]
    df_vali = df[int(rows * 0.6) + 1:int(rows * 0.8)].reset_index(drop=True)
    df_test = df[int(rows * 0.8) + 1:].reset_index(drop=True)

    model = GraphRec(df_train, df_vali)

    df_test = df_test.sort_values('user')
    predictions = model.predict(df_test)

    qid_test_counts = df_test.groupby(['user']).size().tolist()
    y_test = df_test['rate'].tolist()

    predictions_by_user = np.split(predictions, np.cumsum(qid_test_counts))
    y_test_by_user = np.split(y_test, np.cumsum(qid_test_counts))

    ndcg_eval = ndcg(y_test_by_user, predictions_by_user, k=10)
    print(np.mean(ndcg_eval))
