import time
import pandas as pd

from metrics.gini import gini

from model.utils import Attr, AttrType
from model.tree import ClsTree
from model.vfdt import VfdtTree
from model.efdt import EfdtTree
from collections import Counter

if __name__ == '__main__':
    start_time = time.time()

    # dataset
    df = pd.read_csv('./dataset/bank.csv', sep=';')
    df = df.sample(frac=1).reset_index(drop=True)  # random sample

    candidate_attr = []
    for idx, name in enumerate(df.columns[:-1]):
        if df[name].dtype == object:
            candidate_attr.append(Attr(idx, AttrType.CATE, name))
        else:
            candidate_attr.append(Attr(idx, AttrType.NUME, name))

    n_class = len(set(df.iloc[:, -1]))

    split = 1000
    train_X, train_y = df.iloc[:-split, :-1].values, df.iloc[:-split, -1].values
    test_X, test_y = df.iloc[-split:, :-1].values, df.iloc[-split:, -1].values

    # hyperparameter
    delta = 0.01
    max_depth = 100
    min_sample = 5

    # entropy: from scipy.stats import entropy
    # entropy(class_freq)

    # Cls Tree
    print('Cls Tree')
    model = ClsTree(candidate_attr, max_depth, min_sample)
    model.fit(train_X, train_y, lambda class_freq: -gini(class_freq))
    # model.print()
    pred_y = model.predict(train_X)
    print(Counter(pred_y))
    print(f'train: {sum(pred_y == train_y) / len(train_y)}')
    pred_y = model.predict(test_X)
    print(Counter(pred_y))
    print(f'test: {sum(pred_y == test_y) / len(test_y)}')

    # VFDT Tree
    print('VFDT Tree')
    model = VfdtTree(candidate_attr, n_class, delta, max_depth, min_sample)
    model.update(train_X, train_y, lambda class_freq: -gini(class_freq))
    # model.print()
    pred_y = model.predict(train_X)
    print(Counter(pred_y))
    print(f'train: {sum(pred_y == train_y) / len(train_y)}')
    pred_y = model.predict(test_X)
    print(Counter(pred_y))
    print(f'test: {sum(pred_y == test_y) / len(test_y)}')

    # EFDT Tree
    print('EFDT Tree')
    model = EfdtTree(candidate_attr, n_class, delta, max_depth, min_sample)
    model.update(train_X, train_y, lambda class_freq: -gini(class_freq))
    # model.print()
    pred_y = model.predict(train_X)
    print(Counter(pred_y))
    print(f'train: {sum(pred_y == train_y) / len(train_y)}')
    pred_y = model.predict(test_X)
    print(Counter(pred_y))
    print(f'test: {sum(pred_y == test_y) / len(test_y)}')




