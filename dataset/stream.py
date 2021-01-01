import pandas as pd

import sys
sys.path.append('..')
from model.utils import Attr, AttrType
from collections import Counter

class DataStream:
    def __init__(self, infile, attrTypes=None, shuffle=False, seed=0, cyclic=False):
        df = pd.read_csv(infile)

        self.shuffle = shuffle
        self.seed = seed
        
        if shuffle:
            df = df.sample(frac=1, random_state=seed)

        self.attributes = []
        for idx, name in enumerate(df.columns[:-1]):
            if attrTypes is None:
                if df[name].dtype == object:
                    self.attributes.append(Attr(idx, AttrType.CATE, name))
                else:
                    self.attributes.append(Attr(idx, AttrType.NUME, name))
            else:
                self.attributes.append(Attr(idx, attrTypes[idx], name))

        self.n_class = len(set(df.iloc[:, -1]))
        self.X, self.y = df.iloc[:, :-1].values, df.iloc[:, -1].values

        for index, attr in enumerate(self.attributes):
            if attr.type == AttrType.CATE:
                attr.values = list(set(self.X[:, index]))

        print('data distribution:')
        print(Counter(self.y))
        self.num_inst = self.X.shape[0]
        self.pointer = 0

        self.cyclic = cyclic

    def nextInstance(self):
        if not self.cyclic and self.pointer >= self.num_inst:
            return None

        x, y = self.X[self.pointer], self.y[self.pointer]
        self.pointer += 1
        if self.pointer == self.num_inst and self.cyclic:
            self.pointer = 0

        return x, y
    
    def reset(self):
        self.pointer = 0

