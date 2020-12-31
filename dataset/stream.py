import pandas as pd

import sys
sys.path.append('..')
from model.utils import Attr, AttrType

class DataStream:
    def __init__(self, infile, shuffle=False, seed=0, cyclic=False):
        df = pd.read_csv(infile)
        
        if shuffle:
            df = df.sample(frac=1, random_state=seed)

        self.attributes = []
        for idx, name in enumerate(df.columns[:-1]):
            if df[name].dtype == object:
                self.attributes.append(Attr(idx, AttrType.CATE, name))
            else:
                self.attributes.append(Attr(idx, AttrType.NUME, name))

        self.n_class = len(set(df.iloc[:, -1]))
        self.X, self.y = df.iloc[:, :-1].values, df.iloc[:, -1].values
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
