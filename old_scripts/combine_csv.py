# -*- coding: utf-8 -*-
import pandas as pd

a = pd.read_csv("TrainOut.csv")
b = pd.read_csv("TrainOut_2.csv")
b = b.dropna(axis=1)
merged = a.merge(b, how="left", on=["author_id", "paper_id"])
write = merged['wrote_paper']
merged.drop(labels=['wrote_paper'], axis=1,inplace = True)
merged.insert(len(merged.columns),'wrote_paper', write)
merged.to_csv("Train_out_final.csv", index=False)