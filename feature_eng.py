import sys
sys.path.extend(['/Users/xinyux/PycharmProjects/CS249'])
import numpy as np
import pandas as pd
import jellyfish as jf
import pickle
from fuzzywuzzy import fuzz

def name_features(author_join, train_out):
    # Name Distance Features
    print "Add name distance features"
    name_df = author_join[["author_id", "paper_id", "author_name", "author_name_clean", "paper_author_name", "paper_author_name_clean","author_affiliation_clean","paper_author_affiliation_clean"]].copy()
    author_name_splits =  name_df['author_name_clean'].str.split(' ', 1, expand=True)
    name_df["author_first_name"] = author_name_splits[0]
    name_df["author_last_name"] = author_name_splits[1]

    paper_author_name_splits =  name_df['paper_author_name_clean'].str.split(' ', 1, expand=True)
    name_df["paper_author_first_name"] = paper_author_name_splits[0]
    name_df["paper_author_last_name"] = paper_author_name_splits[1]

    name_features = name_df[["author_id", "paper_id"]].copy()

    name_features["matched_substring_name_ratio"] = name_df.apply(
        lambda row: fuzz.ratio(unicode(row["author_name_clean"]), unicode(row["paper_author_name_clean"])),
        axis=1)
    name_features["matched_substring_affiliation_ratio"] = name_df.apply(
        lambda row: fuzz.ratio(unicode(row["author_affiliation_clean"]), unicode(["paper_author_affiliation_clean"])),
        axis=1)
    name_features["matched_substring_first_name_ratio"] = name_df.apply(
        lambda row: fuzz.ratio(unicode(row["author_first_name"]), unicode(row["paper_author_first_name"])),
        axis=1)
    name_features["matched_substring_last_name_ratio"] = name_df.apply(
        lambda row: fuzz.ratio(unicode(row["author_last_name"]), unicode(row["paper_author_last_name"])),
        axis=1)

    name_author_features = author_join[['author_id','author_name']].copy().drop_duplicates()
    name_author_features_group = name_author_features.groupby(["author_name"], sort=False)["author_id"].agg({"authors":(lambda group: list(group))})
    name_author_features_group = name_author_features_group.reset_index()
    name_author_features_group["num_authors_same_name"] = name_author_features_group.apply(lambda row: len(row["authors"]), axis=1)
    name_author_features = pd.merge(name_author_features, name_author_features_group, how="left", on="author_name")
    train_out = pd.merge(train_out, name_author_features, how="left", on="author_id")
    train_out = pd.merge(train_out, name_features, how="left", on=["author_id", "paper_id"])
    return train_out

print "Reading TrainOut.csv"
train_out = pd.read_csv("./data/TrainOut.csv")
print "Reading author_join"
author_join = pd.read_pickle("./pkl/author_join.pkl")


train_out = name_features(author_join, train_out)
train_out.sort_values(by="author_id").to_csv("./data/TrainOutNew.csv", index=False)

