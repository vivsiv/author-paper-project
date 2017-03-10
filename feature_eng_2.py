# -*- coding: utf-8 -*-
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import numpy as np
import pandas as pd
import pickle
import jellyfish as jf

#def paper_conference_journal_features(train_out, paper_join):
#    print "Generating paper conference journal features"
#    author_paper = paper_join.groupby(["author_id"], sort=False)
#    author_conference_groups = author_paper["conference_id"].agg({"conference_cnt":(lambda x: x.nunique)})
#    author_conference_groups = author_conference_groups.reset_index()
#    
#    author_journal_groups = author_paper["journal_id"].agg({"journal_cnt":(lambda x: x.nunique)})
#    author_journal_groups = author_journal_groups.reset_index()
#    paper_conference_journal_features = pd.merge(author_conference_groups, author_journal_groups, how="left", on=["author_id"])
#    train_out = pd.merge(train_out, paper_conference_journal_features, how="left", on=["author_id"])
#    return train_out
    
def keyword_features(train_out,paper_join):
    print "Generating keyword features"
    author_paper = paper_join[["author_id","paper_id","paper_keyword"]]
    author_paper = author_paper.reset_index()
    author_paper["key_cnt"] = author_paper["paper_keyword"].apply(lambda x: len(x.split()))
    author_paper = author_paper.fillna(0)
#    author_paper["distance"] = author_paper.groupby("author_id")["paper_keyword"].apply(lambda x: max(x.split()))
    key_features = author_paper[["author_id","paper_id","key_cnt"]]
    
    train_out = pd.merge(train_out, key_features, how="left", on=["author_id", "paper_id"])
    return train_out


def paper_number_features(train_out, paper_join):
	print "Generating paper number features"
	author_paper = paper_join.groupby(["author_id"], sort=False)
	author_paper_groups = author_paper["paper_id"].agg({"papers":(lambda group: list(group))})
	author_paper_groups["paper_count"] = author_paper_groups.apply(lambda row: len(row["papers"]), axis=1)
	author_paper_groups = author_paper_groups.reset_index()

	paper_author_groups = paper_join[["author_id","paper_id"]]
	paper_author_groups = paper_author_groups.merge(author_paper_groups, how="left", on=["author_id"])
	temp = paper_author_groups.groupby("paper_id", sort = False)["paper_count"].sum().rename("coauthor_paper_count")
	temp = temp.reset_index()
	paper_number_features = paper_join[["author_id","paper_id"]].merge(temp, how="left", on=["paper_id"])
	train_out = pd.merge(train_out, paper_number_features, how="left", on=["author_id", "paper_id"])
	return train_out
     
def paper_conf_num_features(train_out, paper_join):
	print "Generating conference number features"
	author_paper = paper_join.groupby(["author_id"], sort=False)
	author_conf_groups = author_paper["conference_id"].agg({"conference":(lambda group: list(group))})
	author_conf_groups["conf_count"] = author_conf_groups.apply(lambda row: np.count_nonzero(row["conference"]), axis=1)
	author_conf_groups = author_conf_groups.reset_index()

	conf_author_groups = paper_join[["author_id","paper_id"]]
	conf_author_groups = conf_author_groups.merge(author_conf_groups, how="left", on=["author_id"])
	temp = conf_author_groups.groupby("paper_id", sort = False)["conf_count"].sum().rename("coauthor_conf_count")
	temp = temp.reset_index()
	conf_number_features = paper_join[["author_id","paper_id"]].merge(temp, how="left", on=["paper_id"])
	train_out = pd.merge(train_out, conf_number_features, how="left", on=["author_id", "paper_id"])
	return train_out
     
def paper_journal_num_features(train_out, paper_join):
	print "Generating journal number features"
	author_paper = paper_join.groupby(["author_id"], sort=False)
	author_jour_groups = author_paper["journal_id"].agg({"journal":(lambda group: list(group))})
	author_jour_groups["journal_count"] = author_jour_groups.apply(lambda row: np.count_nonzero(row["journal"]), axis=1)
	author_jour_groups = author_jour_groups.reset_index()

	jour_author_groups = paper_join[["author_id","paper_id"]]
	jour_author_groups = jour_author_groups.merge(author_jour_groups, how="left", on=["author_id"])
	temp = jour_author_groups.groupby("paper_id", sort = False)["jour_count"].sum().rename("coauthor_jour_count")
	temp = temp.reset_index()
	jour_number_features = paper_join[["author_id","paper_id"]].merge(temp, how="left", on=["paper_id"])
	train_out = pd.merge(train_out, jour_number_features, how="left", on=["author_id", "paper_id"])
	return train_out     

        

print "Reading author_join"
author_join = pd.read_pickle("./pkl/author_join.pkl")
print "Reading paper_join"
paper_join = pd.read_pickle("./pkl/paper_join.pkl")
print "Reading train_base"
train_out = pd.read_pickle("./pkl/train_base.pkl")

#train_out = paper_conference_journal_features(paper_join, train_out)
train_out = keyword_features(train_out, paper_join)


# train_out.pkl("./pkl/train_features.pkl")

out_columns=["author_id", 
		"paper_id",
		"key_cnt",
		"coauthor_paper_count",
		"coauthor_conf_count",
		"coauthor_jour_count"
	]

train_out.sort_values(by="author_id").to_csv("./TrainOut_2.csv", index=False, 
	columns=out_columns)

