import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import numpy as np
import pandas as pd
import pickle
import editdistance as edit
import jellyfish as jf

def has_features(author_join, paper_join, train_out):
	# Has Features
	print "Generating has features"
	has_author_features = author_join[["author_id", "paper_id"]].copy()
	has_author_features["has_author_name"] = np.where(author_join["author_name"] == "", 0, 1)
	has_author_features["has_author_affiliation"] = np.where(author_join["author_affiliation"] == "", 0, 1)

	train_out = pd.merge(train_out, has_author_features, how="left", on=["author_id", "paper_id"])

	has_paper_features = paper_join[["author_id", "paper_id"]].copy()
	has_paper_features["has_paper_title"] = np.where(paper_join["paper_title"] == "", 0, 1)
	has_paper_features["has_paper_year"] = np.where(paper_join["paper_year"] == 0, 0, 1)
	has_paper_features["has_conference_id"] = np.where(paper_join["conference_id"] == 0, 0, 1)
	has_paper_features["has_journal_id"] = np.where(paper_join["journal_id"] == 0, 0, 1)
	has_paper_features["has_paper_keyword"] = np.where(paper_join["paper_keyword"] == "", 0, 1)
	
	train_out = pd.merge(train_out, has_paper_features, how="left", on=["author_id", "paper_id"])

	return train_out

def name_features(author_join, train_out):
	# Name Distance Features
	print "Generating name_distance features"
	name_df = author_join[["author_id", "paper_id", "author_name", "author_name_clean", "paper_author_name", "paper_author_name_clean"]].copy()
	author_name_splits =  name_df['author_name_clean'].str.split(' ', 1, expand=True)
	name_df["author_first_name"] = author_name_splits[0]
	name_df["author_last_name"] = author_name_splits[1]

	paper_author_name_splits =  name_df['paper_author_name_clean'].str.split(' ', 1, expand=True)
	name_df["paper_author_first_name"] = paper_author_name_splits[0]
	name_df["paper_author_last_name"] = paper_author_name_splits[1]

	name_features = name_df[["author_id", "paper_id"]].copy()

	name_features["name_clean_lev_dist"] = name_df.apply(
		lambda row: jf.levenshtein_distance(unicode(row["author_name_clean"]), unicode(row["paper_author_name_clean"])), 
		axis=1)
	name_features["first_name_lev_dist"] = name_df.apply(
		lambda row: jf.levenshtein_distance(unicode(row["author_first_name"]), unicode(row["paper_author_first_name"])), 
		axis=1)
	name_features["last_name_lev_dist"] = name_df.apply(
		lambda row: jf.levenshtein_distance(unicode(row["author_last_name"]), unicode(row["paper_author_last_name"])), 
		axis=1)

	name_features["name_clean_jaro_dist"] = name_df.apply(
		lambda row: jf.jaro_distance(unicode(row["author_name_clean"]), unicode(row["paper_author_name_clean"])), 
		axis=1)
	name_features["first_name_jaro_dist"] = name_df.apply(
		lambda row: jf.jaro_distance(unicode(row["author_first_name"]), unicode(row["paper_author_first_name"])), 
		axis=1)
	name_features["last_name_jaro_dist"] = name_df.apply(
		lambda row: jf.jaro_distance(unicode(row["author_last_name"]), unicode(row["paper_author_last_name"])), 
		axis=1)

	train_out = pd.merge(train_out, name_features, how="left", on=["author_id", "paper_id"])

	return train_out

def affiliation_features(author_join, train_out):
	# Affiliation Distance Features
	print "Generating affiliation_distance features"
	affiliation_features = author_join[["author_id", "paper_id"]].copy()

	affiliation_features["affiliation_clean_lev_dist"] = author_join.apply(
		lambda row: jf.levenshtein_distance(unicode(row["author_affiliation_clean"]), unicode(row["paper_author_affiliation_clean"])), 
		axis=1)
	affiliation_features["affiliation_clean_jaro_dist"] = author_join.apply(
		lambda row: jf.jaro_distance(unicode(row["author_affiliation_clean"]), unicode(row["paper_author_affiliation_clean"])), 
		axis=1)

	train_out = pd.merge(train_out, affiliation_features, how="left", on=["author_id", "paper_id"])

	return train_out

def author_year_features(paper_join, train_out):
	# Author Publication Year Features
	print "Generating publication_year features"
	author_years = paper_join[paper_join.paper_year != 0].groupby(['author_id'], sort=False)['paper_year']

	ays_min = author_years.min().rename("min_pub_year")
	ays_max = author_years.max().rename("max_pub_year")
	ays_mean = author_years.mean().rename("mean_pub_year")
	ays_median = author_years.median().rename("median_pub_year")

	author_year_stats = pd.concat([ays_min, ays_max, ays_mean, ays_median], axis=1)
	author_year_stats['author_id'] = author_year_stats.index

	author_year_features = paper_join[["author_id", "paper_id", "paper_year"]].copy()
	author_year_features = pd.merge(author_year_features, author_year_stats, how="left", on="author_id")
	author_year_features["min_year_diff"] = author_year_features["min_pub_year"] - author_year_features["paper_year"]
	author_year_features["max_year_diff"] = author_year_features["max_pub_year"] - author_year_features["paper_year"]
	author_year_features["mean_year_diff"] = author_year_features["mean_pub_year"] - author_year_features["paper_year"]
	author_year_features["median_year_diff"] = author_year_features["median_pub_year"] - author_year_features["paper_year"]

	author_year_features_merge = author_year_features[["author_id", "paper_id", "min_year_diff", "max_year_diff", "mean_year_diff", "median_year_diff"]]

	train_out = pd.merge(train_out, author_year_features_merge, how="left", on=["author_id", "paper_id"])

	return train_out

def co_author_features(paper_join, train_out):
	co_authors = paper_join.groupby(["paper_id"], sort=False)["author_id"].agg({"co_authors":(lambda group: list(group))})
	co_authors["paper_id"] = co_authors.index
	co_authors["num_authors"] = co_authors.apply(lambda row: len(row["co_authors"]), axis=1)
	co_author_features = paper_join[["paper_id", "author_id"]].copy()
	co_author_features = pd.merge(co_author_features, co_authors[["paper_id", "num_authors"]], how="left", on="paper_id")

	train_out = pd.merge(train_out, co_author_features, how="left", on=["author_id", "paper_id"])

	return train_out


print "Reading author_join"
author_join = pd.read_pickle("./pkl/author_join.pkl")
print "Reading paper_join"
paper_join = pd.read_pickle("./pkl/paper_join.pkl")
print "Reading train_base"
train_out = pd.read_pickle("./pkl/train_base.pkl")

train_out = has_features(author_join, paper_join, train_out)

train_out = name_features(author_join, train_out)

train_out = affiliation_features(author_join, train_out)

train_out = author_year_features(paper_join, train_out)

train_out.pkl("./pkl/train_features.pkl")
