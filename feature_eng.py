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

def author_features(author_info, author_join, paper_join, train_out):
	print "Generating author features"
	author_join_groupby = author_join.groupby(["author_id"])
	paper_join_groupby = paper_join.groupby(["author_id"])

	author_features = author_join[["author_id", "paper_id"]].copy()

	# Number of papers by the author
	author_paper_count = author_join_groupby["paper_id"].nunique()
	author_paper_count = author_paper_count.reset_index()
	author_paper_count = author_paper_count.rename(columns={"paper_id":"paper_count"})

	author_features = pd.merge(author_features, author_paper_count, how="left", on="author_id")
	author_features = author_features.fillna(0)
	author_features["paper_count"] = author_features["paper_count"].astype(int)

	# Number of papers by the author in journals
	valid_journal_ids = paper_join[paper_join.journal_id != 0]
	num_journal_papers = valid_journal_ids.groupby("author_id")["paper_id"].nunique()
	num_journal_papers = num_journal_papers.reset_index()
	num_journal_papers = num_journal_papers.rename(columns={"paper_id":"journal_paper_count"})

	author_features = pd.merge(author_features, num_journal_papers, how="left", on="author_id")
	author_features = author_features.fillna(0)
	author_features["journal_paper_count"] = author_features["journal_paper_count"].astype(int)

	# Journal Fraction
	journal_fraction = pd.merge(author_paper_count, num_journal_papers, how="left", on="author_id")
	journal_fraction["journal_paper_count"] = journal_fraction["journal_paper_count"].fillna(0)
	journal_fraction["paper_count"] = journal_fraction["paper_count"].astype(float)
	journal_fraction["journal_paper_fraction"] = journal_fraction["journal_paper_count"] / journal_fraction["paper_count"]

	author_features = pd.merge(author_features, journal_fraction[["author_id", "journal_paper_fraction"]], how="left", on="author_id")

	# Number of papers by the author in conferences
	valid_conference_ids = paper_join[paper_join.conference_id != 0]
	num_conference_papers = valid_conference_ids.groupby("author_id")["paper_id"].nunique()
	num_conference_papers = num_conference_papers.reset_index()
	num_conference_papers = num_conference_papers.rename(columns={"paper_id":"conference_paper_count"})

	author_features = pd.merge(author_features, num_conference_papers, how="left", on="author_id")
	author_features = author_features.fillna(0)
	author_features["conference_paper_count"] = author_features["conference_paper_count"].astype(int)

	# Conference Fraction
	conference_fraction = pd.merge(author_paper_count, num_conference_papers, how="left", on="author_id")
	conference_fraction["conference_paper_count"] = conference_fraction["conference_paper_count"].fillna(0)
	conference_fraction["paper_count"] = conference_fraction["paper_count"].astype(float)
	conference_fraction["conference_paper_fraction"] = conference_fraction["conference_paper_count"] / conference_fraction["paper_count"]

	author_features = pd.merge(author_features, conference_fraction[["author_id", "conference_paper_fraction"]], how="left", on="author_id")

	#Total Fraction
	author_features["total_paper_fraction"] = author_features["journal_paper_fraction"] + author_features["conference_paper_fraction"]

	# Number of journals the author had papers in
	num_journal_ids = paper_join_groupby["journal_id"].nunique()
	num_journal_ids = num_journal_ids.reset_index()
	num_journal_ids = num_journal_ids.rename(columns={"journal_id":"journal_count"})

	author_features = pd.merge(author_features, num_journal_ids, how="left", on="author_id")
	author_features = author_features.fillna(0)
	author_features["journal_count"] = author_features["journal_count"].astype(int)

	# Number of conferences the author had papers in
	num_conference_ids = paper_join_groupby["conference_id"].nunique()
	num_conference_ids = num_conference_ids.reset_index()
	num_conference_ids = num_conference_ids.rename(columns={"conference_id":"conference_count"})

	author_features = pd.merge(author_features, num_conference_ids, how="left", on="author_id")
	author_features = author_features.fillna(0)
	author_features["conference_count"] = author_features["conference_count"].astype(int)
	
	# Number of years the author was published in
	num_paper_years = paper_join_groupby["paper_year"].nunique()
	num_paper_years = num_paper_years.reset_index()
	num_paper_years = num_paper_years.rename(columns={"paper_year":"year_count"})

	author_features = pd.merge(author_features, num_paper_years, how="left", on="author_id")
	author_features = author_features.fillna(0)
	author_features["year_count"] = author_features["year_count"].astype(int)

	# Number of authors with same affiliation as this author
	valid_affiliation = author_info[author_info["author_affiliation_clean"] != ""].copy()
	valid_affiliation["same_affiliation_count"] = valid_affiliation.groupby("author_affiliation_clean")["author_affiliation_clean"].transform("count")
	num_same_affiliation = valid_affiliation[["author_id", "same_affiliation_count"]].copy()

	author_features = pd.merge(author_features, num_same_affiliation, how="left", on="author_id")
	author_features = author_features.fillna(0)
	author_features["same_affiliation_count"] = author_features["same_affiliation_count"].astype(int)

	print "Saving author_info to ./pkl/author_info.pkl"
	author_info.to_pickle("./pkl/author_info.pkl")

	train_out = pd.merge(train_out, author_features, how="left", on=["author_id", "paper_id"])

	return train_out


def name_features(author_join, train_out):
	# Name Distance Features
	print "Generating name distance features"
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
	print "Generating affiliation distance features"
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
	print "Generating publication year features"
	author_years = paper_join[paper_join.paper_year != 0].groupby(['author_id'], sort=False)['paper_year']

	ays_min = author_years.min().rename("min_pub_year")
	ays_max = author_years.max().rename("max_pub_year")
	ays_mean = author_years.mean().rename("mean_pub_year")
	ays_median = author_years.median().rename("median_pub_year")

	author_year_stats = pd.concat([ays_min, ays_max, ays_mean, ays_median], axis=1)
	author_year_stats['author_id'] = author_year_stats.index

	author_year_features = paper_join[["author_id", "paper_id", "paper_year"]].copy()

	author_year_features = pd.merge(author_year_features, author_year_stats, how="left", on="author_id")
	author_year_features = author_year_features.fillna(0)

	author_year_features["min_year_diff"] = author_year_features["min_pub_year"] - author_year_features["paper_year"]
	author_year_features["max_year_diff"] = author_year_features["max_pub_year"] - author_year_features["paper_year"]
	author_year_features["mean_year_diff"] = author_year_features["mean_pub_year"] - author_year_features["paper_year"]
	author_year_features["median_year_diff"] = author_year_features["median_pub_year"] - author_year_features["paper_year"]

	author_year_features_merge = author_year_features[["author_id", "paper_id", "min_year_diff", "max_year_diff", "mean_year_diff", "median_year_diff"]]

	train_out = pd.merge(train_out, author_year_features_merge, how="left", on=["author_id", "paper_id"])

	return train_out

def co_author_features(author_join, paper_join, train_out):
	print "Generating co author features"
	paper_author_groups = paper_join.groupby(["paper_id"], sort=False)["author_id"].agg({"authors":(lambda group: list(group))})
	paper_author_groups = paper_author_groups.reset_index()
	paper_author_groups["author_count"] = paper_author_groups.apply(lambda row: len(row["authors"]), axis=1)
	# paper_author_groups = pd.merge(paper_author_groups, paper_join[["paper_id", "author_id"]], how="left", on="paper_id")

	# co_authors = pd.DataFrame(paper_author_groups.authors.tolist(), index=[paper_author_groups.paper_id, paper_author_groups.author_id]).stack()
	# co_authors = co_authors.reset_index()[["paper_id", "author_id", 0]]
	# co_authors = co_authors.rename(columns={0:"co_author_id"})
	# co_authors["co_author_id"] = co_authors["co_author_id"].astype(int)
	# co_authors = co_authors[co_authors.author_id != co_authors.co_author_id]

	# author_affiliations = author_join[["author_id", "author_affiliation_clean"]].copy().drop_duplicates("author_id")
	# co_author_affiliations = author_affiliations.copy().rename(columns={"author_id":"co_author_id", "author_affiliation_clean":"co_author_affiliation_clean"})

	# co_authors = pd.merge(co_authors, author_affiliations, how="left", on="author_id")
	# co_authors["author_affiliation_clean"] = co_authors["author_affiliation_clean"].fillna("")
	# co_authors = pd.merge(co_authors, co_author_affiliations, how="left", on="co_author_id")
	# co_authors["co_author_affiliation_clean"] = co_authors["co_author_affiliation_clean"].fillna("")

	# co_authors["co_author_affilation_dist"] = co_authors.apply(
	# 	lambda row: jf.levenshtein_distance(unicode(row["author_affiliation_clean"]), unicode(row["co_author_affiliation_clean"])), 
	# 	axis=1)

	# affiliation_distances = co_authors.groupby(["author_id"], sort=False)["co_author_affiliation_dist"]
	# aff_min = affiliation_distances.min().rename("min_co_author_affiliation_dist")
	# aff_max = affiliation_distances.max().rename("min_co_author_affiliation_dist")
	# affiliation_stats = pd.concat([aff_min, aff_max], axis=1)

	# author_year_stats['author_id'] = author_year_stats.index

	co_author_features = paper_join[["paper_id", "author_id"]].copy()
	co_author_features = pd.merge(co_author_features, paper_author_groups[["paper_id", "author_count"]], how="left", on="paper_id")
	

	train_out = pd.merge(train_out, co_author_features, how="left", on=["author_id", "paper_id"])

	return train_out

print "Reading author_info"
author_info = pd.read_pickle("./pkl/author_info.pkl")

print "Reading author_join"
author_join = pd.read_pickle("./pkl/author_join.pkl")
print "Reading paper_join"
paper_join = pd.read_pickle("./pkl/paper_join.pkl")
print "Reading train_base"
train_out = pd.read_pickle("./pkl/train_base.pkl")

# train_out = has_features(author_join, paper_join, train_out)

train_out = author_features(author_info, author_join, paper_join, train_out)

# train_out = name_features(author_join, train_out)

# train_out = affiliation_features(author_join, train_out)

# train_out = author_year_features(paper_join, train_out)

# train_out = co_author_features(author_join, paper_join, train_out)

# train_out.pkl("./pkl/train_features.pkl")

out_columns=["author_id", 
		"paper_id", 
		"paper_count",
		"journal_paper_count",
		"journal_paper_fraction",
		"conference_paper_count",
		"conference_paper_fraction",
		"total_paper_fraction",
		"journal_count",
		"conference_count",
		"year_count",
		"same_affiliation_count"
		"wrote_paper"
	]


# out_columns=["author_id", 
# 		"paper_id", 
# 		"has_author_name",
# 		"has_author_affiliation",
# 		"has_paper_title",
# 		"has_paper_year",
# 		"has_conference_id",
# 		"has_journal_id",
# 		"has_paper_keyword",
# 		"name_clean_lev_dist",
# 		"first_name_lev_dist",
# 		"last_name_lev_dist",
# 		"name_clean_jaro_dist",
# 		"first_name_jaro_dist",
# 		"last_name_jaro_dist",
# 		"affiliation_clean_lev_dist",
# 		"affiliation_clean_jaro_dist",
# 		"min_year_diff",
# 		"max_year_diff",
# 		"mean_year_diff",
# 		"median_year_diff",
# 		"author_count",
# 		"wrote_paper"
# 	]

train_out.sort_values(by="author_id").to_csv("./TrainOut.csv", index=False, 
	columns=out_columns)

