import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import numpy as np
import pandas as pd
import pickle
import editdistance as edit
import jellyfish as jf

def author_features(author_join, train_out):
	print "Generating author features"	
	author_features = author_join[["author_id", "paper_id"]].copy()

	# Has features
	author_features["has_author_name"] = np.where(author_join["author_name"] == "", 0, 1)
	author_features["has_author_affiliation"] = np.where(author_join["author_affiliation"] == "", 0, 1)

	# Number of papers by the author
	valid_paper_ids = author_join[author_join["paper_id"] != 0]
	author_paper_count = valid_paper_ids.groupby("author_id")["paper_id"].nunique().reset_index().rename(columns={"paper_id":"author_paper_count"})
	
	author_features = pd.merge(author_features, author_paper_count, how="left", on="author_id")
	author_features["author_paper_count"] = author_features["author_paper_count"].fillna(0).astype(int)

	train_out = pd.merge(train_out, author_features, how="left", on=["author_id", "paper_id"])

	return train_out

def author_journal_features(author_join, paper_join, train_out):
	print "Generating author journal features"
	author_features = author_join[["author_id", "paper_id"]].copy()

	# Number of papers by the author
	valid_paper_ids = author_join[author_join["paper_id"] != 0]
	author_paper_count = valid_paper_ids.groupby("author_id")["paper_id"].nunique().reset_index().rename(columns={"paper_id":"author_paper_count"})

	valid_journal_ids = paper_join[paper_join["journal_id"] != 0]

	# Number of journals the author had papers in
	num_journal_ids = valid_journal_ids.groupby("author_id")["journal_id"].nunique().reset_index().rename(columns={"journal_id":"author_journal_count"})
	author_features = pd.merge(author_features, num_journal_ids, how="left", on="author_id")
	author_features["author_journal_count"] = author_features["author_journal_count"].fillna(0).astype(int)

	# Number of papers by the author in journals
	num_journal_papers = valid_journal_ids.groupby("author_id")["paper_id"].nunique().reset_index().rename(columns={"paper_id":"author_journal_paper_count"})
	author_features = pd.merge(author_features, num_journal_papers, how="left", on="author_id")
	author_features["author_journal_paper_count"] = author_features["author_journal_paper_count"].fillna(0).astype(int)

	# Journal Fraction
	journal_fraction = pd.merge(author_paper_count, num_journal_papers, how="left", on="author_id")
	journal_fraction["author_journal_paper_count"] = journal_fraction["author_journal_paper_count"].fillna(0).astype(float)
	journal_fraction["author_paper_count"] = journal_fraction["author_paper_count"].astype(float)
	journal_fraction["author_journal_paper_fraction"] = journal_fraction["author_journal_paper_count"] / journal_fraction["author_paper_count"]

	author_features = pd.merge(author_features, journal_fraction[["author_id", "author_journal_paper_fraction"]], how="left", on="author_id")
	author_features["author_journal_paper_fraction"] = author_features["author_journal_paper_fraction"].fillna(0)

	# Journal stats
	journal_counts = valid_journal_ids.groupby(["author_id", "journal_id"]).size().to_frame().reset_index().rename(columns={0:"author_journal_paper_count"})
	journal_count_groups = journal_counts.groupby("author_id")["author_journal_paper_count"]

	journal_min = journal_count_groups.min().rename("author_journal_min_freq")
	journal_max = journal_count_groups.max().rename("author_journal_max_freq")
	journal_mean = journal_count_groups.mean().rename("author_journal_mean_freq")
	journal_median = journal_count_groups.median().rename("author_journal_median_freq")
	journal_std = journal_count_groups.std().rename("author_journal_std_freq")

	journal_stats = pd.concat([journal_min, journal_max, journal_mean, journal_median, journal_std], axis=1).reset_index()

	author_features = pd.merge(author_features, journal_stats, how="left", on="author_id")
	author_features = author_features.fillna(0)

	# Total Fraction
	# author_features["total_paper_fraction"] = author_features["journal_paper_fraction"] + author_features["conference_paper_fraction"]

	train_out = pd.merge(train_out, author_features, how="left", on=["author_id", "paper_id"])

	return train_out

def author_conference_features(author_join, paper_join, train_out):
	print "Generating author conference features"
	author_features = author_join[["author_id", "paper_id"]].copy()

	# Number of papers by the author
	valid_paper_ids = author_join[author_join["paper_id"] != 0]
	author_paper_count = valid_paper_ids.groupby("author_id")["paper_id"].nunique().reset_index().rename(columns={"paper_id":"author_paper_count"})

	valid_conference_ids = paper_join[paper_join["conference_id"] != 0]

	# Number of conferences the author had papers in
	num_conference_ids = valid_conference_ids.groupby("author_id")["conference_id"].nunique().reset_index().rename(columns={"conference_id":"author_conference_count"})
	author_features = pd.merge(author_features, num_conference_ids, how="left", on="author_id")
	author_features["author_conference_count"] = author_features["author_conference_count"].fillna(0).astype(int)

	# Number of papers by the author in conferences
	num_conference_papers = valid_conference_ids.groupby("author_id")["paper_id"].nunique().reset_index().rename(columns={"paper_id":"author_conference_paper_count"})
	author_features = pd.merge(author_features, num_conference_papers, how="left", on="author_id")
	author_features["author_conference_paper_count"] = author_features["author_conference_paper_count"].fillna(0).astype(int)

	# Conference Fraction
	conference_fraction = pd.merge(author_paper_count, num_conference_papers, how="left", on="author_id")
	conference_fraction["author_conference_paper_count"] = conference_fraction["author_conference_paper_count"].fillna(0).astype(float)
	conference_fraction["author_paper_count"] = conference_fraction["author_paper_count"].astype(float)
	conference_fraction["author_conference_paper_fraction"] = conference_fraction["author_conference_paper_count"] / conference_fraction["author_paper_count"]
	author_features = pd.merge(author_features, conference_fraction[["author_id", "author_conference_paper_fraction"]], how="left", on="author_id")

	# Conference Stats
	conference_counts = valid_conference_ids.groupby(["author_id", "conference_id"]).size().to_frame().reset_index().rename(columns={0:"author_conference_paper_count"})
	conference_count_groups = conference_counts.groupby("author_id")["author_conference_paper_count"]

	conference_min = conference_count_groups.min().rename("author_conference_min_freq")
	conference_max = conference_count_groups.max().rename("author_conference_max_freq")
	conference_mean = conference_count_groups.mean().rename("author_conference_mean_freq")
	conference_median = conference_count_groups.median().rename("author_conference_median_freq")
	conference_std = conference_count_groups.std().rename("author_conference_std_freq")

	conference_stats = pd.concat([conference_min, conference_max, conference_mean, conference_median, conference_std], axis=1).reset_index()

	author_features = pd.merge(author_features, conference_stats, how="left", on="author_id")
	author_features = author_features.fillna(0)

	train_out = pd.merge(train_out, author_features, how="left", on=["author_id", "paper_id"])

	return train_out

def author_affiliation_features(author_info, author_join, train_out):
	# Affiliation Distance Features
	print "Generating affiliation features"
	affiliation_features = author_join[["author_id", "paper_id", "author_affiliation_clean"]].copy()

	# Number of authors with same affiliation as this author
	valid_affiliation = author_info[author_info["author_affiliation_clean"] != ""].copy()
	num_same_affiliation = valid_affiliation.groupby("author_affiliation_clean")["author_id"].nunique().reset_index().rename(columns={"author_id":"author_same_affiliation_count"})
	affiliation_features = pd.merge(affiliation_features, num_same_affiliation, how="left", on="author_affiliation_clean")
	affiliation_features["author_same_affiliation_count"] = affiliation_features["author_same_affiliation_count"].fillna(0).astype(int)
	affiliation_features = affiliation_features.drop(["author_affiliation_clean"], axis=1)

	# String distance between affiliation in author table and affiliation in paper author table
	valid_affiliation = author_join[author_join["author_affiliation_clean"] != ""].copy()
	affiliation_distance = author_join[["author_id", "paper_id"]].copy()
	affiliation_distance["author_affiliation_lev_dist"] = valid_affiliation.apply(
		lambda row: jf.levenshtein_distance(unicode(row["author_affiliation_clean"]), unicode(row["paper_author_affiliation_clean"])), 
		axis=1)
	affiliation_distance["author_affiliation_jaro_dist"] = valid_affiliation.apply(
		lambda row: jf.jaro_distance(unicode(row["author_affiliation_clean"]), unicode(row["paper_author_affiliation_clean"])), 
		axis=1)
	affiliation_features = pd.merge(affiliation_features, affiliation_distance, how="left", on=["author_id", "paper_id"])
	affiliation_features["author_affiliation_lev_dist"] = affiliation_features["author_affiliation_lev_dist"].fillna(1).astype(int)
	affiliation_features["author_affiliation_jaro_dist"] = affiliation_features["author_affiliation_jaro_dist"].fillna(0.5)

	train_out = pd.merge(train_out, affiliation_features, how="left", on=["author_id", "paper_id"])

	return train_out

def author_year_features(paper_join, train_out):
	# Author Publication Year Features
	print "Generating publication year features"
	year_features = author_join[["author_id", "paper_id"]].copy()

	# Number of years the author was published in
	valid_paper_years = paper_join[paper_join["paper_year"] != 0]
	author_years = valid_paper_years.groupby("author_id")["paper_year"]

	num_paper_years = author_years.nunique().reset_index().rename(columns={"paper_year":"author_year_count"})
	year_features = pd.merge(year_features, num_paper_years, how="left", on="author_id")
	year_features["author_year_count"] = year_features["author_year_count"].fillna(0).astype(int)

	# Year Frequency Stats
	year_counts = valid_paper_years.groupby(["author_id", "paper_year"]).size().to_frame().reset_index().rename(columns={0:"author_year_count"})
	year_count_groups = year_counts.groupby("author_id")["author_year_count"]

	year_min = year_count_groups.min().rename("author_year_min_freq")
	year_max = year_count_groups.max().rename("author_year_max_freq")
	year_mean = year_count_groups.mean().rename("author_year_mean_freq")
	year_median = year_count_groups.median().rename("author_year_median_freq")
	year_std = year_count_groups.std().rename("author_year_std_freq")

	year_frequency_stats = pd.concat([year_min, year_max, year_mean, year_median, year_std], axis=1).reset_index()
	year_features = pd.merge(year_features, year_frequency_stats, how="left", on="author_id")
	year_features = year_features.fillna(0)

	# Year Difference Stats
	author_min_year = author_years.min().rename("author_min_year")
	author_max_year = author_years.max().rename("author_max_year")
	author_mean_year = author_years.mean().rename("author_mean_year")
	author_median_year = author_years.median().rename("author_median_year")
	author_std_year = author_years.std().rename("author_std_year")

	author_year_stats = pd.concat([author_min_year, author_max_year, author_mean_year, author_median_year, author_std_year], axis=1).reset_index()

	year_features = pd.merge(year_features, author_year_stats, how="left", on="author_id")
	year_features = year_features.fillna(0)

	# year_features["author_min_year_diff"] = year_features["author_min_year"] - year_features["paper_year"]
	# year_features["author_max_year_diff"] = year_features["author_max_year"] - year_features["paper_year"]
	# year_features["author_mean_year_diff"] = year_features["author_mean_pub_year"] - year_features["paper_year"]
	# year_features["author_median_year_diff"] = year_features["author_median_pub_year"] - year_features["paper_year"]

	# year_features = year_features.drop(["paper_year"], axis=1)
	# year_features_merge = year_features[["author_id", "paper_id", "min_year_diff", "max_year_diff", "mean_year_diff", "median_year_diff"]]

	train_out = pd.merge(train_out, year_features, how="left", on=["author_id", "paper_id"])

	return train_out


def paper_features(author_join, paper_join, train_out):
	print "Generating paper features"
	paper_features = paper_join[["author_id", "paper_id"]].copy()

	paper_features["has_paper_title"] = np.where(paper_join["paper_title"] == "", 0, 1)
	paper_features["has_paper_year"] = np.where(paper_join["paper_year"] == 0, 0, 1)
	paper_features["has_conference_id"] = np.where(paper_join["conference_id"] == 0, 0, 1)
	paper_features["has_journal_id"] = np.where(paper_join["journal_id"] == 0, 0, 1)
	paper_features["has_paper_keyword"] = np.where(paper_join["paper_keyword"] == "", 0, 1)

	# Total number of authors on this paper
	valid_author_ids = paper_join[paper_join["author_id"] != 0]
	paper_author_count = valid_author_ids.groupby("paper_id")["author_id"].nunique().reset_index().rename(columns={"author_id":"paper_author_count"})
	paper_features = pd.merge(paper_features, paper_author_count, how="left", on="paper_id")
	paper_features["paper_author_count"] = paper_features["paper_author_count"].fillna(0).astype(int)

	# Total number of papers by authors on this paper
	valid_paper_ids = author_join[author_join["paper_id"] != 0]
	author_paper_count = valid_paper_ids.groupby("author_id")["paper_id"].nunique().reset_index().rename(columns={"paper_id":"author_paper_count"})

	paper_author_groups = valid_author_ids[["paper_id", "author_id"]].copy()
	paper_author_groups = pd.merge(paper_author_groups, author_paper_count, how="left", on="author_id")
	paper_author_groups["author_paper_count"] = paper_author_groups["author_paper_count"].fillna(0).astype(int)
	total_author_paper_count = paper_author_groups.groupby("paper_id")["author_paper_count"].sum().reset_index().rename(columns={"author_paper_count":"paper_total_author_count"})

	paper_features = pd.merge(paper_features, total_author_paper_count, how="left", on="paper_id")
	paper_features["paper_total_author_count"] = paper_features["paper_total_author_count"].fillna(0).astype(int)

	train_out = pd.merge(train_out, paper_features, how="left", on=["paper_id","author_id"])

	return train_out

def paper_journal_features(author_join, paper_join, train_out):
	print "Generating paper journal features"

	paper_journal_features = paper_join[["author_id", "paper_id"]].copy()

	valid_author_ids = paper_join[paper_join["author_id"] != 0]
	valid_journal_ids = paper_join[paper_join["journal_id"] != 0]

	# Total number of journals the authors of this paper have been in
	author_journal_count = valid_journal_ids.groupby("author_id")["journal_id"].nunique().reset_index().rename(columns={"journal_id":"author_journal_count"})
	paper_author_groups = valid_author_ids[["paper_id", "author_id"]].copy()
	paper_author_groups = pd.merge(paper_author_groups, author_journal_count, how="left", on="author_id")
	paper_author_groups["author_journal_count"] = paper_author_groups["author_journal_count"].fillna(0).astype(int)
	total_author_journal_count = paper_author_groups.groupby("paper_id")["author_journal_count"].sum().reset_index().rename(columns={"author_journal_count":"paper_author_journal_count"})

	paper_journal_features = pd.merge(paper_journal_features, total_author_journal_count, how="left", on="paper_id")
	paper_journal_features["paper_author_journal_count"] = paper_journal_features["paper_author_journal_count"].fillna(0).astype(int)

	# Total number of papers in journals by all authors on current paper
	author_journal_paper_count = valid_journal_ids.groupby("author_id")["paper_id"].nunique().reset_index().rename(columns={"paper_id":"author_journal_paper_count"})
	paper_author_groups = valid_author_ids[["paper_id", "author_id"]].copy()
	paper_author_groups = pd.merge(paper_author_groups, author_journal_paper_count, how="left", on="author_id")
	paper_author_groups["author_journal_paper_count"] = paper_author_groups["author_journal_paper_count"].fillna(0).astype(int)
	total_author_journal_paper_count = paper_author_groups.groupby("paper_id")["author_journal_paper_count"].sum().reset_index().rename(columns={"author_journal_paper_count":"paper_author_journal_paper_count"})

	paper_journal_features = pd.merge(paper_journal_features, total_author_journal_paper_count, how="left", on="paper_id")
	paper_journal_features["paper_author_journal_paper_count"] = paper_journal_features["paper_author_journal_paper_count"].fillna(0).astype(int)

	# Total number of papers published by all authors in same journal as current paper
	valid_paper_ids = author_join[author_join["paper_id"] != 0]
	author_paper_count = valid_paper_ids.groupby("author_id")["paper_id"].nunique().reset_index().rename(columns={"paper_id":"author_paper_count"})

	journal_author_groups = valid_journal_ids[["journal_id", "author_id"]].copy().sort_values(by="journal_id")
	journal_author_groups = pd.merge(journal_author_groups, author_paper_count, how="left", on="author_id")
	journal_author_groups["author_paper_count"] = journal_author_groups["author_paper_count"].fillna(0).astype(int)
	journal_author_count = journal_author_groups.groupby("journal_id")["author_paper_count"].sum().reset_index().rename(columns={"author_paper_count":"paper_journal_author_count"})

	journal_author_count_merge = paper_join[["author_id", "paper_id", "journal_id"]].copy()
	journal_author_count_merge = pd.merge(journal_author_count_merge, journal_author_count, how="left", on="journal_id")
	journal_author_count_merge["paper_journal_author_count"] = journal_author_count_merge["paper_journal_author_count"].fillna(0).astype(int)
	journal_author_count_merge = journal_author_count_merge.drop("journal_id", axis=1)

	paper_journal_features = pd.merge(paper_journal_features, journal_author_count_merge, how="left", on=["paper_id","author_id"])

	train_out = pd.merge(train_out, paper_journal_features, how="left", on=["paper_id","author_id"])

	return train_out

def paper_conference_features(author_join, paper_join, train_out):
	print "Generating paper conference features"

	paper_conference_features = paper_join[["author_id", "paper_id"]].copy()

	valid_author_ids = paper_join[paper_join["author_id"] != 0]
	valid_conference_ids = paper_join[paper_join["conference_id"] != 0]

	# Total number of conferences the authors of this paper have been in
	author_conference_count = valid_conference_ids.groupby("author_id")["conference_id"].nunique().reset_index().rename(columns={"conference_id":"author_conference_count"})
	paper_author_groups = valid_author_ids[["paper_id", "author_id"]].copy()
	paper_author_groups = pd.merge(paper_author_groups, author_conference_count, how="left", on="author_id")
	paper_author_groups["author_conference_count"] = paper_author_groups["author_conference_count"].fillna(0).astype(int)
	total_author_conference_count = paper_author_groups.groupby("paper_id")["author_conference_count"].sum().reset_index().rename(columns={"author_conference_count":"paper_author_conference_count"})

	paper_conference_features = pd.merge(paper_conference_features, total_author_conference_count, how="left", on="paper_id")
	paper_conference_features["paper_author_conference_count"] = paper_conference_features["paper_author_conference_count"].fillna(0).astype(int)

	# Total number of papers in conferences by all authors on current paper
	author_conference_paper_count = valid_conference_ids.groupby("author_id")["paper_id"].nunique().reset_index().rename(columns={"paper_id":"author_conference_paper_count"})
	paper_author_groups = valid_author_ids[["paper_id", "author_id"]].copy()
	paper_author_groups = pd.merge(paper_author_groups, author_conference_paper_count, how="left", on="author_id")
	paper_author_groups["author_conference_paper_count"] = paper_author_groups["author_conference_paper_count"].fillna(0).astype(int)
	total_author_conference_paper_count = paper_author_groups.groupby("paper_id")["author_conference_paper_count"].sum().reset_index().rename(columns={"author_conference_paper_count":"paper_author_conference_paper_count"})

	paper_conference_features = pd.merge(paper_conference_features, total_author_conference_paper_count, how="left", on="paper_id")
	paper_conference_features["paper_author_conference_paper_count"] = paper_conference_features["paper_author_conference_paper_count"].fillna(0).astype(int)

	# Total number of papers published by all authors in same conference as current paper
	valid_paper_ids = author_join[author_join["paper_id"] != 0]
	author_paper_count = valid_paper_ids.groupby("author_id")["paper_id"].nunique().reset_index().rename(columns={"paper_id":"author_paper_count"})

	conference_author_groups = valid_conference_ids[["conference_id", "author_id"]].copy().sort_values(by="conference_id")
	conference_author_groups = pd.merge(conference_author_groups, author_paper_count, how="left", on="author_id")
	conference_author_groups["author_paper_count"] = conference_author_groups["author_paper_count"].fillna(0).astype(int)
	conference_author_count = conference_author_groups.groupby("conference_id")["author_paper_count"].sum().reset_index().rename(columns={"author_paper_count":"paper_conference_author_count"})

	conference_author_count_merge = paper_join[["author_id", "paper_id", "conference_id"]].copy()
	conference_author_count_merge = pd.merge(conference_author_count_merge, conference_author_count, how="left", on="conference_id")
	conference_author_count_merge["paper_conference_author_count"] = conference_author_count_merge["paper_conference_author_count"].fillna(0).astype(int)
	conference_author_count_merge = conference_author_count_merge.drop("conference_id", axis=1)

	paper_conference_features = pd.merge(paper_conference_features, conference_author_count_merge, how="left", on=["paper_id","author_id"])

	train_out = pd.merge(train_out, paper_conference_features, how="left", on=["paper_id","author_id"])

	return train_out

def paper_year_features(author_join, paper_join, train_out):
	print "Generating paper year features"

	paper_year_features = paper_join[["author_id", "paper_id"]].copy()

	valid_paper_years = paper_join[paper_join["paper_year"] != 0]
	# Total number of papers by authors in same year as year of paper
	author_paper_years = valid_paper_years.groupby(["author_id", "paper_year"]).size().to_frame().reset_index().rename(columns={0:"author_paper_year_count"})

	total_year_count_groups = valid_paper_years[["paper_id", "author_id", "paper_year"]].copy()
	total_year_count_groups = pd.merge(total_year_count_groups, author_paper_years, how="left", on=["author_id", "paper_year"])
	paper_author_year_count = total_year_count_groups.groupby("paper_id")["author_paper_year_count"].sum().reset_index().rename(columns={"author_paper_year_count":"paper_author_year_count"})

	paper_year_features = pd.merge(paper_year_features, paper_author_year_count, how="left", on="paper_id")
	paper_year_features["paper_author_year_count"] = paper_year_features["paper_author_year_count"].fillna(0).astype(int)

	train_out = pd.merge(train_out, paper_year_features, how="left", on=["paper_id","author_id"])

	return train_out


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


out_columns = ["author_id", "paper_id"]

train_out = author_features(author_join, train_out)
author_feature_list = [
	"has_author_name",
	"has_author_affiliation",
	"author_paper_count"
]
out_columns += author_feature_list

train_out = author_journal_features(author_join, paper_join, train_out)
author_journal_feature_list = [
	"author_journal_count",
	"author_journal_paper_count",
	"author_journal_paper_fraction",
    "author_journal_min_freq",
    "author_journal_max_freq",
    "author_journal_mean_freq",
    "author_journal_median_freq",
    "author_journal_std_freq"
]
out_columns += author_journal_feature_list

train_out = author_conference_features(author_join, paper_join, train_out)
author_conference_feature_list = [
	"author_conference_count",
    "author_conference_paper_count",
	"author_conference_paper_fraction",
    "author_conference_min_freq",
    "author_conference_max_freq",
    "author_conference_mean_freq",
    "author_conference_median_freq",
    "author_conference_std_freq"
]
out_columns += author_conference_feature_list

train_out["author_total_paper_fraction"] = train_out["author_journal_paper_fraction"] + train_out["author_conference_paper_fraction"]
out_columns += ["author_total_paper_fraction"]

train_out = author_affiliation_features(author_info, author_join, train_out)
author_affiliation_feature_list = [
	"author_same_affiliation_count",
    "author_affiliation_lev_dist",
	"author_affiliation_jaro_dist"
]
out_columns += author_affiliation_feature_list

train_out = author_year_features(paper_join, train_out)
author_year_feature_list = [
	"author_year_count",
	"author_year_min_freq",
	"author_year_max_freq",
	"author_year_mean_freq",
	"author_year_median_freq",
	"author_year_std_freq",
	"author_min_year",
	"author_max_year",
	"author_mean_year",
	"author_median_year",
	"author_std_year"
]
out_columns += author_year_feature_list

train_out = paper_features(author_join, paper_join, train_out)
paper_feature_list = [
	"has_paper_title",
	"has_paper_year",
	"has_conference_id",
	"has_journal_id",
	"has_paper_keyword",
	"paper_author_count",
	"paper_total_author_count"
]
out_columns += paper_feature_list

train_out = paper_journal_features(author_join, paper_join, train_out)
paper_journal_feature_list = [
	"paper_author_journal_count",
	"paper_author_journal_paper_count",
	"paper_journal_author_count"
]
out_columns += paper_journal_feature_list

train_out = paper_conference_features(author_join, paper_join, train_out)
paper_conference_feature_list = [
	"paper_author_conference_count",
	"paper_author_conference_paper_count",
	"paper_conference_author_count"
]
out_columns += paper_conference_feature_list

train_out = paper_year_features(author_join, paper_join, train_out)
paper_year_feature_list = [
	"paper_author_year_count"
]
out_columns += paper_year_feature_list

out_columns += ["wrote_paper"]

# train_out.pkl("./pkl/train_features.pkl")

train_out.sort_values(by="author_id").to_csv("./TrainOut.csv", index=False, 
	columns=out_columns)

