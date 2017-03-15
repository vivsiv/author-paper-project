import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import numpy as np
import pandas as pd
import pickle
import jellyfish as jf

def author_features(author_join, train_out):
	print "Generating author features"	
	author_features = train_out[["author_id", "paper_id"]].copy()

	# 1. If the author has a name
	author_join["has_name"] = np.where(author_join["author_name"] == "", 0, 1)

	# 2. If the author has an affiliation
	author_join["has_affiliation"] = np.where(author_join["author_affiliation"] == "", 0, 1)

	author_features = pd.merge(author_features, author_join[["author_id", "paper_id", "has_name", "has_affiliation"]], how="left", on=["author_id", "paper_id"])
	author_features = author_features.fillna(0)

	# 3. Total number of papers by the author
	valid_paper_ids = author_join[author_join["paper_id"] != 0]
	author_paper_count = valid_paper_ids.groupby("author_id")["paper_id"].nunique().reset_index(name="paper_count")
	
	author_features = pd.merge(author_features, author_paper_count, how="left", on="author_id")
	author_features["paper_count"] = author_features["paper_count"].fillna(0).astype(int)
	author_features.rename(
		columns={
			"has_name":"af1",
			"has_affiliation":"af2",
			"paper_count":"af3"
		}, 
		inplace=True)

	return author_features

def author_journal_features(author_join, paper_join, train_out):
	print "Generating author journal features"
	author_journal_features = train_out[["author_id", "paper_id"]].copy()

	author_paper_count = author_join[author_join["paper_id"] != 0].groupby("author_id")["paper_id"].nunique().reset_index(name="paper_count")
	valid_journal_ids = paper_join[paper_join["journal_id"] != 0]

	# 1. Number of journals the author had papers in
	journal_count = valid_journal_ids.groupby("author_id")["journal_id"].nunique().reset_index(name="journal_count")
	author_journal_features = pd.merge(author_journal_features, journal_count, how="left", on="author_id")
	author_journal_features["journal_count"] = author_journal_features["journal_count"].fillna(0).astype(int)

	# 2. Number of papers by the author in journals
	paper_in_journal_count = valid_journal_ids.groupby("author_id")["paper_id"].nunique().reset_index(name="paper_in_journal_count")
	author_journal_features = pd.merge(author_journal_features, paper_in_journal_count, how="left", on="author_id")
	author_journal_features["paper_in_journal_count"] = author_journal_features["paper_in_journal_count"].fillna(0).astype(int)

	# 3. Fraction of the author's papers that are in journals
	journal_fraction = pd.merge(author_paper_count, paper_in_journal_count, how="left", on="author_id")
	journal_fraction["paper_in_journal_count"] = journal_fraction["paper_in_journal_count"].fillna(0).astype(float)
	journal_fraction["paper_count"] = journal_fraction["paper_count"].astype(float)
	journal_fraction["journal_paper_fraction"] = journal_fraction["paper_in_journal_count"] / journal_fraction["paper_count"]

	author_journal_features = pd.merge(author_journal_features, journal_fraction[["author_id", "journal_paper_fraction"]], how="left", on="author_id")
	author_journal_features["journal_paper_fraction"] = author_journal_features["journal_paper_fraction"].fillna(0).round(3)

	# 4-8. Stats on the number of papers the author has in a single journal
	journal_counts = valid_journal_ids.groupby(["author_id", "journal_id"]).size().to_frame().reset_index().rename(columns={0:"paper_in_journal_count"})
	journal_count_groups = journal_counts.groupby("author_id")["paper_in_journal_count"]

	journal_stats = pd.concat(
		[journal_count_groups.min().rename("min_paper_in_journal_count"), 
		 journal_count_groups.max().rename("max_paper_in_journal_count"),
		 journal_count_groups.mean().round(3).rename("mean_paper_in_journal_count"),
		 journal_count_groups.median().rename("median_paper_in_journal_count"),
		 journal_count_groups.std().round(3).rename("dev_paper_in_journal_count")], 
		axis=1).reset_index()

	author_journal_features = pd.merge(author_journal_features, journal_stats, how="left", on="author_id")
	author_journal_features = author_journal_features.fillna(0)
	author_journal_features.rename(
		columns={
			"journal_count":"ajf1",
			"paper_in_journal_count":"ajf2",
			"journal_paper_fraction":"ajf3",
			"min_paper_in_journal_count":"ajf4",
			"max_paper_in_journal_count":"ajf5",
			"mean_paper_in_journal_count":"ajf6",
			"median_paper_in_journal_count":"ajf7",
			"dev_paper_in_journal_count":"ajf8"
		}, 
		inplace=True)
	return author_journal_features

def author_conference_features(author_join, paper_join, train_out):
	print "Generating author conference features"
	author_conference_features = train_out[["author_id", "paper_id"]].copy()

	author_paper_count = author_join[author_join["paper_id"] != 0].groupby("author_id")["paper_id"].nunique().reset_index(name="paper_count")
	valid_conference_ids = paper_join[paper_join["conference_id"] != 0]

	# 1. Number of conferences the author had papers in
	conference_count = valid_conference_ids.groupby("author_id")["conference_id"].nunique().reset_index(name="conference_count")
	author_conference_features = pd.merge(author_conference_features, conference_count, how="left", on="author_id")
	author_conference_features["conference_count"] = author_conference_features["conference_count"].fillna(0).astype(int)

	# 2. Number of papers by the author in conferences
	paper_in_conference_count = valid_conference_ids.groupby("author_id")["paper_id"].nunique().reset_index(name="paper_in_conference_count")
	author_conference_features = pd.merge(author_conference_features, paper_in_conference_count, how="left", on="author_id")
	author_conference_features["paper_in_conference_count"] = author_conference_features["paper_in_conference_count"].fillna(0).astype(int)

	# 3. Fraction of the author's papers that are in conferences
	conference_fraction = pd.merge(author_paper_count, paper_in_conference_count, how="left", on="author_id")
	conference_fraction["paper_in_conference_count"] = conference_fraction["paper_in_conference_count"].fillna(0).astype(float)
	conference_fraction["paper_count"] = conference_fraction["paper_count"].astype(float)
	conference_fraction["conference_paper_fraction"] = conference_fraction["paper_in_conference_count"] / conference_fraction["paper_count"]

	author_conference_features = pd.merge(author_conference_features, conference_fraction[["author_id", "conference_paper_fraction"]], how="left", on="author_id")
	author_conference_features["conference_paper_fraction"] = author_conference_features["conference_paper_fraction"].fillna(0).round(3)

	# 4-8. Stats on the number of papers the author has in a single conference
	conference_counts = valid_conference_ids.groupby(["author_id", "conference_id"]).size().to_frame().reset_index().rename(columns={0:"paper_in_conference_count"})
	conference_count_groups = conference_counts.groupby("author_id")["paper_in_conference_count"]

	conference_stats = pd.concat(
		[conference_count_groups.min().rename("min_paper_in_conference_count"), 
		 conference_count_groups.max().rename("max_paper_in_conference_count"),
		 conference_count_groups.mean().round(3).rename("mean_paper_in_conference_count"),
		 conference_count_groups.median().rename("median_paper_in_conference_count"),
		 conference_count_groups.std().round(3).rename("dev_paper_in_conference_count")], 
		axis=1).reset_index()

	author_conference_features = pd.merge(author_conference_features, conference_stats, how="left", on="author_id")
	author_conference_features = author_conference_features.fillna(0)
	author_conference_features.rename(
		columns={
			"conference_count":"acf1",
			"paper_in_conference_count":"acf2",
			"conference_paper_fraction":"acf3",
			"min_paper_in_conference_count":"acf4",
			"max_paper_in_conference_count":"acf5",
			"mean_paper_in_conference_count":"acf6",
			"median_paper_in_conference_count":"acf7",
			"dev_paper_in_conference_count":"acf8"
		}, 
		inplace=True)

	return author_conference_features

def author_affiliation_features(author_join, train_out):
	print "Generating author affiliation features"
	author_affiliation_features = pd.merge(train_out[["author_id","paper_id"]], 
		author_join[["author_id", "paper_id", "author_affiliation_clean", "paper_author_affiliation_clean"]], 
		how="left", on=["author_id", "paper_id"])

	# 1. Number of authors with the same affiliation in Author.csv
	same_affiliation_count_author = author_join[author_join["author_affiliation_clean"] != ""].groupby("author_affiliation_clean")["author_id"].nunique().reset_index(name="same_affiliation_count_author")
	author_affiliation_features = pd.merge(author_affiliation_features, same_affiliation_count_author, how="left", on="author_affiliation_clean")
	author_affiliation_features["same_affiliation_count_author"] = author_affiliation_features["same_affiliation_count_author"].fillna(0).astype(int)
	

	# 2-3. String Distance between affiliation in Author.csv and affiliation in PaperAuthor.csv
	author_affiliation_features["affiliation_lev_dist"] = author_affiliation_features.apply(
		lambda row: jf.levenshtein_distance(unicode(row["author_affiliation_clean"]), unicode(row["paper_author_affiliation_clean"])), 
		axis=1)
	author_affiliation_features["affiliation_jaro_dist"] = author_affiliation_features.apply(
		lambda row: jf.jaro_distance(unicode(row["author_affiliation_clean"]), unicode(row["paper_author_affiliation_clean"])), 
		axis=1)
	author_affiliation_features["affiliation_lev_dist"] = author_affiliation_features["affiliation_lev_dist"].fillna(0).astype(int)
	author_affiliation_features["affiliation_jaro_dist"] = author_affiliation_features["affiliation_jaro_dist"].fillna(0.0).round(3)

	# 4. Number of with the same affiliation in PaperAuthor.csv
	same_affiliation_count_paper = author_join[author_join["paper_author_affiliation_clean"] != ""].groupby("paper_author_affiliation_clean")["author_id"].nunique().reset_index(name="same_affiliation_count_paper")
	author_affiliation_features = pd.merge(author_affiliation_features, same_affiliation_count_paper, how="left", on="paper_author_affiliation_clean")
	author_affiliation_features["same_affiliation_count_paper"] = author_affiliation_features["same_affiliation_count_paper"].fillna(0).astype(int)

	author_affiliation_features.drop(["author_affiliation_clean", "paper_author_affiliation_clean"], axis=1, inplace=True)

	author_affiliation_features.rename(
		columns={
			"same_affiliation_count_author":"aaf1",
			"affiliation_lev_dist":"aaf2",
			"affiliation_jaro_dist":"aaf3",
			"same_affiliation_count_paper":"aaf4"
		}, 
		inplace=True)

	return author_affiliation_features

def author_year_features(paper_join, train_out):
	# Author Publication Year Features
	print "Generating author year features"
	author_year_features = train_out[["author_id", "paper_id"]].copy()

	valid_paper_years = paper_join[paper_join["paper_year"] != 0]
	author_years = valid_paper_years.groupby("author_id")["paper_year"]
	
	# 1. The number of years the author was published in
	year_count = author_years.nunique().reset_index(name="year_count")
	author_year_features = pd.merge(author_year_features, year_count, how="left", on="author_id")
	author_year_features["year_count"] = author_year_features["year_count"].fillna(0).astype(int)

	# 2-6. Stats on the years the author was published in
	year_stats = pd.concat(
		[author_years.min().rename("min_pub_year"), 
		 author_years.max().rename("max_pub_year"), 
		 author_years.mean().round(3).rename("mean_year"), 
		 author_years.median().rename("median_pub_year"), 
		 author_years.std().round(3).rename("dev_pub_year")], 
		axis=1).reset_index()

	author_year_features = pd.merge(author_year_features, year_stats, how="left", on="author_id")
	author_year_features = author_year_features.fillna(0)

	# 7-11. Stats on the number of papers the author had published in a single year
	year_counts = valid_paper_years.groupby(["author_id", "paper_year"]).size().to_frame().reset_index().rename(columns={0:"paper_in_year_count"})
	year_count_groups = year_counts.groupby("author_id")["paper_in_year_count"]

	year_count_stats = pd.concat(
		[year_count_groups.min().rename("min_paper_in_year_count"), 
		 year_count_groups.max().rename("max_paper_in_year_count"), 
		 year_count_groups.mean().round(3).rename("mean_paper_in_year_count"), 
		 year_count_groups.median().rename("median_paper_in_year_count"), 
		 year_count_groups.std().round(3).rename("dev_paper_in_year_count")], 
		axis=1).reset_index()

	author_year_features = pd.merge(author_year_features, year_count_stats, how="left", on="author_id")
	author_year_features = author_year_features.fillna(0)

	author_year_features.rename(
		columns={
			"year_count":"ayf1",
			"min_pub_year":"ayf2", 
		    "max_pub_year":"ayf3", 
		    "mean_year":"ayf4", 
		    "median_pub_year":"ayf5", 
		    "dev_pub_year":"ayf6",
		    "min_paper_in_year_count":"ayf7", 
		    "max_paper_in_year_count":"ayf8", 
		    "mean_paper_in_year_count":"ayf9", 
		    "median_paper_in_year_count":"ayf10", 
			"dev_paper_in_year_count":"ayf11"
		}, 
		inplace=True)

	return author_year_features

def author_name_features(author_join, train_out):
	print "Generating author name features"
	author_name_splits =  author_join["author_name_clean"].str.split(' ', 1, expand=True)
	author_join["author_first_name"] = author_name_splits[0].fillna("")
	author_join["author_last_name"] = author_name_splits[1].fillna("").apply(lambda name: name.split(" ")[-1])

	paper_author_name_splits =  author_join["paper_author_name_clean"].str.split(' ', 1, expand=True)
	author_join["paper_author_first_name"] = paper_author_name_splits[0].fillna("")
	author_join["paper_author_last_name"] = paper_author_name_splits[1].fillna("").apply(lambda name: name.split(" ")[-1])

	author_name_features = pd.merge(train_out[["author_id","paper_id"]], 
		author_join[["author_id", "paper_id", "author_name_clean", "author_first_name", "author_last_name","paper_author_name_clean", "paper_author_first_name", "paper_author_last_name"]], 
		how="left", on=["author_id", "paper_id"])


	# 1. Authors with the same last name in Author.csv
	same_last_name_count_author = author_join.groupby("author_last_name")["author_id"].nunique().reset_index(name="same_last_name_count_author")
	author_name_features = pd.merge(author_name_features, same_last_name_count_author, how="left", on="author_last_name")
	author_name_features["same_last_name_count_author"] = author_name_features["same_last_name_count_author"].fillna(0).astype(int)

	# 2. Authors with the same last name in PaperAuthor.csv
	same_last_name_count_paper = author_join.groupby("paper_author_last_name")["author_id"].nunique().reset_index(name="same_last_name_count_paper")
	author_name_features = pd.merge(author_name_features, same_last_name_count_paper, how="left", on="paper_author_last_name")
	author_name_features["same_last_name_count_paper"] = author_name_features["same_last_name_count_paper"].fillna(0).astype(int)

	# 3-4. Distance between names in Author.csv and PaperAuthor.csv
	author_name_features["name_lev_dist"] = author_name_features.apply(
		lambda row: jf.levenshtein_distance(unicode(row["author_name_clean"]), unicode(row["paper_author_name_clean"])), 
		axis=1)
	author_name_features["name_jaro_dist"] = author_name_features.apply(
		lambda row: jf.jaro_distance(unicode(row["author_name_clean"]), unicode(row["paper_author_name_clean"])), 
		axis=1)
	author_name_features["name_lev_dist"] = author_name_features["name_lev_dist"].fillna(0).astype(int)
	author_name_features["name_jaro_dist"] = author_name_features["name_jaro_dist"].fillna(0.0).round(3)

	# 5-6. Distance between last names in Author.csv and PaperAuthor.csv
	author_name_features["last_name_lev_dist"] = author_name_features.apply(
		lambda row: jf.levenshtein_distance(unicode(row["author_last_name"]), unicode(row["paper_author_last_name"])), 
		axis=1)
	author_name_features["last_name_jaro_dist"] = author_name_features.apply(
		lambda row: jf.jaro_distance(unicode(row["author_last_name"]), unicode(row["paper_author_last_name"])), 
		axis=1)
	author_name_features["last_name_lev_dist"] = author_name_features["last_name_lev_dist"].fillna(0).astype(int)
	author_name_features["last_name_jaro_dist"] = author_name_features["last_name_jaro_dist"].fillna(0.0).round(3)

	# 7-8. Distance between first names in Author.csv and PaperAuthor.csv
	author_name_features["first_name_lev_dist"] = author_name_features.apply(
		lambda row: jf.levenshtein_distance(unicode(row["author_first_name"]), unicode(row["paper_author_first_name"])), 
		axis=1)
	author_name_features["first_name_jaro_dist"] = author_name_features.apply(
		lambda row: jf.jaro_distance(unicode(row["author_first_name"]), unicode(row["paper_author_first_name"])), 
		axis=1)
	author_name_features["first_name_lev_dist"] = author_name_features["first_name_lev_dist"].fillna(0).astype(int)
	author_name_features["first_name_jaro_dist"] = author_name_features["first_name_jaro_dist"].fillna(0.0).round(3)

	author_name_features.drop(["author_name_clean", "author_first_name", "author_last_name","paper_author_name_clean", "paper_author_first_name", "paper_author_last_name"], axis=1, inplace=True)

	author_name_features.rename(
		columns={
			"same_last_name_count_author":"anf1",
			"same_last_name_count_paper":"anf2",
			"name_lev_dist":"anf3",
			"name_jaro_dist":"anf4",
			"last_name_lev_dist":"anf5",
			"last_name_jaro_dist":"anf6",
			"first_name_lev_dist":"anf7",
			"first_name_jaro_dist":"anf8",
		}, 
		inplace=True)

	return author_name_features


def paper_features(author_join, paper_join, train_out):
	print "Generating paper features"
	paper_features = train_out[["author_id", "paper_id"]].copy()

	# 1 - 6. If the paper has a title, year, conference_id, journal_id, keyword
	paper_join["has_title"] = np.where(paper_join["paper_title"] == "", 0, 1)
	paper_join["has_paper_year"] = np.where(paper_join["paper_year"] == 0, 0, 1)
	paper_join["has_conference_id"] = np.where(paper_join["conference_id"] == 0, 0, 1)
	paper_join["has_journal_id"] = np.where(paper_join["journal_id"] == 0, 0, 1)
	paper_join["has_keyword"] = np.where(paper_join["paper_keyword"] == "", 0, 1)

	paper_features = pd.merge(paper_features, paper_join[["paper_id", "author_id", "paper_year", "has_title", "has_paper_year", "has_conference_id", "has_journal_id", "has_keyword"]], how="left", on=["paper_id", "author_id"])
	paper_features = paper_features.fillna(0)

	valid_author_ids = paper_join[paper_join["author_id"] != 0]

	# 7. Number of authors of this paper
	paper_author_count = valid_author_ids.groupby("paper_id")["author_id"].nunique().reset_index(name="author_count")
	paper_features = pd.merge(paper_features, paper_author_count, how="left", on="paper_id")
	paper_features["author_count"] = paper_features["author_count"].fillna(0).astype(int)

	author_paper_count = author_join[author_join["paper_id"] != 0].groupby("author_id")["paper_id"].nunique().reset_index(name="paper_count")

	
	paper_author_counts = pd.merge(valid_author_ids[["paper_id", "author_id"]], author_paper_count, how="left", on="author_id")
	paper_author_counts["paper_count"] = paper_author_counts["paper_count"].fillna(0).astype(int)
	paper_author_groups = paper_author_counts.groupby("paper_id")["paper_count"]

	# 8. Total number of papers written by authors of this paper
	authors_total_paper_count = paper_author_groups.sum().reset_index().rename(columns={"paper_count":"authors_total_paper_count"})

	paper_features = pd.merge(paper_features, authors_total_paper_count, how="left", on="paper_id")
	paper_features["authors_total_paper_count"] = paper_features["authors_total_paper_count"].fillna(0).astype(int)

	# 9 - 13. Stats about number of papers written by authors of this paper
	author_stats = pd.concat(
		[paper_author_groups.min().rename("min_author_paper_count"), 
		 paper_author_groups.max().rename("max_author_paper_count"), 
		 paper_author_groups.mean().round(3).rename("mean_author_paper_count"), 
		 paper_author_groups.median().rename("median_author_paper_count"), 
		 paper_author_groups.std().round(3).rename("dev_author_paper_count")], 
		axis=1).reset_index()

	paper_features = pd.merge(paper_features, author_stats, how="left", on="paper_id")
	paper_features = paper_features.fillna(0)

	paper_features.rename(
		columns={
			"paper_year":"pf1",
			"has_title":"pf2",
			"has_paper_year":"pf3",
			"has_conference_id":"pf4",
			"has_journal_id":"pf5",
			"has_keyword":"pf6",
			"author_count":"pf7",
			"authors_total_paper_count":"pf8",
			"min_author_paper_count":"pf9", 
		 	"max_author_paper_count":"pf10", 
			"mean_author_paper_count":"pf11", 
			"median_author_paper_count":"pf12", 
			"dev_author_paper_count":"pf13"
		}, 
		inplace=True)

	return paper_features

def paper_journal_features(author_join, paper_join, train_out):
	print "Generating paper journal features"

	paper_journal_features = train_out[["author_id", "paper_id"]].copy()

	valid_author_ids = paper_join[paper_join["author_id"] != 0]
	valid_journal_ids = paper_join[paper_join["journal_id"] != 0]

	author_journal_counts = valid_journal_ids.groupby("author_id")["journal_id"].nunique().reset_index(name="journal_count")
	author_journal_counts = pd.merge(valid_author_ids[["paper_id", "author_id"]], author_journal_counts, how="left", on="author_id")
	author_journal_counts["journal_count"] = author_journal_counts["journal_count"].fillna(0).astype(int)
	journal_count_groups = author_journal_counts.groupby("paper_id")["journal_count"]

	# 1. Total number of journals the authors of this paper have been in
	authors_total_journal_count = journal_count_groups.sum().reset_index(name="authors_total_journal_count")

	paper_journal_features = pd.merge(paper_journal_features, authors_total_journal_count, how="left", on="paper_id")
	paper_journal_features["authors_total_journal_count"] = paper_journal_features["authors_total_journal_count"].fillna(0).astype(int)

	# 2 - 6. Stats on the number of journals the authors of this paper have been in
	journal_stats = pd.concat(
		[journal_count_groups.min().rename("min_author_journal_count"), 
		 journal_count_groups.max().rename("max_author_journal_count"), 
		 journal_count_groups.mean().round(3).rename("mean_author_journal_count"), 
		 journal_count_groups.median().rename("median_author_journal_count"), 
		 journal_count_groups.std().round(3).rename("dev_author_journal_count")], 
		axis=1).reset_index()

	paper_journal_features = pd.merge(paper_journal_features, journal_stats, how="left", on="paper_id")
	paper_journal_features = paper_journal_features.fillna(0)


	# 7. Total number of papers in journals written by all authors of this paper
	journal_paper_counts = valid_journal_ids.groupby("author_id")["paper_id"].nunique().reset_index(name="journal_paper_count")
	journal_paper_counts = pd.merge(valid_author_ids[["paper_id", "author_id"]], journal_paper_counts, how="left", on="author_id")
	journal_paper_counts["journal_paper_count"] = journal_paper_counts["journal_paper_count"].fillna(0).astype(int)
	journal_paper_count_groups = journal_paper_counts.groupby("paper_id")["journal_paper_count"]
	authors_total_journal_paper_count = journal_paper_count_groups.sum().reset_index().rename(columns={"journal_paper_count":"authors_total_journal_paper_count"})

	paper_journal_features = pd.merge(paper_journal_features, authors_total_journal_paper_count, how="left", on="paper_id")
	paper_journal_features["authors_total_journal_paper_count"] = paper_journal_features["authors_total_journal_paper_count"].fillna(0).astype(int)

	# 8 - 12. Stats on the number of papers in journals written by authors of this paper
	journal_paper_stats = pd.concat(
		[journal_paper_count_groups.min().rename("min_author_journal_paper_count"), 
		 journal_paper_count_groups.max().rename("max_author_journal_paper_count"), 
		 journal_paper_count_groups.mean().round(3).rename("mean_author_journal_paper_count"), 
		 journal_paper_count_groups.median().rename("median_author_journal_paper_count"), 
		 journal_paper_count_groups.std().round(3).rename("dev_author_journal_paper_count")], 
		axis=1).reset_index()

	paper_journal_features = pd.merge(paper_journal_features, journal_paper_stats, how="left", on="paper_id")
	paper_journal_features = paper_journal_features.fillna(0)

	# Total number of papers published by all authors in same journal as current paper
	# valid_paper_ids = author_join[author_join["paper_id"] != 0]
	# author_paper_count = valid_paper_ids.groupby("author_id")["paper_id"].nunique().reset_index().rename(columns={"paper_id":"author_paper_count"})

	# journal_author_groups = valid_journal_ids[["journal_id", "author_id"]].copy().sort_values(by="journal_id")
	# journal_author_groups = pd.merge(journal_author_groups, author_paper_count, how="left", on="author_id")
	# journal_author_groups["author_paper_count"] = journal_author_groups["author_paper_count"].fillna(0).astype(int)
	# journal_author_count = journal_author_groups.groupby("journal_id")["author_paper_count"].sum().reset_index().rename(columns={"author_paper_count":"paper_journal_author_count"})

	# journal_author_count_merge = paper_join[["author_id", "paper_id", "journal_id"]].copy()
	# journal_author_count_merge = pd.merge(journal_author_count_merge, journal_author_count, how="left", on="journal_id")
	# journal_author_count_merge["paper_journal_author_count"] = journal_author_count_merge["paper_journal_author_count"].fillna(0).astype(int)
	# journal_author_count_merge = journal_author_count_merge.drop("journal_id", axis=1)

	# paper_journal_features = pd.merge(paper_journal_features, journal_author_count_merge, how="left", on=["paper_id","author_id"])

	paper_journal_features.rename(
		columns={
			"authors_total_journal_count":"pjf1",
			"min_author_journal_count":"pjf2", 
			"max_author_journal_count":"pjf3", 
			"mean_author_journal_count":"pjf4", 
			"median_author_journal_count":"pjf5", 
			"dev_author_journal_count":"pjf6", 
			"authors_total_journal_paper_count":"pjf7",
			"min_author_journal_paper_count":"pjf8", 
			"max_author_journal_paper_count":"pjf9", 
		 	"mean_author_journal_paper_count":"pjf10", 
		 	"median_author_journal_paper_count":"pjf11", 
		 	"dev_author_journal_paper_count":"pjf12"
		}, 
		inplace=True)

	return paper_journal_features

def paper_conference_features(author_join, paper_join, train_out):
	print "Generating paper conference features"

	paper_conference_features = train_out[["author_id", "paper_id"]].copy()

	valid_author_ids = paper_join[paper_join["author_id"] != 0]
	valid_conference_ids = paper_join[paper_join["conference_id"] != 0]

	author_conference_counts = valid_conference_ids.groupby("author_id")["conference_id"].nunique().reset_index(name="conference_count")
	author_conference_counts = pd.merge(valid_author_ids[["paper_id", "author_id"]], author_conference_counts, how="left", on="author_id")
	author_conference_counts["conference_count"] = author_conference_counts["conference_count"].fillna(0).astype(int)
	conference_count_groups = author_conference_counts.groupby("paper_id")["conference_count"]

	# 1. Total number of conferences the authors of this paper have been in
	authors_total_conference_count = conference_count_groups.sum().reset_index(name="authors_total_conference_count")

	paper_conference_features = pd.merge(paper_conference_features, authors_total_conference_count, how="left", on="paper_id")
	paper_conference_features["authors_total_conference_count"] = paper_conference_features["authors_total_conference_count"].fillna(0).astype(int)

	# 2 - 6. Stats on the number of conferences the authors of this paper have been in
	conference_stats = pd.concat(
		[conference_count_groups.min().rename("min_author_conference_count"), 
		 conference_count_groups.max().rename("max_author_conference_count"), 
		 conference_count_groups.mean().round(3).rename("mean_author_conference_count"), 
		 conference_count_groups.median().rename("median_author_conference_count"), 
		 conference_count_groups.std().round(3).rename("dev_author_conference_count")], 
		axis=1).reset_index()

	paper_conference_features = pd.merge(paper_conference_features, conference_stats, how="left", on="paper_id")
	paper_conference_features = paper_conference_features.fillna(0)


	# 7. Total number of papers in conferences written by all authors of this paper
	conference_paper_counts = valid_conference_ids.groupby("author_id")["paper_id"].nunique().reset_index(name="conference_paper_count")
	conference_paper_counts = pd.merge(valid_author_ids[["paper_id", "author_id"]], conference_paper_counts, how="left", on="author_id")
	conference_paper_counts["conference_paper_count"] = conference_paper_counts["conference_paper_count"].fillna(0).astype(int)
	conference_paper_count_groups = conference_paper_counts.groupby("paper_id")["conference_paper_count"]
	authors_total_conference_paper_count = conference_paper_count_groups.sum().reset_index().rename(columns={"conference_paper_count":"authors_total_conference_paper_count"})

	paper_conference_features = pd.merge(paper_conference_features, authors_total_conference_paper_count, how="left", on="paper_id")
	paper_conference_features["authors_total_conference_paper_count"] = paper_conference_features["authors_total_conference_paper_count"].fillna(0).astype(int)

	# 8 - 12. Stats on the number of papers in conferences written by authors of this paper
	conference_paper_stats = pd.concat(
		[conference_paper_count_groups.min().rename("min_author_conference_paper_count"), 
		 conference_paper_count_groups.max().rename("max_author_conference_paper_count"), 
		 conference_paper_count_groups.mean().round(3).rename("mean_author_conference_paper_count"), 
		 conference_paper_count_groups.median().rename("median_author_conference_paper_count"), 
		 conference_paper_count_groups.std().round(3).rename("dev_author_conference_paper_count")], 
		axis=1).reset_index()

	paper_conference_features = pd.merge(paper_conference_features, conference_paper_stats, how="left", on="paper_id")
	paper_conference_features = paper_conference_features.fillna(0)

	# Total number of papers published by all authors in same conference as current paper
	# valid_paper_ids = author_join[author_join["paper_id"] != 0]
	# author_paper_count = valid_paper_ids.groupby("author_id")["paper_id"].nunique().reset_index().rename(columns={"paper_id":"author_paper_count"})

	# conference_author_groups = valid_conference_ids[["conference_id", "author_id"]].copy().sort_values(by="conference_id")
	# conference_author_groups = pd.merge(conference_author_groups, author_paper_count, how="left", on="author_id")
	# conference_author_groups["author_paper_count"] = conference_author_groups["author_paper_count"].fillna(0).astype(int)
	# conference_author_count = conference_author_groups.groupby("conference_id")["author_paper_count"].sum().reset_index().rename(columns={"author_paper_count":"paper_conference_author_count"})

	# conference_author_count_merge = paper_join[["author_id", "paper_id", "conference_id"]].copy()
	# conference_author_count_merge = pd.merge(conference_author_count_merge, conference_author_count, how="left", on="conference_id")
	# conference_author_count_merge["paper_conference_author_count"] = conference_author_count_merge["paper_conference_author_count"].fillna(0).astype(int)
	# conference_author_count_merge = conference_author_count_merge.drop("conference_id", axis=1)

	# paper_conference_features = pd.merge(paper_conference_features, conference_author_count_merge, how="left", on=["paper_id","author_id"])

	paper_conference_features.rename(
		columns={
			"authors_total_conference_count":"pcf1",
			"min_author_conference_count":"pcf2", 
			"max_author_conference_count":"pcf3", 
			"mean_author_conference_count":"pcf4", 
			"median_author_conference_count":"pcf5", 
			"dev_author_conference_count":"pcf6", 
			"authors_total_conference_paper_count":"pcf7",
			"min_author_conference_paper_count":"pcf8", 
			"max_author_conference_paper_count":"pcf9", 
		 	"mean_author_conference_paper_count":"pcf10", 
		 	"median_author_conference_paper_count":"pcf11", 
		 	"dev_author_conference_paper_count":"pcf12"
		}, 
		inplace=True)

	return paper_conference_features

def paper_year_features(author_join, paper_join, train_out):
	print "Generating paper year features"
	paper_year_features = train_out[["author_id", "paper_id"]].copy()

	valid_paper_years = paper_join[paper_join["paper_year"] != 0]

	# 1. Total number of papers published by authors of this paper in the same year as the year of this paper
	author_year_paper_counts = valid_paper_years.groupby(["author_id", "paper_year"]).size().to_frame().reset_index().rename(columns={0:"year_paper_count"})

	year_paper_count_groups = pd.merge(valid_paper_years[["paper_id", "author_id", "paper_year"]], author_year_paper_counts, how="left", on=["author_id", "paper_year"])
	year_paper_count_groups = year_paper_count_groups.groupby("paper_id")["year_paper_count"]
	year_total_paper_count = year_paper_count_groups.sum().reset_index(name="year_author_total_paper_count")

	paper_year_features = pd.merge(paper_year_features, year_total_paper_count, how="left", on="paper_id")
	paper_year_features["year_author_total_paper_count"] = paper_year_features["year_author_total_paper_count"].fillna(0).astype(int)

	# 2 - 6. Stats on the number of papers published by authors in the year this paper was published
	year_author_stats = pd.concat(
		[year_paper_count_groups.min().rename("min_author_year_paper_count"), 
		 year_paper_count_groups.max().rename("max_author_year_paper_count"), 
		 year_paper_count_groups.mean().round(3).rename("mean_author_year_paper_count"), 
		 year_paper_count_groups.median().rename("median_author_year_paper_count"), 
		 year_paper_count_groups.std().round(3).rename("dev_author_year_paper_count")], 
		axis=1).reset_index()

	paper_year_features = pd.merge(paper_year_features, year_author_stats, how="left", on="paper_id")
	paper_year_features = paper_year_features.fillna(0)

	paper_year_features.rename(
		columns={
			"year_author_total_paper_count":"pyf1",
			"min_author_year_paper_count":"pyf2", 
			"max_author_year_paper_count":"pyf3", 
		 	"mean_author_year_paper_count":"pyf4", 
			"median_author_year_paper_count":"pyf5", 
			"dev_author_year_paper_count":"pyf6"
		}, 
		inplace=True)

	return paper_year_features



def main():
	print "--- START feature_eng.py ---"

	base = "train"
	if len(sys.argv) > 1:
		if sys.argv[1] == "valid":
			base = "valid"
		elif sys.argv[1] == "test":
			base = "test"
		else:
			base = "train"
	print "Building features for: {0}.csv...".format(base.capitalize())

	save_intermediate = False
	if len(sys.argv) > 2 and sys.argv[2] == "save_intermediate":
		save_intermediate = True
		print "Saving intermediates in:./{0}/ ...".format(base)

	print "Reading author_join"
	author_join = pd.read_pickle("./pkl/author_join.pkl")
	print "Reading paper_join"
	paper_join = pd.read_pickle("./pkl/paper_join.pkl")
	print "Reading {0}_base".format(base)
	train_out = pd.read_pickle("./pkl/{0}_base.pkl".format(base))

	feature_dfs = []

	a_features = author_features(author_join, train_out)
	feature_dfs.append(a_features)
	if save_intermediate:
		a_features.sort_values(by="author_id").to_csv("./{0}/author_features.csv".format(base), index=False, columns=list(a_features.columns.values))
	

	a_journal_features = author_journal_features(author_join, paper_join, train_out)
	feature_dfs.append(a_journal_features.drop(["author_id", "paper_id"], axis=1))
	if save_intermediate:
		a_journal_features.sort_values(by="author_id").to_csv("./{0}/author_journal_features.csv".format(base), index=False, columns=list(a_journal_features.columns.values))
	

	a_conference_features = author_conference_features(author_join, paper_join, train_out)
	feature_dfs.append(a_conference_features.drop(["author_id", "paper_id"], axis=1))
	if save_intermediate:
		a_conference_features.sort_values(by="author_id").to_csv("./{0}/author_conference_features.csv".format(base), index=False, columns=list(a_conference_features.columns.values))
	

	a_affiliation_features = author_affiliation_features(author_join, train_out)
	feature_dfs.append(a_affiliation_features.drop(["author_id", "paper_id"], axis=1))
	if save_intermediate:
		a_affiliation_features.sort_values(by="author_id").to_csv("./{0}/author_affiliation_features.csv".format(base), index=False, columns=list(a_affiliation_features.columns.values))
	

	a_year_features = author_year_features(paper_join, train_out)
	feature_dfs.append(a_year_features.drop(["author_id", "paper_id"], axis=1))
	if save_intermediate:
		a_year_features.sort_values(by="author_id").to_csv("./{0}/author_year_features.csv".format(base), index=False, columns=list(a_year_features.columns.values))
	
	
	a_name_features = author_name_features(author_join, train_out)
	feature_dfs.append(a_name_features.drop(["author_id", "paper_id"], axis=1))
	if save_intermediate:
		a_name_features.sort_values(by="author_id").to_csv("./{0}/author_name_features.csv".format(base), index=False, columns=list(a_name_features.columns.values))
	
	
	p_features = paper_features(author_join, paper_join, train_out)
	feature_dfs.append(p_features.drop(["author_id", "paper_id"], axis=1))
	if save_intermediate:
		p_features.sort_values(by="author_id").to_csv("./{0}/paper_features.csv".format(base), index=False, columns=list(p_features.columns.values))
	

	p_journal_features = paper_journal_features(author_join, paper_join, train_out)
	feature_dfs.append(p_journal_features.drop(["author_id", "paper_id"], axis=1))
	if save_intermediate:
		p_journal_features.sort_values(by="author_id").to_csv("./{0}/paper_journal_features.csv".format(base), index=False, columns=list(p_journal_features.columns.values))
	

	
	p_conference_features = paper_conference_features(author_join, paper_join, train_out)
	feature_dfs.append(p_conference_features.drop(["author_id", "paper_id"], axis=1))
	if save_intermediate:
		p_conference_features.sort_values(by="author_id").to_csv("./{0}/paper_conference_features.csv".format(base), index=False, columns=list(p_conference_features.columns.values))
	

	
	p_year_features = paper_year_features(author_join, paper_join, train_out)
	feature_dfs.append(p_year_features.drop(["author_id", "paper_id"], axis=1))
	if save_intermediate:
		p_year_features.sort_values(by="author_id").to_csv("./{0}/paper_year_features.csv".format(base), index=False, columns=list(p_year_features.columns.values))
	
	
	merged_features = pd.concat(feature_dfs, axis=1)
	train_out = pd.merge(train_out, merged_features, how="left", on=["author_id", "paper_id"])

	if base == "train":
		out_columns = list(train_out.drop(["wrote_paper"], axis=1).columns.values)
		out_columns.append("wrote_paper")
		train_out.sort_values(by="author_id").to_csv("./{0}/{1}Out.csv".format(base, base.capitalize()), index=False, columns=out_columns)
	else:
		train_out.sort_values(by="author_id").to_csv("./{0}/{1}Out.csv".format(base, base.capitalize()), index=False)

	print "--- END feature_eng.py ---"


if __name__ == "__main__": main()

