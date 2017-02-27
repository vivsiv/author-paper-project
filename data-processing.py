import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import numpy as np
import pandas as pd
import pickle
import editdistance as edit

string_replacements = {
	"\\.*\\s+": " ",
	"['\\-]": "",
	"[^a-zA-Z ]": ""
}

def save_obj(obj, out_file_name):
    with open(out_file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(in_file_name):
    with open(in_file_name, 'rb') as in_file:
        return pickle.load(in_file)

def standardize_name(name):
	name_arr = name.split(" ")
	first_name = name_arr[0]
	last_name = name_arr[len(name_arr) - 1].lower()
	middle_name = ""
	if len(name_arr) == 3:
		middle_name = name_arr[1].lower()
	
	first_initial = first_name[:1]
	last_initial = last_name[:1]
	middle_initial = ""
	if middle_name != "":
		middle_initial = middle_name[:1]
	
	name1 = first_name + " " + last_name
	name2 = first_initial + " " + last_name
	name3 = first_name + " " + middle_name + " " + last_name
	name4 = first_name + " " + middle_initial + " " + last_name
	name5 = first_initial + " " + middle_initial + " " + last_name
	
	if middle_name == "":
		name3 = first_name + " " + last_name
		name4 = first_name + " " + last_name
		name5 = first_initial + " " + last_name
	return [name1, name2, name3, name4, name5]

def main():
	train_data = "./data/Train.csv"
	print "Reading Train.csv from {0}".format(train_data)
	train_df = pd.read_csv(train_data)

	train_conf = pd.DataFrame(train_df.ConfirmedPaperIds.str.split(" ").tolist(), index=train_df.AuthorId).stack()
	train_conf = train_conf.reset_index()[['AuthorId', 0]]
	train_conf.columns = ["author_id", "paper_id"]
	train_conf["wrote_paper"] = 1

	train_deleted = pd.DataFrame(train_df.DeletedPaperIds.str.split(" ").tolist(), index=train_df.AuthorId).stack()
	train_deleted = train_deleted.reset_index()[['AuthorId', 0]]
	train_deleted.columns = ["author_id", "paper_id"]
	train_deleted["wrote_paper"] = 0

	train_out = pd.concat([train_conf, train_deleted])
	train_out["paper_id"] = train_out["paper_id"].astype(int)

	author_data = "./data/Author.csv"
	print "Reading Author.csv from {0}".format(author_data)
	author_df = pd.read_csv(author_data)
	author_df["Name"] = author_df["Name"].fillna("")
	author_df["Affiliation"] = author_df["Affiliation"].fillna("")
	author_df = author_df.rename(columns={"Id":"author_id", "Name":"author_name", "Affiliation":"author_affiliation"})
	author_df["author_name_clean"] = author_df["author_name"].replace(string_replacements, regex=True).str.lower()
	author_df["author_affiliation_clean"] = author_df["author_affiliation"].replace(string_replacements, regex=True).str.lower()


	paper_data = "./data/Paper.csv"
	print "Reading Paper.csv from {0}".format(paper_data)
	paper_df = pd.read_csv(paper_data)
	paper_df["Title"] = paper_df["Title"].fillna("")
	paper_df["Keyword"] = paper_df["Keyword"].fillna("")
	#paper_df["Year"] = paper_df["Year"].fillna(0)
	paper_df["JournalId"] = paper_df["JournalId"].fillna(0)
	paper_df["ConferenceId"] = paper_df["ConferenceId"].fillna(0)
	paper_df = paper_df.rename(columns={
		"Id":"paper_id",
		"Title":"paper_title", 
		"Year":"paper_year", 
		"ConferenceId":"conference_id", 
		"JournalId":"journal_id", 
		"Keyword":"paper_keyword"
	})


	paper_author_data = "./data/PaperAuthor.csv"
	print "Reading PaperAuthor.csv from {0}".format(paper_author_data)
	paper_author_df = pd.read_csv(paper_author_data)
	paper_author_df = paper_author_df.drop_duplicates(subset=["AuthorId", "PaperId"])
	paper_author_df["Name"] = paper_author_df["Name"].fillna("")
	paper_author_df["Affiliation"] = paper_author_df["Affiliation"].fillna("")
	paper_author_df = paper_author_df.rename(columns={
		"PaperId":"paper_id", 
		"AuthorId":"author_id", 
		"Name":"paper_author_name", 
		"Affiliation":"paper_author_affiliation"
	})
	paper_author_df["paper_author_name_clean"] = paper_author_df["paper_author_name"].replace(string_replacements, regex=True).str.lower()
	paper_author_df["paper_author_affiliation_clean"] = paper_author_df["paper_author_affiliation"].replace(string_replacements, regex=True).str.lower()

	#Joins
	author_join = pd.merge(author_df, paper_author_df, how="left", on="author_id")
	author_join["paper_id"] = author_join["paper_id"].fillna(0).astype(int)

	paper_join = pd.merge(paper_df, paper_author_df, how="left", on="paper_id")
	paper_join["author_id"] = paper_join["author_id"].fillna(0).astype(int)

	#Has Features
	has_author_features = author_join[["author_id", "paper_id"]]
	has_author_features["has_author_name"] = np.where(author_join["author_name"] == "", 0, 1)
	has_author_features["has_author_affiliation"] = np.where(author_join["author_affiliation"] == "", 0, 1)

	has_paper_features = paper_join[["author_id", "paper_id"]]
	has_paper_features["has_paper_title"] = np.where(paper_join["paper_title"] == "", 0, 1)
	has_paper_features["has_paper_year"] = np.where(np.logical_or(paper_join["paper_year"] == 0, paper_join["paper_year"] == None), 0, 1)
	has_paper_features["has_conference_id"] = np.where(np.logical_or(paper_join["conference_id"] == 0, paper_join["conference_id"] == None), 0, 1)
	has_paper_features["has_journal_id"] = np.where(np.logical_or(paper_join["journal_id"] == 0, paper_join["journal_id"] == None), 0, 1)
	has_paper_features["has_paper_keyword"] = np.where(paper_join["paper_keyword"] == "", 0, 1)

	train_out = pd.merge(train_out, has_author_features, how="left", on=["author_id", "paper_id"])
	train_out = pd.merge(train_out, has_paper_features, how="left", on=["author_id", "paper_id"])

		# Name Edit Distance Features
	name_df = author_join[["author_id", "paper_id", "author_name", "author_name_clean", "paper_author_name", "paper_author_name_clean"]]
	author_name_splits =  name_df['author_name_clean'].str.split(' ', 1, expand=True)
	name_df["author_first_name"] = author_name_splits[0]
	name_df["author_last_name"] = author_name_splits[1]

	paper_author_name_splits =  name_df['paper_author_name_clean'].str.split(' ', 1, expand=True)
	name_df["paper_author_first_name"] = paper_author_name_splits[0]
	name_df["paper_author_last_name"] = paper_author_name_splits[1]

	name_features = name_df[["author_id", "paper_id"]]

	name_features["name_clean_dist"] = name_df.apply(lambda row: edit.eval(row["author_name_clean"], row["paper_author_name_clean"]), axis=1)
	name_features["name_clean_dist"] = name_df.apply(lambda row: edit.eval(row["author_first_name"], row["paper_author_first_name"]), axis=1)
	name_features["name_clean_dist"] = name_df.apply(lambda row: edit.eval(row["author_last_name"], row["paper_author_last_name"]), axis=1)

	train_out = pd.merge(train_out, name_features, how="left", on=["author_id", "paper_id"])


	# author_name_df = author_df[["author_id", "author_name", "author_name_clean"]]
	# author_name_splits =  author_name_df['author_name_clean'].str.split(' ', 1, expand=True)
	# author_name_df["author_first_name"] = author_name_splits[0]
	# author_name_df["author_last_name"] = author_name_splits[1]

	# paper_author_name_df = paper_author_df[["paper_id", "author_id", "paper_author_name", "paper_author_name_clean"]]
	# paper_author_name_splits =  paper_author_name_df['paper_author_name_clean'].str.split(' ', 1, expand=True)
	# paper_author_name_df["paper_author_first_name"] = paper_author_name_splits[0]
	# paper_author_name_df["paper_author_last_name"] = paper_author_name_splits[1]


	# name_join = pd.merge(author_name_df, paper_author_name_df, how="left", on="author_id")
	# name_join["paper_id"] = name_join["paper_id"].fillna(0.0).astype(int)
	# name_join["paper_author_name"] = name_join["paper_author_name"].fillna("")
	# name_join["paper_author_name_clean"] = name_join["paper_author_name_clean"].fillna("")
	# name_join["paper_author_first_name"] = name_join["paper_author_first_name"].fillna("")
	# name_join["paper_author_last_name"] = name_join["paper_author_last_name"].fillna("")


	# name_clean_dist = name_join.apply(lambda row: edit.eval(row["author_name_clean"], row["paper_author_name_clean"]), axis=1)
	# first_name_dist = name_join.apply(lambda row: edit.eval(row["author_first_name"], row["paper_author_first_name"]), axis=1)
	# last_name_dist = name_join.apply(lambda row: edit.eval(row["author_last_name"], row["paper_author_last_name"]), axis=1)

	# name_join_features = name_join[["author_id", "paper_id"]]
	# name_join_features["feature_1"] = name_clean_dist
	# name_join_features["feature_2"] = first_name_dist
	# name_join_features["feature_3"] = last_name_dist
	# train_out = pd.merge(train_out, name_join_features, how="left", on=["author_id", "paper_id"])

	# Affiliation Edit Distance Features
	affiliation_df = author_join[["author_id", "paper_id", "author_affiliation", "author_affiliation_clean", "paper_author_affiliation", "paper_author_affiliation_clean"]]

	affiliation_features = affiliation_df[["author_id", "paper_id"]]

	affiliation_features["affiliation_clean_dist"] = name_df.apply(lambda row: edit.eval(row["author_affiliation_clean"], row["paper_author_affiliation_clean"]), axis=1)

	train_out = pd.merge(train_out, affiliation_features, how="left", on=["author_id", "paper_id"])
	#Author Publication Year Features
	

	author_years = paper_join[paper_join.paper_year != 0].groupby(['author_id'], sort=False)['paper_year']

	ays_min = author_years.min().rename("min_pub_year")
	ays_max = author_years.max().rename("max_pub_year")
	ays_mean = author_years.mean().rename("mean_pub_year")
	ays_median = author_years.median().rename("median_pub_year")

	author_year_stats = pd.concat([ays_min, ays_max, ays_mean, ays_median], axis=1)
	author_year_stats['author_id'] = author_year_stats.index

	author_year_features = paper_join[["author_id", "paper_id", "paper_year"]]
	author_year_features = pd.merge(author_year_features, author_year_stats, how="left", on="author_id")
	author_year_features["min_year_diff"] = author_year_features["min_pub_year"] - author_year_features["paper_year"]
	author_year_features["max_year_diff"] = author_year_features["max_pub_year"] - author_year_features["paper_year"]
	author_year_features["mean_year_diff"] = author_year_features["mean_pub_year"] - author_year_features["paper_year"]
	author_year_features["median_year_diff"] = author_year_features["median_pub_year"] - author_year_features["paper_year"]

	author_year_features_merge = author_year_features[["author_id", "paper_id", "min_year_diff", "max_year_diff", "mean_year_diff", "median_year_diff"]]
	train_out = pd.merge(train_out, author_year_features_merge, how="left", on=["author_id", "paper_id"])

if __name__ == "__main__": main()