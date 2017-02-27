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
	name_arr = name.split(' ')
	first_name = name_arr[0].lower()
	middle_name = ""
	last_name = name_arr[len(name_arr) - 1].lower()

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

	train_mod = pd.concat([train_conf, train_deleted])
	train_mod = train_mod["paper_id"].astype(int)

	author_data = "./data/Author.csv"
	print "Reading Author.csv from {0}".format(author_data)
	author_df = pd.read_csv(author_data)
	author_df = author_df.fillna("")
	author_df["NameClean"] = author_df["Name"].replace(string_replacements, regex=True)
	author_df["NameClean"] = author_df["NameClean"].str.lower()
	author_df["AffiliationClean"] = author_df["Affiliation"].replace(string_replacements, regex=True)
	author_df["AffiliationClean"] = author_df["AffiliationClean"].str.lower()
	author_df.columns = ["author_id", "author_name", "author_affiliation", "author_name_clean", "author_affiliation_clean"]

	
	paper_data = "./data/Paper.csv"
	print "Reading Paper.csv from {0}".format(paper_data)
	paper_df = pd.read_csv(paper_data)
	paper_df.columns = ['paper_id', 'paper_title', 'paper_year', 'conference_id', 'journal_id', 'paper_keyword']

	paper_author_data = "./data/PaperAuthor.csv"
	print "Reading PaperAuthor.csv from {0}".format(paper_author_data)
	paper_author_df = pd.read_csv(paper_author_data)
	paper_author_df = paper_author_df.fillna("")
	paper_author_df["NameClean"] = paper_author_df["Name"].replace(string_replacements, regex=True)
	paper_author_df["NameClean"] = paper_author_df["NameClean"].str.lower()
	paper_author_df["AffiliationClean"] = paper_author_df["Affiliation"].replace(string_replacements, regex=True)
	paper_author_df["AffiliationClean"] = paper_author_df["AffiliationClean"].str.lower()
	paper_author_df.columns = ["paper_id", "author_id", "paper_author_name", "paper_author_affiliation", "paper_author_name_clean", "paper_author_affiliation_clean"]


	#Basic Name Edit Distance
	author_name_df = author_df[['author_id', 'author_name', 'author_name_clean']]
	paper_author_name_df = paper_author_df[['paper_id', 'author_id', 'paper_author_name', 'paper_author_name_clean']]
	name_join = pd.merge(author_name_df, paper_author_name_df, how="left", on="author_id")

	name_join['edit_dist'] = name_join.apply(lambda row: ed.eval(row['author_name_clean'], row['paper_author_name_clean']), axis=1)
	name_join_train = name_join[['author_id', 'paper_id', 'edit_distance']]
	train_mod = pd.merge(train_mod, name_join_train, how="left", on=['author_id', 'paper_id'])


	#author_affiliation_df = author_df[['author_id', 'author_name', 'author_name_clean']]

	#Generating author year stats
	paper_join = pd.merge(paper_df, paper_author_df, how="left", on="paper_id")

	author_years = paper_join.groupby(['author_Id'], sort=False)['Year']

	ays_min = author_years.min().rename("min_pub_year")
	ays_max = author_years.max().rename("max_pub_year")
	ays_mean = author_years.mean().rename("mean_pub_year")

	author_year_stats = pd.concat([ays_min, ays_max, ays_mean], axis=1)
	author_year_stats['author_id'] = author_year_stats.index

	author_df = pd.merge(author_df, author_year_stats, how="left", on="Id")

if __name__ == "__main__": main()