import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import numpy as np
import pandas as pd
import pickle

string_replacements = {
	"\\.*\\s+": " ",
	"['\\-]": "",
	"[^a-zA-Z ]": ""
}

def main():
	train_file = "./data/Train.csv"
	print "Reading Train.csv from {0}".format(train_file)
	train_csv = pd.read_csv(train_file)

	train_conf = pd.DataFrame(train_csv.ConfirmedPaperIds.str.split(" ").tolist(), index=train_csv.AuthorId).stack()
	train_conf = train_conf.reset_index()[['AuthorId', 0]]
	train_conf.columns = ["author_id", "paper_id"]
	train_conf["wrote_paper"] = 1

	train_deleted = pd.DataFrame(train_csv.DeletedPaperIds.str.split(" ").tolist(), index=train_csv.AuthorId).stack()
	train_deleted = train_deleted.reset_index()[['AuthorId', 0]]
	train_deleted.columns = ["author_id", "paper_id"]
	train_deleted["wrote_paper"] = 0

	train_df = pd.concat([train_conf, train_deleted])
	train_df["paper_id"] = train_df["paper_id"].astype(int)
	print "Saving train_base to ./pkl/author_join.pkl"
	train_df.to_pickle("./pkl/train_base.pkl")

	# Author.csv
	author_data = "./data/Author.csv"
	print "Reading Author.csv from {0}".format(author_data)
	author_df = pd.read_csv(author_data)

	author_df["Name"] = author_df["Name"].fillna("")
	author_df["Affiliation"] = author_df["Affiliation"].fillna("")

	author_df = author_df.rename(columns={"Id":"author_id", "Name":"author_name", "Affiliation":"author_affiliation"})

	author_df["author_name_clean"] = author_df["author_name"].replace(string_replacements, regex=True).str.lower()
	author_df["author_affiliation_clean"] = author_df["author_affiliation"].replace(string_replacements, regex=True).str.lower()


	# Paper.csv
	paper_data = "./data/Paper.csv"
	print "Reading Paper.csv from {0}".format(paper_data)
	paper_df = pd.read_csv(paper_data)

	paper_df["Title"] = paper_df["Title"].fillna("")
	paper_df["Keyword"] = paper_df["Keyword"].fillna("")
	paper_df["Year"] = paper_df["Year"].fillna(0)
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


	# Paper_Author.csv
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

	# Joins
	author_join = pd.merge(author_df, paper_author_df, how="left", on="author_id")
	author_join["paper_id"] = author_join["paper_id"].fillna(0).astype(int)
	print "Saving author_join to ./pkl/author_join.pkl"
	author_join.to_pickle("./pkl/author_join.pkl")

	paper_join = pd.merge(paper_df, paper_author_df, how="left", on="paper_id")
	paper_join["author_id"] = paper_join["author_id"].fillna(0).astype(int)
	print "Saving paper_join to ./pkl/author_join.pkl"
	paper_join.to_pickle("./pkl/paper_join.pkl")


	# Valid.csv
	valid_file = "./data/Valid.csv"
	print "Reading Valid.csv from {0}".format(valid_file)
	valid_csv = pd.read_csv(valid_file)


	valid_data = pd.DataFrame(valid_csv.PaperIds.str.split(" ").tolist(), index=valid_csv.AuthorId).stack()
	valid_data = valid_data.reset_index()[['AuthorId', 0]]
	valid_data.columns = ["author_id", "paper_id"]
	valid_data["paper_id"] = valid_data["paper_id"].fillna(0).astype(int)
	valid_data.to_pickle("./pkl/valid_base.pkl")
	# valid_data["wrote_paper"] = 1

if __name__ == "__main__": main()

