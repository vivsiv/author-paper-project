import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import os
import numpy as np
import pandas as pd


def main():
	print "Reading train_base"
	train_base = pd.read_pickle("./pkl/train_base.pkl")

	df_arr = []

	index = 0
	for features_file in os.listdir("./train"):
		if features_file.endswith(".csv"):
			print "Reading {0}".format(features_file)

			features_path = "./train/{0}".format(features_file)
			features_df = pd.read_csv(features_path)
			if index > 0:
				features_df.drop(["author_id", "paper_id"], axis=1, inplace=True)

			df_arr.append(features_df)
			index += 1

	
	merged_features = pd.concat(df_arr, axis=1)

	train_out = pd.merge(train_base, merged_features, how="left", on=["author_id", "paper_id"])


	out_columns = list(train_out.drop(["wrote_paper"], axis=1).columns.values)
	out_columns.append("wrote_paper")
	train_out.sort_values(by="author_id").to_csv("./train/TrainOut.csv", index=False, columns=out_columns)


if __name__ == "__main__": main()
