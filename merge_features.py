import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import os
import numpy as np
import pandas as pd


def main():
	base = "train"
	if len(sys.argv) > 1 and sys.argv[1] == "valid":
		base = "valid"

	print "Merging features for: {0}...".format(base.capitalize())

	base_df = pd.read_pickle("./pkl/{0}_base.pkl".format(base))

	df_arr = []

	index = 0
	for features_file in os.listdir("./{0}".format(base)):
		if features_file.endswith(".csv"):
			print "Reading {0}".format(features_file)

			features_path = "./{0}/{1}".format(base, features_file)
			features_df = pd.read_csv(features_path)
			if index > 0:
				features_df.drop(["author_id", "paper_id"], axis=1, inplace=True)

			df_arr.append(features_df)
			index += 1

	
	merged_features = pd.concat(df_arr, axis=1)

	out_df = pd.merge(base_df, merged_features, how="left", on=["author_id", "paper_id"])


	out_columns = list(out_df.drop(["wrote_paper"], axis=1).columns.values)
	out_columns.append("wrote_paper")
	out_df.sort_values(by="author_id").to_csv("./{0}/{1}Out.csv".format(base, base.capitalize()), index=False, columns=out_columns)


if __name__ == "__main__": main()
