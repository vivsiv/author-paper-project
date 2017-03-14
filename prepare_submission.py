import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import numpy as np
import pandas as pd
import pickle


def main():
	prob_predictions = pd.read_csv("./valid/ValidProbabilities.csv")


	submission = prob_predictions.sort_values(by="wrote_paper_prob", ascending=True).groupby("author_id")["paper_id"].agg({"papers":(lambda group: list(group))})
	submission = submission.reset_index()

	submission["papers"] = submission.apply(lambda row: " ".join(str(paper_id) for paper_id in row["papers"]), axis=1)
	submission.rename(columns={"author_id":"AuthorId","papers":"PaperIds"}, inplace=True)

	submission.sort_values(by="AuthorId").to_csv("./Submission.csv", index=False)

	# pd.groupby(["author_id","wrote_paper"])["paper_id"].agg({"papers":(lambda group: list(group))})


if __name__ == "__main__": main()