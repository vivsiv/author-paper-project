# import sys
# sys.path.append("/usr/local/lib/python2.7/site-packages")
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.ensemble as en
from sklearn.model_selection import cross_val_score

# import sklearn.cluster 
# from sklearn.feature_selection import SelectKBest, f_classif
# import matplotlib.pyplot as plt

print "Reading in Training Data"
train_data = pd.read_csv("./data/TrainOutNew.csv")

#Select K Best Factors
# predictors = ["CmpPct", "IntPct", "Y/A", "AY/A", "ANY/A", "TD/Int", "Rate", "Y/G"]
# selector = SelectKBest(f_classif, k=5)
# selector.fit(inactive_qbs[predictors], inactive_qbs["HOF"])
# scores = -np.log10(selector.pvalues_)

# plt.bar(range(len(predictors)), scores)
# plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()

features = ["has_author_name",
		"has_author_affiliation",
		"has_paper_title",
		"has_paper_year",
		"has_conference_id",
		"has_journal_id",
		"has_paper_keyword",
		"name_clean_lev_dist",
		"first_name_lev_dist",
		"last_name_lev_dist",
		"name_clean_jaro_dist",
		"first_name_jaro_dist",
		"last_name_jaro_dist",
		"affiliation_clean_lev_dist",
		"affiliation_clean_jaro_dist",
		"min_year_diff",
		"max_year_diff",
		"mean_year_diff",
		"median_year_diff",
		"author_count",
		"matched_substring_name_ratio",
		"matched_substring_affiliation_ratio",
		"matched_substring_first_name_ratio",
		"matched_substring_last_name_ratio",
		"num_authors_same_name"]

# print "Using RandomForest with 20 trees"
# random_forest_model = en.RandomForestClassifier(n_estimators=30)
# print "Performing Cross Validation"
# scores = cross_val_score(random_forest_model, train_data[features], train_data["wrote_paper"], cv=5)
# print("Accuracy Over 5 Folds: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


print "Using GradientBoosting model"
gradient_boosting_model = en.GradientBoostingClassifier(loss="exponential")
print "Performing Cross Validation"
scores = cross_val_score(gradient_boosting_model, train_data[features], train_data["wrote_paper"], cv=5)
print("Accuracy Over 5 Folds: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# print("Average Score over 5 folds is:", scores.mean())
# model.fit(inactive_qbs[predictors], inactive_qbs["wrote_paper"])

# preds = model.predict((inactive_qbs[predictors]).head(25))
# print((inactive_qbs["HOF"]).head(25),preds)
#print(inactive_qbs["Y/A"].median())
#print(active_qbs["Y/A"].median())

# active_qbs = pandas.read_csv("ValidOut.csv")
# preds = model.predict(active_qbs[predictors])

# result = pandas.DataFrame({
# 	"Name": active_qbs["Name"],
# 	"HOF": preds
# })

# hof = result.loc[result["HOF"] == 1]
# print(hof)