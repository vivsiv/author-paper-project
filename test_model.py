import pandas as pd
from sklearn.neural_network import MLPClassifier
# confusion matrix to evaluate result
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import cross_val_score

#this file may need to be run individually after we each model. this way saves time. 

# we need train the new testing data and save into valid_data.csv
dataRead = pd.read_csv('valid_data.csv')  #the file name could be changed


#add new features behind:
selectData = dataRead[["has_author_name",
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
		"wrote_paper"]]

# add new features behind:
X_test = selectData[["has_author_name",
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
		"author_count"
		]]

y_true = selectData[["wrote_paper"]]

y_prediction = clf.predict(X_test)

#confusion matrix to evaluate the result.
print('result confusion matrix: ')
print(confusion_matrix(y_true,y_prediction))
print(classification_report(y_true,y_prediction))




