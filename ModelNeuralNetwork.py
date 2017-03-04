# neural network model with feature inputs and output
# train features 
# we need X, y, X_test
# output y_test
import pandas as pd
from sklearn.neural_network import MLPClassifier
# confusion matrix to evaluate result
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

pd.set_option('display.mpl_style', 'default')
#figuresize(15,5)

dataRead = pd.read_csv('TrainOut.csv')

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
		"author_count"
		]
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
train, test = train_test_split(selectData, test_size = 0.2)


#X = [[0., 0.], [1., 1.]]
#X_train = # input, features list of arrays
X_train = train[["has_author_name",
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
#y = [0, 1]
y_train = train[["wrote_paper"]]# output, 0 and 1

#how many features in each layer of 3 layers
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(25, 15, 3), random_state=1)
#clf = MLPClassifier(
#                hidden_layer_sizes=(25, 15, 5)) 0.77
clf.fit(X_train,y_train)

X_test = test[["has_author_name",
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
#y = [0, 1]
y_test = test[["wrote_paper"]]# output, 0 and 1
y_prediction = clf.predict(X_test)

#confusion matrix to evaluate the result.
print('result confusion matrix: ')
print(confusion_matrix(y_test,y_prediction))
print(classification_report(y_test,y_prediction))

#cross validation
X = selectData[["has_author_name",
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
y = selectData[["wrote_paper"]]
#scores = cross_val_score(clf,X, y, cv = 3)
#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(clf,selectData[features], selectData["wrote_paper"], cv = 5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



