#after run any model run this file

# 1 valid_train_features
# 2 any model
# 3 this file

#out put are two csv files, with or without features
import numpy as np
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
import pickle

#this file may need to be run individually after we each model. this way saves time. 

# we need train the new testing data and save into valid_data.csv
dataRead = pd.read_csv('ValidTrainOut.csv')  #the file name could be changed

# add new features behind:
X_test = dataRead[["has_author_name",
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


#need to delete every time for old classifier
#with open('classifer.pkl', 'rb') as f:
#    clf = pickle.load(f)
#need above





dataRead['y_prediction'] = clf.predict(X_test)
dataRead.to_csv("features_with_prediction.csv", index = False)

#select the data we want
newDataRead = pd.read_csv('features_with_prediction.csv')
result_we_want = newDataRead[["author_id","paper_id","y_prediction"]]

result_we_want.to_csv("prediction_result.csv", index = False)

# validSolution
valid_csv = pd.read_csv('ValidSolution.csv')
valid_Solution_we_need = pd.DataFrame(valid_csv.PaperIds.str.split(" ").tolist(), index=valid_csv.AuthorId).stack()
valid_Solution_we_need = valid_Solution_we_need.reset_index()[['AuthorId', 0]]
valid_Solution_we_need.columns = ["author_id", "paper_id"]
valid_Solution_we_need["paper_id"] = valid_Solution_we_need["paper_id"].fillna(0).astype(int)
valid_Solution_we_need['wrote_paper_true_result']=1
                      
                      


#predic and real solution merge
frames = [result_we_want,valid_Solution_we_need]
merge_predict_and_real =result_we_want.merge(valid_Solution_we_need, how = "inner",on =["author_id","paper_id"] )
merge_predict_and_real.to_csv("merge_predict_and_real.csv", index = False)

#calculate accuracy
df_accu = pd.DataFrame(merge_predict_and_real,columns = ['y_prediction','wrote_paper_true_result'])
df_accu['correct_predict'] = np.where((df_accu['y_prediction']==df_accu['wrote_paper_true_result']),1,0)
accuracy_result = df_accu['correct_predict'].mean()
print('the accuracy is:')
print(accuracy_result)







