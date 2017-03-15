import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
import numpy as np
import pandas as pd
import pickle 
import sklearn as sk
import sklearn.ensemble as en
import sklearn.tree as tree
import sklearn.linear_model as linear
import sklearn.naive_bayes as nb
import sklearn.neural_network as nn
import sklearn.feature_selection as fs
import sklearn.model_selection as ms


def feature_selection(train_data,all_features,n_features=5):
	print "Selecting {0} BEST features from {1} total...".format(n_features, len(all_features))
	selector = fs.SelectKBest(fs.f_classif,n_features)
	selector_out = selector.fit(train_data[all_features],train_data["wrote_paper"])
	# print selector_out.get_support()
	selections = [feature for feature, selected in zip(all_features, selector_out.get_support()) if selected]
	print "Selected Features:"
	print selections

	return selections


# Options:
# - max_features: The number of features to consider when looking for the best split
# - max_depth: The maximum depth of the tree
def decision_tree_classifier(train_data,features,cv_folds=5):
	model = tree.DecisionTreeClassifier()
	print "Testing {0}".format()
	scores = ms.cross_val_score(model, train_data[features], train_data["wrote_paper"], cv=cv_folds)
	print "Decision Tree. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
	model.fit(train_data[features],train_data["wrote_paper"])
	return {"model":model,"score":scores.mean()}

# Options:
# - n_estimators: The number of trees in the forest
# - max_features: The number of features to consider when looking for the best split
# - max_depth: The maximum depth of the tree
def random_forest_classifier(train_data,features,n_trees=50,cv_folds=5):
	model = en.RandomForestClassifier(n_estimators=n_trees)
	print "Testing Random Forest model..."
	scores = ms.cross_val_score(model, train_data[features], train_data["wrote_paper"], cv=cv_folds)
	print "Random Forest. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
	model.fit(train_data[features],train_data["wrote_paper"])
	return {"model":model,"score":scores.mean()}

# Options
# - base_estimator: The base estimator from which the boosted ensemble is built (default DecisionTree)
# - n_estimators: The number of trees in the forest
# - learning_rate: Learning rate shrinks the contribution of each classifier by learning_rate
# - random_state: State or seed for random number generator
def adaboost_classifier(train_data,features,n_estimators=50,cv_folds=5):
	model = en.AdaBoostClassifier(n_estimators=n_estimators)
	print "Testing Adaboost model..."
	scores = ms.cross_val_score(model, train_data[features], train_data["wrote_paper"], cv=cv_folds)
	print "Adaboost. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
	model.fit(train_data[features],train_data["wrote_paper"])
	return {"model":model,"score":scores.mean()}


# Options
# - loss: loss function to be optimized (deviance or exponential)
# - learning_rate: Learning rate shrinks the contribution of each classifier by learning_rate
# - n_estimators: The number of boosting stages to perform
def gradient_boosting_classifier(train_data,features,cv_folds=5):
	model = en.GradientBoostingClassifier(loss="exponential")
	print "Testing Gradient Boosting model..."
	scores = ms.cross_val_score(model, train_data[features], train_data["wrote_paper"], cv=cv_folds)
	print "Gradient Boosting. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
	model.fit(train_data[features],train_data["wrote_paper"])
	return {"model":model,"score":scores.mean()}

# Options
# - C: inverse or regularization of strength
def logistic_classifier(train_data,features,cv_folds=5):
	model = linear.LogisticRegression()
	print "Testing Logistic Regression model..."
	scores = ms.cross_val_score(model, train_data[features], train_data["wrote_paper"], cv=cv_folds)
	print "Logistic. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
	model.fit(train_data[features],train_data["wrote_paper"])
	return {"model":model,"score":scores.mean()}

def bayes_classifier(train_data,features,cv_folds=5):
	model = nb.BernoulliNB()
	print "Testing Naive Bayes model..."
	scores = ms.cross_val_score(model, train_data[features], train_data["wrote_paper"], cv=cv_folds)
	print "Naive Bayes. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
	model.fit(train_data[features],train_data["wrote_paper"])
	return {"model":model,"score":scores.mean()}

# Options
# solver: The solver for weight optimization
# alpha: L2 penalty (regularization term) parameter
# hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer
# ranom_state: State or seed for random number generator
def neural_network_classifier(train_data,features,cv_folds=5):
	model = sk.neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25, 15, 3), random_state=1)
	print "Testing {0}".format()
	scores = ms.cross_val_score(model, train_data[features], train_data["wrote_paper"], cv=cv_folds)
	print "Neural Network. Average Score over %d folds is %0.2f (+/- %0.2f)" % (cv_folds,scores.mean(),scores.std() * 2)
	model.fit(train_data[features],train_data["wrote_paper"])
	return {"model":model,"score":scores.mean()}

def grid_search(train_data,features,predict,cv_folds=5):
	models = {
	    "linear":(sk.linear_model.LogisticRegression(),[{'C':[0.01,.1,.5]}]),
	    "tree":(en.RandomForestClassifier(),[{'n_estimators':[1,5,10]}]),
	    "bayes":(sk.naive_bayes.BernoulliNB(),[{'alpha':[0,0.5,1.0]}])
	}

	grid_searches = {}
	for model_name,model_info in models.iteritems():
		model = model_info[0]
		params = model_info[1]
		grid_model = ms.GridSearchCV(model,params,verbose=5,n_jobs=1,cv=cv_folds)
		grid_model.fit(train_data[features], train_data["wrote_paper"])
		grid_searches[model_name] = grid_model

	best_model = None
	best_score = 0
	for name,gs in grid_searches.iteritems():
		print "Model: " + name + ", Best Score: " + str(gs.best_score_)
		if gs.best_score_ > best_score:
			best_score = gs.best_score_
			best_model = gs.best_estimator_
	print "Best Model {0}".format(best_model)
	return {"model":best_model,"score":best_score}

def get_best_model(train_data,features):
	best_score = 0
	best_model = None
	best_model_name = ""

	results = random_forest_classifier(train_data,features)
	if results["score"] > best_score:
		best_model_name = "RandomForest"
		best_model = results["model"]
		best_score = results["score"]

	# results = decision_tree_classifier(train_data,features)
	# if results["score"] > best_score:
	# 	best_model_name = "DecisionTree"
	# 	best_model = results["model"]
	# 	best_score = results["score"]

	# results = adaboost_classifier(train_data,features)
	# if results["score"] > best_score:
	# 	best_model_name = "Adaboost"
	# 	best_model = results["model"]
	# 	best_score = results["score"]

	# 	results = gradient_boosting_classifier(train_data,features)
	# if results["score"] > best_score:
	# 	best_model_name = "GradientBoosting"
	# 	best_model = results["model"]
	# 	best_score = results["score"]

	# results = logistic_classifier(train_data,features)
	# if results["score"] > best_score:
	# 	best_model_name = "Logistic"
	# 	best_model = results["model"]
	# 	best_score = results["score"]

	# results = bayes_classifier(train_data,features)
	# if results["score"] > best_score:
	# 	best_model_name = "Bayes"
	# 	best_model = results["model"]
	# 	best_score = results["score"]

	# results = neural_network_classifier(train_data,features)
	# if results["score"] > best_score:
	# 	best_model_name = "Neural Network"
	# 	best_model = results["model"]
	# 	best_score = results["score"]

	# results = grid_search(train_data,features)
	# if results["score"] > best_score:
	# 	best_model_name = "Grid Search"
	# 	best_model = results["model"]
	# 	best_score = results["score"]

	print "Best Model:{0}, score:{1}".format(best_model_name, best_score)

	return best_model

def predict(model,predict_data,features):
	result = predict_data[["author_id","paper_id"]].copy()
	predictions = pd.DataFrame(model.predict(predict_data[features]))
	result["wrote_paper"] = predictions[0]

	print "Saving predictions..."
	result.sort_values(by="author_id").to_csv("./valid/ValidPredictions.csv", index=False)

	return result

	
def predict_prob(model,predict_data,features):
	prob_result = predict_data[["author_id","paper_id"]].copy()
	probabilities = pd.DataFrame(model.predict_proba(predict_data[features]))
	prob_result["wrote_paper_prob"] = probabilities[0]

	print "Saving probabilities..."
	prob_result.sort_values(by="author_id").to_csv("./test/TestProbabilities.csv", index=False)

	return prob_result


def evaluate(predictions):
	print "Reading in valid solution"
	solution = pd.read_pickle("./pkl/valid_solution.pkl")
	solution_compare = predictions.merge(solution, how="inner", on=["author_id","paper_id"])
	solution_compare["correct_prediction"] = np.where((solution_compare["wrote_paper"] == solution_compare["wrote_paper_actual"]),1,0)
	total_predictions = len(solution_compare)
	correct_predictions = len(solution_compare[solution_compare["correct_prediction"] == 1])
	percent_correct = float(correct_predictions) / float(total_predictions)

	print "Got %d/%d predictions correct (%0.2f)" % (correct_predictions, total_predictions, percent_correct)

def prepare_submission(probabilities):
	submission = probabilities.sort_values(by="wrote_paper_prob", ascending=True).groupby("author_id")["paper_id"].agg({"papers":(lambda group: list(group))}).reset_index()
	submission["papers"] = submission.apply(lambda row: " ".join(str(paper_id) for paper_id in row["papers"]), axis=1)
	submission.rename(columns={"author_id":"AuthorId","papers":"PaperIds"}, inplace=True)

	print "Saving submission..."
	submission.sort_values(by="AuthorId").to_csv("./Submission.csv", index=False)


def main():
	print "Reading in Training Data"
	train_data = pd.read_csv("./train/TrainOut.csv")

	print "Reading in Predict Data"
	base = "valid"
	if len(sys.argv) > 1 and sys.argv[1] == "test":
		base = "test"
		print "Saving intermediates in:./{0}/ ...".format(base)

	predict_data = pd.read_csv("./{0}/{1}Out.csv".format(base, base.capitalize()))

	features = list(train_data.drop(["author_id","paper_id","wrote_paper"], axis=1).columns.values)

	if len(sys.argv) > 2:
		num_features = int(sys.argv[2])
		if num_features > 0 and num_features <= len(features):
			features = feature_selection(train_data, features, num_features)
		
	model = get_best_model(train_data, features)

	if base == "test":
		probabilities = predict_prob(model,predict_data,features)
		prepare_submission(probabilities)
	else:
		predictions = predict(model,predict_data,features)
		evaluate(predictions)


if __name__ == "__main__": main()

