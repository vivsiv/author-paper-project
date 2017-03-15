# author-paper-project
CS 249 final project


Required Libraries:
To run this program you will need python and the following libraries:
  - pandas
  - numpy
  - sklearn
  - pickle
  - jellyfish
 All of these libraries can be installed via pip

Directory Structure: Prior to running the program the workind directory should look like this:
./
 data/
 pkl/
 test/
 train/
 valid/
 feature_eng.py
 merge_features.py
 model.py
 run_all.py


Getting the data:
All relevant data files can be acquired from here: https://www.kaggle.com/c/kdd-cup-2013-author-paper-identification-challenge/data
Download: dataRev2.zip, Test.csv, ValidSolution.csv
dataRev2.zip unpacks into: Author.csv, Conference.csv, Journal.csv, Paper.csv, PaperAuthor.csv, Train.csv, Valid.csv
Move all csv files into the data/ folder

Running the Program
The run_all.py script can be used to run the whole program: data processing, feature engineering, modelling, and predictions.
Run it by calling: "python run_all.py"

Running this program will require at least 8GB of memory, 10 to be safe.

data_proc.py will run first and save the following files in pkl/
	- author_join.pkl
	- paper_info.pkl
	- paper_join.pkl
	- test_base.pkl
	- train_base.pkl
	- valid_base.pkl
	- valid_solution.pkl
	data_proc.py can be run on its own by calling: "python data_proc.py"

feature_eng.py will run second and save:
	- TrainOut.csv in train/
	- ValidOut.csv in valid/
	- TestOut.csv in test/

	feature_eng.py can be run on its own by calling: "python feature_eng.py < base > < save_intermediate >"
	- base: {train, valid, test} is the file that features will be engineered on
	- save_intermediate: { save_intermediate } is an optional parameter that will save the intermediate feature csvs. 
	Intermediate csvs can be merged with merge_features.py

model.py will run last and save:
	- ValidPredictions.csv in valid/
	- TestProbabilities.csv in test/
	- Submission.csv in .
	model.py can be run on its own by calling: "python model.py < base > < num_features >"

	- base: {valid, test} is the file that predictions will be made on
	- num_features: { any int } is an optional parameter that will perform feature selection of num_features features
	If you want to try testing the various models uncomment line 272 in model.py
	If you want to run cross validation on the best model uncomment lines 280-283 in model.py


