import sys
import csv
import os
import pickle

def save_obj(obj, out_file_name):
    with open(out_file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(in_file_name):
    with open(in_file_name, 'rb') as in_file:
        return pickle.load(in_file)

def standardize_name(name):
	name_arr = name.split(' ')
	first_name = name_arr[0].lower()
	middle_name = ""
	last_name = name_arr[len(name_arr) - 1].lower()

	if len(name_arr) == 3:
		middle_name = name_arr[1].lower()

	first_initial = first_name[:1]
	last_initial = last_name[:1]
	middle_initial = ""
	if middle_name != "":
		middle_initial = middle_name[:1]
	
	name1 = first_name + " " + last_name
	name2 = first_initial + " " + last_name

	name3 = first_name + " " + middle_name + " " + last_name
	name4 = first_name + " " + middle_initial + " " + last_name
	name5 = first_initial + " " + middle_initial + " " + last_name

	if middle_name == "":
		name3 = first_name + " " + last_name
		name4 = first_name + " " + last_name
		name5 = first_initial + " " + last_name

	return [name1, name2, name3, name4, name5]

def make_author_map(csv_reader, obj_dir):
	author_map = {}

	for read_row in csv_reader:
		author_map[read_row['Id']] = (read_row['Name'], read_row['Affiliation'])

	out_file_name = obj_dir + "author_map.pkl"
	save_obj(author_map, out_file_name)

	return author_map

def make_paper_map(csv_reader, obj_dir):
	paper_map = {}

	for read_row in csv_reader:
		paper_map[read_row['Id']] = read_row['Title']

	out_file_name = obj_dir + "paper_map.pkl"
	save_obj(paper_map, out_file_name)

	return paper_map

def make_author_paper_map(csv_reader, obj_dir):
	author_paper_map = {}

	for read_row in csv_reader:
		author_paper_map[read_row['AuthorId'] + ":" + read_row['PaperId']] = (read_row['Name'], read_row['Affiliation'])

	out_file_name = obj_dir + "author_paper_map.pkl"
	save_obj(author_paper_map, out_file_name)

	return author_paper_map

def make_conference_map(csv_reader, obj_dir):
	conference_map = {}

	for read_row in csv_reader:
		conference_map[read_row['Id']] = (read_row['ShortName'], read_row['FullName'], read_row['HomePage'])

	out_file_name = obj_dir + "conference_map.pkl"
	save_obj(conference_map, out_file_name)

	return conference_map

def make_journal_map(csv_reader, obj_dir):
	journal_map = {}

	for read_row in csv_reader:
		journal_map[read_row['Id']] = (read_row['ShortName'], read_row['FullName'], read_row['HomePage'])

	out_file_name = obj_dir + "journal_map.pkl"
	save_obj(journal_map, out_file_name)

	return journal_map 

def transform_train_data(csv_reader, author_map, paper_map, author_paper_map):
	out_file = open("Train_out.csv", 'w') 
	csv_writer = csv.writer(out_file)
	csv_writer.writerow(['author_id', 'paper_id', 
		'a_name_1', 'a_name_2', 'a_name_3', 'a_name_4', 'a_name_5',
		'a_affiliation',
		'ap_name_1', 'ap_name_2', 'ap_name_3', 'ap_name_4', 'ap_name_5',
		'ap_affiliation',
		'paper_title', 'wrote_paper'])

	for read_row in csv_reader:
		author_id = read_row['AuthorId']

		author_data = author_map.get(author_id, ("",""))

		author_name = author_data[0]
		std_author_names = standardize_name(author_name)

		author_affiliation = author_data[1]
		
		confirmed_papers = read_row['ConfirmedPaperIds'].split(" ")
		for paper_id in confirmed_papers:
			paper_title = paper_map.get(paper_id, "")

			author_paper_data = author_paper_map.get(author_id + ":" + paper_id, ("",""))

			author_paper_name = author_paper_data[0]
			std_author_paper_names = standardize_name(author_paper_name)

			author_paper_affiliation = author_paper_data[1]

			csv_writer.writerow([author_id, paper_id, 
				std_author_names[0], std_author_names[1], std_author_names[2],std_author_names[3],std_author_names[4],
				author_affiliation,
				std_author_paper_names[0], std_author_paper_names[1], std_author_paper_names[2], std_author_paper_names[3], std_author_paper_names[4],
				author_paper_affiliation,
				paper_title, '1'])

		deleted_papers = read_row['DeletedPaperIds'].split(" ")
		for paper_id in deleted_papers:
			paper_title = paper_map.get(paper_id, "")

			author_paper_data = author_paper_map.get(author_id + ":" + paper_id, ("",""))

			author_paper_name = author_paper_data[0]
			std_author_paper_names = standardize_name(author_paper_name)

			author_paper_affiliation = author_paper_data[1]

			csv_writer.writerow([author_id, paper_id, 
				std_author_names[0], std_author_names[1], std_author_names[2],std_author_names[3],std_author_names[4],
				author_affiliation,
				std_author_paper_names[0], std_author_paper_names[1], std_author_paper_names[2], std_author_paper_names[3], std_author_paper_names[4],
				author_paper_affiliation,
				paper_title, '0'])

def main():
	if len(sys.argv) < 3:
		print "Usage python data-processing.py <data dir> <obj_dir> <Train.csv>..."
		exit(1)

	
	data_dir = sys.argv[1]
	obj_dir = sys.argv[2]
	
	author_map = None
	paper_map = None
	conference_map = None
	journal_map = None
	author_paper_map = None

	print "Reading objects from: {0}".format(data_dir)
	for file_name in os.listdir(obj_dir):
		path_name = obj_dir + file_name
		in_file = open(path_name)

		if in_file == None:
			print "Unable to open file: {0}".format(file_name)
			exit(1)

		if file_name == "author_map.pkl":
			print "Loading author_map..."
			author_map = load_obj(path_name)
		elif file_name == "paper_map.pkl":
			print "Loading paper_map..."
			paper_map = load_obj(path_name)
		elif file_name == "author_paper_map.pkl":
			print "Loading author_paper_map..."
			author_paper_map = load_obj(path_name)
		else:
			print "Not using object {0}".format(file_name)


	print "Reading data from: {0}".format(data_dir)
	for file_name in os.listdir(data_dir):
		print "Reading file: {0}".format(file_name)

		path_name = data_dir + file_name
		in_file = open(path_name)

		if in_file == None:
			print "Unable to open file: {0}".format(in_file)
			exit(1)

		if file_name == "Author.csv":
			if author_map == None:
				print "Making author_map... saving to {0}".format(obj_dir)
				author_map = make_author_map(csv.DictReader(in_file), obj_dir)
			else :
				print "Already loaded author_map from obj"
		elif file_name == "Paper.csv":
			if paper_map == None:
				print "Making paper_map... saving to {0}".format(obj_dir)
				paper_map = make_paper_map(csv.DictReader(in_file), obj_dir)
			else :
				print "Already loaded paper_map from obj"
		elif file_name == "Conference.csv":
			if conference_map == None:
				print "Making conference_map... saving to {0}".format(obj_dir)
				conference_map = make_conference_map(csv.DictReader(in_file), obj_dir)
			else :
				print "Already loaded conference_map from obj"
		elif file_name == "Journal.csv":
			if journal_map == None:
				print "Making journal_map... saving to {0}".format(obj_dir)
				journal_map = make_journal_map(csv.DictReader(in_file), obj_dir)
			else :
				print "Already loaded journal_map from obj"
		elif file_name == "PaperAuthor.csv":
			if author_paper_map == None:
				print "Making author_paper_map... saving to {0}".format(obj_dir)
				author_paper_map = make_author_paper_map(csv.DictReader(in_file), obj_dir)
			else :
				print "Already loaded author_paper_map from obj"	
		else:
			print "Not using {0}".format(file_name)


		if len(sys.argv) == 4:
			train_file = open(sys.argv[3])
			if train_file == None:
				print "Unable to open file: {0}".format(file_name)
				exit(1)
			print "Transforming {0}".format(train_file)
			transform_train_data(csv.DictReader(train_file), author_map, paper_map, author_paper_map)

if __name__ == "__main__": main()