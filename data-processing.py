import sys
import csv
import os

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

def make_author_map(csv_reader):
	author_map = {}

	for read_row in csv_reader:
		std_names = standardize_name(read_row['Name'])
		author_map[read_row['Id']] = (std_names, read_row['Affiliation'])

	return author_map

def make_paper_map(csv_reader):
	paper_map = {}

	for read_row in csv_reader:
		paper_map[read_row['Id']] = read_row['Title']

	return paper_map

def make_author_paper_map(csv_reader):
	author_paper_map = {}

	for read_row in csv_reader:
		author_paper_map[read_row['AuthorId'] + ":" + read_row['PaperId']] = (standardize_name(read_row['Name']), read_row['Affiliation'])

	return author_paper_map

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
		author_data = author_map.get(author_id, "")
		author_names = author_data[0]
		author_affiliation = author_data[1]
		
		confirmed_papers = read_row['ConfirmedPaperIds'].split(" ")
		for paper_id in confirmed_papers:
			paper_title = paper_map.get(paper_id, "")
			author_paper_data = author_paper_map.get(author_id + ":" + paper_id, "")
			author_paper_names = author_paper_data[0]
			author_paper_affiliation = author_paper_data[1]

			csv_writer.writerow([author_id, paper_id, 
				author_names[0], author_names[1], author_names[2],author_names[3],author_names[4],
				author_affiliation,
				author_paper_names[0], author_paper_names[1], author_paper_names[2], author_paper_names[3], author_paper_names[4],
				author_paper_affiliation,
				paper_title, '1'])

		deleted_papers = read_row['DeletedPaperIds'].split(" ")
		for paper_id in deleted_papers:
			paper_title = paper_map.get(paper_id, "")
			author_paper_data = author_paper_map.get(author_id + ":" + paper_id, "")
			author_paper_names = author_paper_data[0]
			author_paper_affiliation = author_paper_data[1]

			csv_writer.writerow([author_id, paper_id, 
				author_names[0], author_names[1], author_names[2],author_names[3],author_names[4],
				author_affiliation,
				author_paper_names[0], author_paper_names[1], author_paper_names[2], author_paper_names[3], author_paper_names[4],
				author_paper_affiliation,
				paper_title, '0'])

def main():
	if len(sys.argv) != 2:
		print "Usage python data-processing.py <data dir>..."
		exit(1)

	data_dir = sys.argv[1]
	print "Reading data from: {0}".format(data_dir)

	author_map = {}
	paper_map = {}
	author_paper_map = {}

	for filename in os.listdir(data_dir):
		print "Reading file: {0}".format(filename)

		pathname = data_dir + filename
		infile = open(pathname)

		if infile == None:
			print "Unable to open file: {0}".format(infile)
			exit(1)

		if filename == "Author.csv":
			author_map = make_author_map(csv.DictReader(infile))
		elif filename == "Paper.csv":
			paper_map = make_paper_map(csv.DictReader(infile))
		elif filename == "PaperAuthor.csv":
			author_paper_map = make_author_paper_map(csv.DictReader(infile))
		elif filename == "Train.csv":
			transform_train_data(csv.DictReader(infile), author_map, paper_map, author_paper_map)
			# transform_train_data(csv.DictReader(infile), author_map, paper_map, {})
		else:
			print "Not using {0}".format(filename)


if __name__ == "__main__": main()