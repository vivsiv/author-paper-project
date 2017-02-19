import sys
import csv

def transform_author_paper_data(file_name, csv_reader):
	out_file = open(file_name + "_out.csv", 'w') 
	csv_writer = csv.writer(out_file)
	csv_writer.writerow(['author_id', 'paper_id', 'wrote_paper'])

	for read_row in csv_reader:
		author_id = read_row['AuthorId']
		
		confirmed_papers = read_row['ConfirmedPaperIds'].split(" ")
		for paper_id in confirmed_papers:
			csv_writer.writerow([author_id, paper_id, '1'])

		deleted_papers = read_row['DeletedPaperIds'].split(" ")
		for paper_id in deleted_papers:
			csv_writer.writerow([author_id, paper_id, '0'])


def main():
	if len(sys.argv) < 2:
		print "Usage python data-processing.py <file1.csv> <file2.csv>..."
		exit(1)

	for i in range(1,len(sys.argv)):
		in_file = open(sys.argv[i])
		if in_file == None:
			print "Unable to open file: {0}".format(in_file)
			exit(1)
		csv_reader = csv.DictReader(in_file)
		transform_author_paper_data(sys.argv[i].split('.')[0], csv_reader)

if __name__ == "__main__": main()