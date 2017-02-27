# author-paper-project
CS 249 final project

Feature List

Has Features
1. The (paper_id, author_id) combo has a Name in Author.csv
2. The (paper_id, author_id) combo has an Affiliation in Author.csv
3. The (paper_id, author_id) combo has a Title in Paper.csv
4. The (paper_id, author_id) combo has a Year in Paper.csv
5. The (paper_id, author_id) combo has a ConferenceId in Paper.csv
6. The (paper_id, author_id) combo has a JournalId in Paper.csv
7. The (paper_id, author_id) combo has a Keyword in Paper.csv

Name Distance Features
1. Edit distance between the clean name in Author.csv and clean name in PaperAuthor.csv for a (paper_id, author_id) combo
2. Edit distance between the first name in Author.csv and first name in PaperAuthor.csv for a (paper_id, author_id)_ combo
3. Edit distance between the last name in Author.csv and last name in PaperAuthor.csv for a (paper_id, author_id) combo
**4. Jaro distance between the clean name in Author.csv and clean name in PaperAuthor.csv for a (paper_id, author_id) combo
**5. Jaro distance between the first name in Author.csv and first name in PaperAuthor.csv for a (paper_id, author_id) combo
**6. Jaro distance between the last name in Author.csv and last name in PaperAuthor.csv for a (paper_id, author_id) combo

Affiliation Distance Features
1. Edit distance between the clean affiliation in Author.csv and clean name in PaperAuthor.csv for a (paper_id, author_id) combo
2. Jaro distance between the clean affiliation in Author.csv and clean name in PaperAuthor.csv for a (paper_id, author_id) combo

Year Features
1. Difference between the author's minimum publish year and the paper's publish year for a (paper_id, author_id) combo
2. Difference between the author's maximum publish year and the paper's publish year for a (paper_id, author_id) combo
3. Difference between the author's mean publish year and the paper's publish year for a (paper_id, author_id) combo
4. Difference between the author's median publish year and the paper's publish year for a (paper_id, author_id) combo

Coauthor Features
1.