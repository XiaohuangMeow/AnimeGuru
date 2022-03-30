import csv
import sqlite3

connection = sqlite3.connect('db.sqlite3')

cursor = connection.cursor()

file = open('anime_data_new_processed.csv')
contents = csv.reader(file)

insert_records = """
INSERT INTO anime_recommend_anime
(Anime_PlanetID, Name, Alternative_Name, Rating_score, Number_votes, Tags, Content_Warning,
Type, Episodes, Finished, Duration, StartYear, EndYear, Season,Studios,Synopsis,Url,picture_url,picture)
VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)
"""

# Importing the contents of the file
# into our person table
cursor.executemany(insert_records, contents)

# SQL query to retrieve all data from
# the person table To verify that the
# data of the csv file has been successfully
# inserted into the table
select_all = "SELECT * FROM anime_recommend_anime;"
rows = cursor.execute(select_all).fetchall()

# Output to the console screen
for r in rows:
    print(r)

# Committing the changes
connection.commit()

# closing the database connection
connection.close()
