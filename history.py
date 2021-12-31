import sqlite3
import pandas as pd
con = sqlite3.connect("~/leo/Library/Safari")
df = pd.read_sql_query("SELECT * from surveys WHERE url LIKE %https://eclass.dongguk.edu/%", con)

# Verify that result of SQL query is stored in the dataframe
print(df.head())

con.close()