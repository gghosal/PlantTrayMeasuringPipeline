import sqlite3
import pandas
import sqlite3

import pandas


##Connect to database
def create_date_time(row):
    return pandas.to_datetime(str(str(row['date']) + " " + str(row['time'])[:-2] + ":" + str(row['time'])[-2:]))


cxn = sqlite3.connect('/Users/gghosal/Desktop/gaurav_new_photos/measurements4.db')
df = pandas.read_sql_query('SELECT * FROM plants', cxn)
times = df.groupby(['date', 'time'])
time_indx = [pandas.to_datetime(str(str(j[0]) + " " + str(j[1])[:-2] + ":" + str(j[1])[-2:])) for j in
             times.groups.keys()]
time_df = pandas.DataFrame(index=time_indx)
trays = df.groupby(['potnumber', 'position'])
print(len(trays.groups.keys()))
for i in trays.groups.keys():
    current_df = trays.get_group(i)
    current_df.index = current_df.apply(lambda row: create_date_time(row), axis=1)
    del current_df['date'], current_df['time'], current_df['potnumber'], current_df['position']
    current_df.columns = list([str(i[0] + "-" + i[1])])
    time_df = time_df.join(current_df, how='outer')
    print('iter')
