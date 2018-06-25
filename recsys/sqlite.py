import sqlite3
import os
from tqdm import tqdm_notebook
import pandas as pd

def create_db(db_name):
    conn = sqlite3.connect(db_name)
    conn.close()

def init_db(db_name):
    db = sqlite3.connect(db_name)
    db.cursor().execute('CREATE TABLE rating (itemId text, userId text, rating real)')
    db.cursor().execute('CREATE TABLE user (user_str, user_int)')
    db.cursor().execute('CREATE TABLE item (item_str, item_int)')
    db.commit()
    db.close()  
    
def exclude_record(item_id):
    tmp = item_id.split("_")[-2:]
    if tmp[0] == 'image':
        return True
    else:
        return False

def rate_item(choice, i):
    if choice == 'croix':
        r = -1.0
    elif choice == 'image_' + str(i):
        r = 3.0
    else:
        r = 0.0
    return r

def insert_rating(cursor, values):
    cursor.execute('INSERT INTO rating VALUES (?, ?, ?)', values)
    
def feedback_to_rating(df, db):
    
    if not os.path.exists(db):
        create_db(db)
    else:
        os.remove(db)

    try:
        init_db(db)
    except:
        pass
    
    conn = sqlite3.connect(db)
    conn.text_factory = str

    for row in tqdm_notebook(df.to_records()):
        b = exclude_record(row[3])
        if not b:
            for i in range(4):
                item_i = row[3] + '_image_' + str(i)
                r = rate_item(row[2], i)
                insert_rating(conn.cursor(), (item_i, row[1], r))
      
    conn.commit()    
    conn.close()

def create_mapping(db_name, table, sql_query):
    conn = sqlite3.connect(db_name)
    df = pd.read_sql(sql_query, conn)
    for idx, value in enumerate(df[table + 'Id'].unique()):
        conn.cursor().execute('INSERT INTO ' + table + ' VALUES (?, ?)', (value, idx))
    conn.commit()
    conn.close()