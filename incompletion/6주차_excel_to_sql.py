import pymysql
import pandas as pd

def load_db_score_data():
    file = 'db_score_3_labels.xlsx'
    db_score = pd.read_excel(file)
    conn = pymysql.connect(host = 'localhost', user= 'Soo', password = '1234', db = 'data_science') 
    curs = conn.cursor(pymysql.cursors.DictCursor)

    drop_sql = """drop table if exists db_score"""
    curs.execute(drop_sql)
    conn.commit()

    import sqlalchemy
    database_username = 'Soo'
    database_password = '1234'
    database_ip = 'localhost'
    database_name = 'data_science'
    database_connection = sqlalchemy.create_engine('mysql+pymysql://{0}:{1}@{2}/{3}'.format(database_username,database_password,database_ip,database_name ))
    
    db_score.to_sql(con =database_connection, name = 'db_score', if_exists = 'replace')

load_db_score_data()
