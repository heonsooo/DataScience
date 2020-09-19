import pandas as pd
from sqlalchemy import create_engine
table = pd.read_excel('db_score.xlsx', sheet_name='Sheet1', header=0,)
engine = create_engine("mysql+pymysql://root:gjstndpdu1@@127.0.0.1:3306/data_science", encoding='utf-8-sig')
table.to_sql(name='db_score11', con=engine, if_exists='append', index=False)
