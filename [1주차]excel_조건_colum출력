import pandas as pd
import numpy as np

f = 'db_score.xlsx'

df = pd.read_excel(f)

df1 = df[df.midterm >= 20]              #전체 data 중에서 중간고사 20점 이상 학생 
df1 = df1[df1.final >= 20]              #중간고사 20점 이상 학생 중에서 기말고사 20점 이상 학생 

df1 = df1.drop(df1.columns[[1,2,3,6]], axis = 'columns')

print(df1)

