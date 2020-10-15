import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pymysql


#Importing Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def classification_performance_eval(y, y_predict):
    tp, tn, fp, fn = 0,0,0,0
    
    for y, yp in zip(y,y_predict):
        if y == 1 and yp ==1 :
            tp += 1 
        elif y == 1 and yp == -1 :
            fn += 1 
        elif y == -1 and yp == 1 :
            fp += 1 
        elif y == -1 and yp == -1 :
            tn += 1 
    try :
        accuracy = (tp+ tn)/(tp+tn+fp+fn)
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)
        f1_score = 2*precision*recall / (precision + recall)
        
    except:
        if tp == 0: 
            precision, recall, f1_score = 0 ,0 ,0 
    
    return accuracy, precision, recall, f1_score



## 고정##
conn = pymysql.connect(host = 'localhost', user = 'Soo', password = '1234' , db = 'data_science')
curs = conn.cursor(pymysql.cursors.DictCursor)
sql = 'select*from db_score'
curs.execute(sql)
data = curs.fetchall()
curs.close()
conn.close()
X = [ ( t['homework'], t['discussion'], t['midterm'] )  for t in data ]
#y = [ 0 if (t['grade'] == 'A')  else 1 if (t['grade'] == 'B') else 2 for t in data ]            # muliti class 
y = [ 1 if (t['grade'] == 'B') else -1 for t in data ]                  # binary 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.333 , random_state=42)
## 고정 ## 



## Random Forests## 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42).fit(X_train, y_train)  # n_estimatiors : tree 개수 
y_predict = classifier.predict(X_test)

acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)

print( ' < train_test_split_Random Forest_Binary_tts > ')
print('accuracy = %.4f' %acc)
print('precision = %.4f' %prec)
print('recall = %.4f ' %rec)
print('F1_score = %.4f' %f1, '\n')


from sklearn.model_selection import KFold

accuracy = []
precision = []
recall = []
f1_score = []

kf = KFold(n_splits = 4, random_state =42, shuffle = True)

for train_index, test_index in kf.split(X):
    
    X = np.array(X)
    Y = np.array(y)
    X_train , X_test = X[train_index], X[test_index]
    Y_train , Y_test = Y[train_index], Y[test_index]
    
##########################
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
    classifier.fit(X_train, Y_train)  # n_estimatiors : tree 개수 
    y_predict = classifier.predict(X_test)
###########################

    acc, prec, rec, f1 = classification_performance_eval(Y_test, y_predict)
    accuracy.append(acc)
    precision.append(prec)
    recall.append(rec)
    f1_score.append(f1)

    import statistics 


print( ' < K-fold cross validation by_Logistic Regression-Binary > ')
print('accuracy_평균 = ', round(statistics.mean(accuracy),4))
print('precision_평균 = ', round(statistics.mean(precision),4))
print('recall_평균 = ', round(statistics.mean(recall),4))
print('f1_score_평균 = ', round(statistics.mean(f1_score),4),'\n')
