# from sklearn.metrics import precision_score, recall_score, f1_score
# from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pymysql
import numpy as np



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
    
    # print(tp, tn, fp, fn)

    try :
        accuracy = (tp+ tn)/(tp+tn+fp+fn)
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)
        f1_score = 2*precision*recall / (precision + recall)
        
    except:
        if tp == 0: 
            precision, recall, f1_score = 0 ,0 ,0 
    
    return accuracy, precision, recall, f1_score

conn = pymysql.connect(host = 'localhost', user = 'Soo', password = '1234' , db = 'data_science')
curs = conn.cursor(pymysql.cursors.DictCursor)

sql = 'select*from db_score'
curs.execute(sql)
data = curs.fetchall()

curs.close()
conn.close()

X = [ ( t['homework'], t['discussion'], t['midterm'] )  for t in data ]
Y = [ 1 if (t['grade'] == 'B')  else -1 for t in data ]
X_train , X_test , Y_train, Y_test = train_test_split(X, Y, test_size = 0.33 , random_state = 42) 

Lr = LogisticRegression(C=1)
Lr.fit(X_train, Y_train)
predict = Lr.predict(X_test)

acc, prec, rec, f1 = classification_performance_eval(Y_test, predict)
print( ' < train_test_split by_Logistic Regression_Binary_tts > ')
print('accuracy = %.4f' %acc)
print('precision = %.4f' %prec)
print('recall = %.4f ' %rec)
print('F1_score = %.4f' %f1, '\n')


from sklearn.model_selection import KFold

accuracy = []
precision = []
recall = []
f1_score = []

kf = KFold(n_splits = 3, random_state =42, shuffle = True)

for train_index, test_index in kf.split(X):

    X = np.array(X)
    Y = np.array(Y)
    x_train , x_test = X[train_index], X[test_index]
    y_train , y_test = Y[train_index], Y[test_index]
    

    Lr_k_fold = LogisticRegression(C=1, random_state=0)
    Lr_k_fold.fit(x_train, y_train)
    y_predict = Lr_k_fold.predict(x_test)
    

    acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)
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