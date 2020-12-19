import pymysql
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report


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

conn = pymysql.connect(host = 'localhost', user = 'Soo', password = '1234' , db = 'data_science')
curs = conn.cursor(pymysql.cursors.DictCursor)

sql = 'select*from db_score'
curs.execute(sql)
data = curs.fetchall()

curs.close()
conn.close()

X = [ ( t['homework'], t['discussion'], t['midterm'] )  for t in data ]
X = np.array(X)

Y = [ 1 if (t['grade'] == 'B') else -1  for t in data ]
Y = np.array(Y)

# from sklearn.model_selection import train_test_split
X_train , X_test , Y_train, Y_test = train_test_split(X, Y, test_size = 0.333 , random_state = 42)                #test 사이즈(비율)을 통해 train : test = 0.66 : 0.33 (2:1 비율)로 설정
kernel_list = ['rbf', 'linear','sigmoid','poly']

for a in kernel_list :
    clf = svm.SVC(kernel= a , gamma='auto', C= 1)
    clf.fit(X_train,Y_train)
    predict = clf.predict(X_test)

    acc, prec, rec, f1 = classification_performance_eval(Y_test, predict)

    print( ' < train_test_split by_SVM-Binary >  kernel = ' , a)
    print('accuracy = %.4f' %acc)
    print('precision = %.4f' %prec)
    print('recall = %.4f ' %rec)
    print('F1_score = %.4f' %f1, '\n')



from sklearn.model_selection import KFold

accuracy = []
precision = []
recall = []
f1_score = []


kf = KFold(n_splits = 4, random_state = 42, shuffle = True)
for a in kernel_list:
    for train_index, test_index in kf.split(X):

        
        X_train , X_test = X[train_index], X[test_index]
        Y_train , Y_test = Y[train_index], Y[test_index]
        
        clf = svm.SVC(kernel= a , gamma='auto', C= 1)
        clf_model = clf.fit(X_train,Y_train)
        y_predict = clf_model.predict(X_test)

        
        acc, prec, rec, f1 = classification_performance_eval(Y_test, y_predict)
        accuracy.append(acc)
        precision.append(prec)
        recall.append(rec)
        f1_score.append(f1)

    import statistics 

    # kfold는 운에 좌우되지않고 한결같은 값이 나옴. 
    print( ' < K-fold cross validation by_SVM-Binary >  kernel = ' , a)
    print('accuracy_평균 = ', round(statistics.mean(accuracy),4))
    print('precision_평균 = ', round(statistics.mean(precision),4))
    print('recall_평균 = ', round(statistics.mean(recall),4))
    print('f1_score_평균 = ', round(statistics.mean(f1_score),4),'\n')


#   tts  precision 제일 높은 것 : poly    
#   k fold precision이 제일 높은 것 : rbf 
