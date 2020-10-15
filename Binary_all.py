import pymysql
import numpy as np
from sklearn.model_selection import KFold   
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import statistics 


from sklearn import svm                                 # SVM 

from sklearn.preprocessing import StandardScaler        # Random forest 사용
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression     # Logistical Regression 

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
    #tp+=1 if y==1 and yp ==1 else fn+=1 if y == 1 and yp == -1 else fp += 1 if y == -1 and yp == 1 else tn += 1 if y == -1 and yp == -1 for y, yp in zip(y,y_predict)

    try :
        accuracy = (tp+ tn)/(tp+tn+fp+fn)
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)
        f1_score = 2*precision*recall / (precision + recall)
        
    except:
        if tp == 0: 
            precision, recall, f1_score = 0 ,0 ,0 
    
    return accuracy, precision, recall, f1_score


def SVM_Binary(X_train, X_test,y_train, y_test ):                      
    
    kernel_list = ['rbf', 'linear','sigmoid','poly']

    for a in kernel_list :
        classifier = svm.SVC(kernel= a , gamma='auto', C= 1).fit(X_train, y_train)
        y_predict = classifier.predict(X_test)

        acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)

        print( ' < train_test_split by_SVM-Binary >  kernel = ' , a)
        print('accuracy = %.4f' %acc)
        print('precision = %.4f' %prec)
        print('recall = %.4f ' %rec)
        print('F1_score = %.4f' %f1, '\n')
        
    


def Random_Forest_Binary(X_train, X_test,y_train, y_test): 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42).fit(X_train, y_train)  # n_estimatiors : tree 개수 
    y_predict = classifier.predict(X_test)

    name = 'Random_Forest_Binary'
    return y_test, y_predict, name


def Logistic_Regression_Binary(X_train, X_test,y_train, y_test):
    Lr = LogisticRegression(C=1, random_state=0, solver= 'liblinear' ).fit(X_train, y_train)
    y_predict = Lr.predict(X_test)
    name = 'Logistic Regression_Binary'
    return y_test, y_predict, name


def calcurator(cnf_matrix,Accuracy, Precision, Recall, F1_score ): 

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP, FN, TP, TN =  FP.astype(float), FN.astype(float) , TP.astype(float), TN.astype(float)
   
    ACC = (TP+TN)/(TP+FP+FN+TN)
    prec =TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score = 2*prec*recall / (prec + recall)

    Accuracy.append(ACC)
    Precision.append(prec)
    Recall.append(recall)
    F1_score.append(f1_score)

    return Accuracy, Precision, Recall, F1_score

def calcurator_tts(cnf_matrix): 

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP, FN, TP, TN =  FP.astype(float), FN.astype(float) , TP.astype(float), TN.astype(float)
   
    ACC = (TP+TN)/(TP+FP+FN+TN)
    prec =TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score = 2*prec*recall / (prec + recall)

    return ACC, prec, recall, f1_score


def tts (X,y,a):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33 , random_state=42)
    
    if a == 'svm' :
        SVM_Binary(X_train, X_test,y_train, y_test)
    
    elif a == 'Rf' :
        y_test, y_predict , name =Random_Forest_Binary(X_train, X_test,y_train, y_test)
        acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)

        print( ' < train_test_splitby_',name,'> \n')
        print('accuracy =%.4f' %acc)
        print('precision = %.4f' %prec)
        print('recall = %.4f ' %rec)
        print('F1_score = %.4f' %f1, '\n')

    elif a =='Lr':
        y_test, y_predict, name = Logistic_Regression_Binary(X_train, X_test,y_train, y_test)
        acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)

        print( ' < train_test_splitby_',name,'> \n')
        print('accuracy =%.4f' %acc)
        print('precision = %.4f' %prec)
        print('recall = %.4f ' %rec)
        print('F1_score = %.4f' %f1, '\n')
                
    




def kfold(X, y,a):
    accuracy, precision, recall, f1_score= [], [], [], []
    
    k= int(4)
    kf = KFold(n_splits =k, random_state = 42, shuffle = True)

    if a == 'svm':
        kernel_list = ['rbf', 'linear','sigmoid','poly']
        for a in kernel_list:
            for train_index, test_index in kf.split(X):

                
                X_train , X_test = X[train_index], X[test_index]
                y_train , y_test = y[train_index], y[test_index]
                
                clf = svm.SVC(kernel= a , gamma='auto', C= 1)
                clf_model = clf.fit(X_train,y_train)
                y_predict = clf_model.predict(X_test)

                
                acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)
                accuracy.append(acc)
                precision.append(prec)
                recall.append(rec)
                f1_score.append(f1)
            print( ' < K-fold cross validation by_SVM-Binary >  kernel = ' , a)
            print('accuracy_평균 = ', round(statistics.mean(accuracy),4))
            print('precision_평균 = ', round(statistics.mean(precision),4))
            print('recall_평균 = ', round(statistics.mean(recall),4))
            print('f1_score_평균 = ', round(statistics.mean(f1_score),4),'\n')




    else: 
        for train_index, test_index in kf.split(X):
            X_train , X_test = X[train_index], X[test_index]
            y_train , y_test = y[train_index], y[test_index]

            if a == 'Rf' :
                y_test, y_predict, name = Random_Forest_Binary(X_train, X_test,y_train, y_test)

            elif a =='Lr':
                y_test, y_predict, name = Logistic_Regression_Binary(X_train, X_test,y_train, y_test)

            else: 
                print('알고리즘을 확인해주세요.')

            acc, prec, rec, f1 = classification_performance_eval(y_test, y_predict)
            accuracy.append(acc)
            precision.append(prec)
            recall.append(rec)
            f1_score.append(f1)
        print( ' < K-fold cross validation by_',name,'> \n')
        print('accuracy_평균 = ', round(statistics.mean(accuracy),4))
        print('precision_평균 = ', round(statistics.mean(precision),4))
        print('recall_평균 = ', round(statistics.mean(recall),4))
        print('f1_score_평균 = ', round(statistics.mean(f1_score),4),'\n')

conn = pymysql.connect(host = 'localhost', user = 'Soo', password = '1234' , db = 'data_science')
curs = conn.cursor(pymysql.cursors.DictCursor)
sql = 'select*from db_score'
curs.execute(sql)

data = curs.fetchall()
curs.close()
conn.close()

X = [ ( t['homework'], t['discussion'], t['midterm'] )  for t in data ]
# y = [ 0 if (t['grade'] == 'A')  else 1 if (t['grade'] == 'B') else 2 for t in data ]            # muliti class 

y = [ 1 if (t['grade'] == 'B') else -1 for t in data ]                                        # binary 
X = np.array(X)
y = np.array(y)




tts(X,y, 'svm')
tts(X,y, 'Rf')
tts(X,y, 'Lr')

kfold(X,y, 'svm')
kfold(X,y, 'Rf')
kfold(X,y, 'Lr')