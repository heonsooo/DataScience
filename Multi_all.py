import pymysql
import numpy as np
from sklearn.model_selection import KFold   
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn import svm                                 # SVM 

from sklearn.preprocessing import StandardScaler        # Random forest 사용
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression     # Logistical Regression 



'''
def calcurator_tts(y_test, y_predict): 
    cnf_matrix = confusion_matrix(y_test, y_predict)


    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP, FN, TP, TN =  FP.astype(float), FN.astype(float) , TP.astype(float), TN.astype(float)
   
    ACC = (TP+TN)/(TP+FP+FN+TN)
    prec =TP/(TP+FP)
    recall = TP/(TP+FN)
    f1_score = 2*prec*recall / (prec + recall)
    print('accuracy =', ACC)
    print('precision = ' ,prec)
    print('recall =  ' ,recall)
    print('F1_score = ', f1_score, '\n')
'''

def SVM_MUlti_Class(X_train, X_test,y_train, y_test ):                      #이거만 완성
    classifier = svm.SVC(kernel='linear').fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_predict)
    name = 'SVM_Multi-Class'
    return cnf_matrix, name


def Random_Forest_Multi_Class(X_train, X_test,y_train, y_test): 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42).fit(X_train, y_train)  # n_estimatiors : tree 개수 
    y_predict = classifier.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_predict)
    # calcurator_tts(y_test, y_predict)
    
    name = 'Random_Forest_Multi-Class'
    
    return cnf_matrix, name


def Logistic_Regression_Multi_Class(X_train, X_test,y_train, y_test):
    Lr = LogisticRegression(C=1, random_state=0, solver= 'liblinear' ).fit(X_train, y_train)
    y_predict = Lr.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_predict)
    name = 'Logistic Regression_Multi-Class'
    return cnf_matrix, name



def select_model(a,X_train, X_test,y_train, y_test ):
    if a == 'svm' :
        SVM_MUlti_Class(X_train, X_test,y_train, y_test )

    elif a == 'rF' :
        Random_Forest_Multi_Class(X_train, X_test,y_train, y_test)

    elif a =='LR':
        Logistic_Regression_Multi_Class(X_train, X_test,y_train, y_test)






def calcurator(cnf_matrix): 

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP, FN, TP, TN =  FP.astype(float), FN.astype(float) , TP.astype(float), TN.astype(float)
   
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    Precision =TP/(TP+FP)
    Recall = TP/(TP+FN)
    F1_score = 2*Precision*Recall / (Precision + Recall)

    return Accuracy, Precision, Recall, F1_score


def tts (X,y,a):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33 , random_state=42)

    select_model(a)
    calcurator(cnf_matrix)
    return X_train, X_test, y_train, y_test

def kfold(X, y,a):
    F1_score , Recall, Precision, Accuracy = [], [], [], []
    mean_acc ,mean_prec , mean_rec, mean_f1 = [0,0,0],[0,0,0],[0,0,0],[0,0,0]
    k= int(4)
    kf = KFold(n_splits =k, random_state = 42, shuffle = True)


    for train_index, test_index in kf.split(X):
        X_train , X_test = X[train_index], X[test_index]
        y_train , y_test = y[train_index], y[test_index]

        select_model(a, X_train, X_test,y_train, y_test)
        
        calcurator(cnf_matrix)
        




    for i in range(0, k):
        for j in range(0,3):
            mean_acc[j] += float(Accuracy[i][j])
            mean_prec[j] += float(Precision[i][j])
            mean_rec[j] += float(Recall[i][j])
            mean_f1[j] += float(F1_score[i][j])

    for j in range(0,3):
        mean_acc[j] = round(mean_acc[j]/k,4)
        mean_prec[j] = round(mean_prec[j]/k,4)
        mean_rec[j] =round(mean_rec[j]/k,4)
        mean_f1[j] = round(mean_f1[j]/k,4)
    print('accuracy_평균 = ', mean_acc)
    print('precision_평균 = ', mean_prec )
    print('recall_평균 = ', mean_rec)
    print('f1_score_평균 = ',mean_f1,'\n')
    print( ' < K-fold cross validation by_',name)






####################

conn = pymysql.connect(host = 'localhost', user = 'Soo', password = '1234' , db = 'data_science')
curs = conn.cursor(pymysql.cursors.DictCursor)
sql = 'select*from db_score'
curs.execute(sql)

data = curs.fetchall()
curs.close()
conn.close()

X = [ ( t['homework'], t['discussion'], t['midterm'] )  for t in data ]
y = [ 0 if (t['grade'] == 'A')  else 1 if (t['grade'] == 'B') else 2 for t in data ]            # muliti class 

# y = [ 1 if (t['grade'] == 'B') else -1 for t in data ]                                        # binary 
X = np.array(X)
y = np.array(y)

################### 고정 #############