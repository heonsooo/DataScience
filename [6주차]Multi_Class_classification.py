import pymysql
import numpy as np
from sklearn.model_selection import KFold   
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn import svm                                 # SVM 

from sklearn.preprocessing import StandardScaler        # Random forest 사용
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression     # Logistical Regression 


def SVM_MUlti_Class(X_train, X_test,y_train, y_test ):                      
    classifier = svm.SVC(kernel='linear').fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    name = 'SVM_Multi-Class'
    
    return y_test, y_predict, name


def Random_Forest_Multi_Class(X_train, X_test,y_train, y_test): 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42).fit(X_train, y_train)  # n_estimatiors : tree 개수 
    y_predict = classifier.predict(X_test)
    
    name = 'Random_Forest_Multi-Class'
    return y_test, y_predict, name


def Logistic_Regression_Multi_Class(X_train, X_test,y_train, y_test):
    Lr = LogisticRegression(C=1, random_state=0, solver= 'liblinear' ).fit(X_train, y_train)
    y_predict = Lr.predict(X_test)
    
    name = 'Logistic Regression_Multi-Class'
    return y_test, y_predict, name

def multi_measure_calcurator_kfold(y_test, y_predict, Accuracy, Precision, Recall, F1_score ):
    x1,x2,x3 = [0,0,0], [0,0,0],[0,0,0]
    
    for y, yp in zip(y_test,y_predict):
        if y == 0:
            if y == 0 and yp ==0 :
                x1[0] += 1
            elif y==0 and yp ==1 :
                x1[1] += 1
            elif y  == 0 and yp == 2:
                x1[2] += 1 

        elif y == 1:
            if y == 1 and yp ==0 :
                x2[0] += 1
            elif y==1 and yp ==1 :
                x2[1] += 1
            elif y  == 1 and yp == 2:
                x2[2] += 1 

        elif y == 2:
            if y == 2 and yp ==0 :
                x3[0] += 1
            elif y==2 and yp ==1 :
                x3[1] += 1
            elif y  == 2 and yp == 2:
                x3[2] += 1 

    matrix = [x1, x2, x3]
    total = 0        
    for j in matrix :
        for i in j :
            total += i  

    colum_sum_matrix = [sum(p) for p in zip(*matrix)] 
    row_sum_matrix = [sum(e) for e in matrix]
    
    tp = x1[0] + x2[1] + x3[2]
    accuracy = round(tp / total,4)
    
    try :
        precision_A = round(x1[0]/(colum_sum_matrix[0]), 4)
        precision_B= round(x2[1]/(colum_sum_matrix[1]), 4)
        precision_C= round(x3[2]/(colum_sum_matrix[2]), 4)

        recall_A = round(x1[0]/(row_sum_matrix[0]), 4)
        recall_B = round(x2[1]/(row_sum_matrix[1]), 4)
        recall_C = round(x3[2]/(row_sum_matrix[2]), 4)

        f1_score_A = round(2*precision_A*recall_A / (precision_A + recall_A), 4)
        f1_score_B = round(2*precision_B*recall_B / (precision_B + recall_B), 4)
        f1_score_C = round(2*precision_C*recall_C / (precision_C + recall_C) , 4)


        precision_all = [precision_A , precision_B, precision_C]
        recall_all = [recall_A, recall_B, recall_C]
        f1_score_all = [f1_score_A, f1_score_B, f1_score_C]


    except : 
       print('에러가 발생했꾼요.. ㅠ')

    Accuracy.append(accuracy)
    Precision.append(precision_all)
    Recall.append(recall_all)
    F1_score.append(f1_score_all)

    return Accuracy, Precision, Recall, F1_score



def multi_measure_calcurator(y_test, y_predict, key):
    x1,x2,x3 = [0,0,0], [0,0,0],[0,0,0]
    
    for y, yp in zip(y_test,y_predict):
        if y == 0:
            if y == 0 and yp ==0 :
                x1[0] += 1
            elif y==0 and yp ==1 :
                x1[1] += 1
            elif y  == 0 and yp == 2:
                x1[2] += 1 

        elif y == 1:
            if y == 1 and yp ==0 :
                x2[0] += 1
            elif y==1 and yp ==1 :
                x2[1] += 1
            elif y  == 1 and yp == 2:
                x2[2] += 1 

        elif y == 2:
            if y == 2 and yp ==0 :
                x3[0] += 1
            elif y==2 and yp ==1 :
                x3[1] += 1
            elif y  == 2 and yp == 2:
                x3[2] += 1 

    matrix = [x1, x2, x3]
    total = 0        
    for j in matrix :
        for i in j :
            total += i  

    colum_sum_matrix = [sum(p) for p in zip(*matrix)] 
    row_sum_matrix = [sum(e) for e in matrix]
    
    tp = x1[0] + x2[1] + x3[2]
    accuracy = round(tp / total,4)
    
    try :
        precision_A = round(x1[0]/(colum_sum_matrix[0]), 4)
        precision_B= round(x2[1]/(colum_sum_matrix[1]), 4)
        precision_C= round(x3[2]/(colum_sum_matrix[2]), 4)

        recall_A = round(x1[0]/(row_sum_matrix[0]), 4)
        recall_B = round(x2[1]/(row_sum_matrix[1]), 4)
        recall_C = round(x3[2]/(row_sum_matrix[2]), 4)

        f1_score_A = round(2*precision_A*recall_A / (precision_A + recall_A), 4)
        f1_score_B = round(2*precision_B*recall_B / (precision_B + recall_B), 4)
        f1_score_C = round(2*precision_C*recall_C / (precision_C + recall_C) , 4)


        precision_all = [precision_A , precision_B, precision_C]
        recall_all = [recall_A, recall_B, recall_C]
        f1_score_all = [f1_score_A, f1_score_B, f1_score_C]


    except : 
       print('에러가 발생했꾼요.. ㅠ')

    return accuracy, precision_all, recall_all, f1_score_all


def tts (X,y,a):
    X_train, X_test, y_train, y_test = train_test_split(X, y ,shuffle= True, test_size = 0.33333 , random_state=42)

    if a == 'svm' :
        y_test, y_predict , name = SVM_MUlti_Class(X_train, X_test,y_train, y_test )

    elif a == 'Rf' :
        y_test, y_predict , name =Random_Forest_Multi_Class(X_train, X_test,y_train, y_test)


    elif a =='Lr':
        y_test, y_predict , name = Logistic_Regression_Multi_Class(X_train, X_test,y_train, y_test)

    else: 
        print('알고리즘을 확인해주세요.')

    ACC, prec, recall ,f1_score = multi_measure_calcurator(y_test, y_predict,0)

    print( ' < train_test_splitby_',name,'> \n ' )
    '''
    print('accuracy =', ACC , )
    print('precision = ' ,prec, '    순서 :[A, B, C]' )
    print('recall    = ' ,recall, '순서 : [A, B, C]')
    print('F1_score  = ', f1_score,'순서 : [A, B, C] \n')
    '''
    print('\n Accuracy =', ACC )
    print('Grade A 의  Precision = {a}, Recall = {b}, F1_score = {c}' .format(a= prec[0], b= recall[0], c= f1_score[0]))
    print('Grade B 의  Precision = {a}, Recall = {b}, F1_score = {c}' .format(a= prec[1], b= recall[1], c= f1_score[1]))
    print('Grade C 의  Precision = {a}, Recall = {b}, F1_score = {c}' .format(a= prec[2], b= recall[2], c= f1_score[2]) ,'\n')


def kfold(X, y,a):
    F1_score , Recall, Precision, Accuracy = [], [], [], []
    mean_acc ,mean_prec , mean_rec, mean_f1 = 0,[0,0,0],[0,0,0],[0,0,0]
    k= int(4)
    kf = KFold(n_splits =k, random_state = 42, shuffle = True)


    for train_index, test_index in kf.split(X):
        X_train , X_test = X[train_index], X[test_index]
        y_train , y_test = y[train_index], y[test_index]

        if a == 'svm' :
            y_test, y_predict , name = SVM_MUlti_Class(X_train, X_test,y_train, y_test )
            ACC, prec, recall ,f1_score = multi_measure_calcurator_kfold(y_test, y_predict, Accuracy, Precision, Recall, F1_score )

        elif a == 'Rf' :
            y_test, y_predict , name =Random_Forest_Multi_Class(X_train, X_test,y_train, y_test)
            ACC, prec, recall ,f1_score = multi_measure_calcurator_kfold(y_test, y_predict, Accuracy, Precision, Recall, F1_score)

        elif a =='Lr':
            y_test, y_predict , name = Logistic_Regression_Multi_Class(X_train, X_test,y_train, y_test)
            ACC, prec, recall ,f1_score = multi_measure_calcurator_kfold(y_test, y_predict, Accuracy, Precision, Recall, F1_score )

        else: 
            print('알고리즘을 확인해주세요.')

    for i in range(0, k):
        for j in range(0,3):
            mean_prec[j] += float(prec[i][j])
            mean_rec[j] += float(recall[i][j])
            mean_f1[j] += float(f1_score[i][j])

    

    for j in range(0,3):
        mean_prec[j] = round(mean_prec[j]/k,4)
        mean_rec[j] =round(mean_rec[j]/k,4)
        mean_f1[j] = round(mean_f1[j]/k,4)
    for p in ACC:
        mean_acc += p      

    mean_acc = round(mean_acc/k,4)

        
    print( ' < K-fold cross validation by_',name,'> \n ')
    print('Accuracy_평균  = ',mean_acc)
    '''
    print('precision_평균 = ',mean_prec,'    순서 :[A, B, C]')
    print('recall_평균    = ',mean_rec, '    순서 :[A, B, C]')
    print('f1_score_평균  = ',mean_f1,  '    순서 :[A, B, C] \n')
    '''
    print('Grade A 의  Precision = {a}, Recall = {b}, F1_score = {c}' .format(a= mean_prec[0], b= mean_rec[0], c= mean_f1[0]))
    print('Grade B 의  Precision = {a}, Recall = {b}, F1_score = {c}' .format(a= mean_prec[1], b= mean_rec[1], c= mean_f1[1]))
    print('Grade C 의  Precision = {a}, Recall = {b}, F1_score = {c}' .format(a= mean_prec[2], b= mean_rec[2], c= mean_f1[2]) ,'\n')




















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




tts(X,y, 'svm')
tts(X,y, 'Rf')
tts(X,y, 'Lr')

kfold(X,y, 'svm')
kfold(X,y, 'Rf')
kfold(X,y, 'Lr')
