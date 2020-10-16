def SVM_MUlti_Class(X_train, X_test,y_train, y_test ):                      
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
    
    name = 'Random_Forest_Multi-Class'
    return cnf_matrix, name


def Logistic_Regression_Multi_Class(X_train, X_test,y_train, y_test):
    Lr = LogisticRegression(C=1, random_state=0, solver= 'liblinear' ).fit(X_train, y_train)
    y_predict = Lr.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_predict)
    name = 'Logistic Regression_Multi-Class'
    return cnf_matrix, name



import numpy as np

def multi_measure_calculator(y_test, y_predict):
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
    for i in matrix :
        total += i
    matrix = np.array(matrix)


    
    accuracy = (matrix[0,0] + matrix[1,1], matrix[2,2])/ total
    precision_A = (matrix[0,0])/(matrix[0])
    precision_B = (matrix[1,1])/(matrix[1])
    precision_C = (matrix[2,2])/(matrix[2])

    recall_A = (matrix[0,0])/(matrix[:,0])
    recall_B = (matrix[1,1])/(matrix[:,1])
    recall_C = (matrix[2,2])/(matrix[:,2])

    f1_score_A = 2*precision_A*recall_A / (precision_A + recall_A)
    f1_score_B = 2*precision_B*recall_B / (precision_B + recall_B)
    f1_score_C = 2*precision_C*recall_C / (precision_C + recall_C) 


    precision = [precision_A , precision_B, precision_C]
    recall = [recall_A, recall_B, recall_C]
    f1_score = [f1_score_A, f1_score_B, f1_score_C]


    return accuracy, precision, recall, f1_score