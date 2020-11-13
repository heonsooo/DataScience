import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import pymysql
import celluloid
import time
import statsmodels.api as sm 
import moviepy.editor as mp


def load_dbscore_data(k):
    conn = pymysql.connect(host = 'localhost', user = 'Soo', password = '1234', db = 'data_science')
    curs = conn.cursor(pymysql.cursors.DictCursor)

    sql = 'select* from data_score'
    curs.execute(sql)

    data = curs.fetchall()

    curs.close()
    conn.close()
    if k == 'multi':
        X = [ ( t['attendance'] ,  t['homework'], t['midterm'] ) for t in data]
        
    elif k == 'simple':
        X = [ ( t['midterm'] ) for t in data]

    y = [ (t ['score']) for t in data]
    X = np.array(X)
    y = np.array(y)

    return X, y

def simple_linear(X,y): 
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const)

    ls= model.fit()
    ls.summary()
    ls_c = ls.params[0]  
    ls_m = ls.params[1]

    return ls_c , ls_m

def multi_linear(X,y): 
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const)
    ls= model.fit()
    ls.summary()
    #print(ls.summary())  
    ls_c = ls.params[0]  
    ls_m1 = ls.params[1]
    ls_m2 = ls.params[2]
    ls_m3 = ls.params[3]
    return ls_c, ls_m1, ls_m2, ls_m3

def simple_gradient_descent_vectorized(X, y, ls_c , ls_m ):

    start_time = time.time()
    epochs = 50401
    min_grad = 0.0001
    learning_rate = 0.001
    m , c = 0.0371, 0.0007
    n = len(y)
    c_grad , m_grad= 0.0 , 0.0


    for epoch in range(epochs):
        y_pred = m*X + c
        m_grad = (2*(y_pred-y)*X).sum()/n 
        c_grad = (2*(y_pred- y) ).sum()/n
    
        
        m = m - learning_rate * m_grad
        c = c - learning_rate * c_grad

        if(epoch % 1000 == 0):
            print("epoch %d : m_grad = %f , c_grad = %f , m= %f , c= %f" %(epoch, m_grad, c_grad, m, c))
            plt.scatter(X, y,  color = 'blue')

            plt.ylim(10, 100)
            plt.plot([min(X), max(X)],  [min(y_pred), max(y_pred)], color = 'red')

            plt.text(0, 80, 'm = %.7f'%m )
            plt.text(0, 76, 'c  = %.7f'%c )

            camera.snap() 
            


        if(abs(m_grad) < min_grad and abs(c_grad) < min_grad) :
            break

    end_time = time.time()
    print('%f seconds' %(end_time - start_time))

    print('\nFinal : ')
    print("gdv_m = %f. gdv_c =%f" %(m,c))
    print("ls_m  = %f, ls_c = %f" %(ls_m ,ls_c))

    print('\n animation 생성 중 입니다..')
    animation = camera.animate(interval=100, repeat=True)
    animation.save("201814132_이헌수.mp4")

    print('animation생성이 완료되었습니다. \n gif 생성 중입니다..')
    clip = mp.VideoFileClip("201814132_이헌수.mp4").speedx(4).write_gif('201814132_이헌수.gif')



    return print('\n       -simple linear regression 시각화 작업이 끝났습니다.-\n\n')

def Multi_gradient_descent_vectorized(X, y, ls_m1, ls_m2, ls_m3, ls_cm):
    print('       <multiple linear regression 시작합니다.>\n')
    start_time = time.time()
    epochs = 1000001
    min_grad = 0.0001
    learning_rate = 0.001
    
    m1,m2, m3 = 0.0, 0.0 , 0.0
    c = 0.0
    
    n = len(y)
    
    c_grad = 0.0
    m1_grad, m2_grad, m3_grad = 0.0, 0.0, 0.0

    for epoch in range(epochs):    
    
        y_pred = m1*X[:,0]  +m2*X[:,1]  +m3*X[:,2]  + c
        m1_grad = (2*(y_pred - y)*X[:,0]).sum()/n
        m2_grad = (2*(y_pred - y)*X[:,1]).sum()/n
        m3_grad = (2*(y_pred - y)*X[:,2]).sum()/n
        c_grad = (2*(y_pred - y)).sum()/n

        
        m1 = m1 - learning_rate * m1_grad
        m2 = m2 - learning_rate * m2_grad
        m3 = m3 - learning_rate * m3_grad
        c = c - learning_rate * c_grad 

        

        if ( epoch % 50000 == 0):
            print("epoch %d: m1_grad = %f, m2_grad = %f, m3_grad = %f, c_grad = %f, m1 = %f, m2 = %f, m3 = %f, c = %f" %(epoch, m1_grad, m2_grad,m3_grad, c_grad, m1, m2, m3, c) )
    
        if ( abs(m1_grad) < min_grad and abs(m2_grad) < min_grad and abs(m3_grad) < min_grad and abs(c_grad) < min_grad ):
            break

    end_time = time.time()
    print('%f seconds' %(end_time - start_time))

    print("\n\nFinal:")
    print("gdv_m1 = %f, gdv_m2 = %f, gdv_m3 = %f, gdv_c = %f" %(m1,m2,m3 , c) )
    print("ls_m1  = %f, ls_m2  = %f, ls_m3  = %f, ls_cm = %f" %(ls_m1,ls_m2,ls_m3, ls_cm) )


    return print('\n multiple linear regression 구현 작업이 끝났습니다.\n\n')

#simple linear regression 시각화
X_s, y_s = load_dbscore_data('simple')
ls_c , ls_m = simple_linear(X_s, y_s)
fig, ax = plt.subplots()
camera = celluloid.Camera(fig)
simple_gradient_descent_vectorized(X_s, y_s,ls_c , ls_m )



#multiple linear regression 구현 
X_m, y_m = load_dbscore_data('multi')
ls_cm, ls_m1, ls_m2, ls_m3 = multi_linear(X_m, y_m)
Multi_gradient_descent_vectorized(X_m, y_m, ls_m1, ls_m2, ls_m3, ls_cm)

