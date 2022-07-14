import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# accessing and indexing the data



from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.40,random_state=0)


f1=pd.read_csv("jk.csv")
x=f1.iloc[0:100, 0:3].values
y=f1.iloc[0:100, 3:4].values
x_train=f1.iloc[0:70,0:3]
y_train=f1.iloc[0:70,3:4]
x_test=f1.iloc[70:100,0:3]
y_test=f1.iloc[70:100,3:4]

l=np.arange(30).reshape(30,1)
for x in np.nditer(l.T):
    print(x)
  
#parameters
n_input=3
n_output=1
lr=0.5
ep=250

x_data=np.array([
  [1,1,1]
  ])
    

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

#weights




W1=tf.Variable(tf.random_uniform([n_input,20],-1.2, 1.2))
W2=tf.Variable(tf.random_uniform([20,20],-1.2, 1.2))
W3=tf.Variable(tf.random_uniform([20,20],-1, 1))
W4=tf.Variable(tf.random_uniform([20,1],-1, 1))
#bias
b1=tf.Variable(tf.zeros([20]),name="Bias1")
b2=tf.Variable(tf.zeros([20]),name="Bias1")
b3=tf.Variable(tf.zeros([20]),name="Bias1")
b4=tf.Variable(tf.zeros([1]),name="Bias1")

L2=tf.sigmoid(tf.matmul(X,W1)+b1)
hy=tf.sigmoid(tf.matmul(L2,W2)+b2)
he=tf.sigmoid(tf.matmul(hy,W3)+b3)
ha=tf.sigmoid(tf.matmul(he,W4)+b4)


cost=tf.reduce_mean(-Y*tf.log(ha)-(1-Y)*tf.log(1-ha))
optimizer=tf.train.GradientDescentOptimizer(lr).minimize(cost)

init=tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)
   
    
    for step in range(ep):
        session.run(optimizer,feed_dict={X:x_train,Y:y_train})
        if step %1000==0:
            print (session.run(cost,feed_dict={X:x_train,Y:y_train}))
            
        answer=tf.equal(tf.floor(ha+0.5),Y)
        accuracy=tf.reduce_mean(tf.cast(answer,"float"))
        c=( session.run([ha],feed_dict={X:x_test}))
        print(c)
        
        
        de=( accuracy.eval({X:x_train,Y:y_train}))
        df=de*100
        d=100-df
        print(d)
        
#        plt.plot(step,d)
        
        g=29
        plt.scatter(l,y_test,label="desired output",color='red')
        plt.scatter(l,c,label="network output",color='green')
        plt.xlim(0,30)
        plt.xlabel('samples')
        plt.ylabel('Accept/Reject')
        plt.title('predicted and desired')
        plt.legend()
        plt.show()
        print('The W1:')
        print(session.run(W1))
        print('The W2:')
        print(session.run(W2))
        print('The W3:')
        print(session.run(W3))
        print('The W4:')
        print(session.run(W4))
        print('The B1:')
        print(session.run(b1))
        print('The B2:')
        print(session.run(b2))    
        print('The B3:')
        print(session.run(b3))
        print('The B4:')
        print(session.run(b4))


        
        
    
        
session.close()

        
