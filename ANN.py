
# coding: utf-8

# In[1]:

"""
This program is used to generate nets from scratch, and also to fine tune pre-existing nets
"""
"""
Import all required header files
"""
import keras
import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:
"""
Import all required modules
"""
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras import regularizers
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
model = Sequential()


# In[3]:


class ANN:
	"""
	This class ANN or Artificial Neural Network, generates neural networks from scratch using only the parameters entered by the user.

	:type X: float
	:parameter X: the input feature array

	:type Y: float
	:parameter Y: the labels of the input data

	:type input_dim: int
	:parameter input_dim: The number of columns in the input data

	:type nodes: int
	:parameter nodes: An integer array consisting of the number of nodes in each layer sequentially

	:type model: keras 
	:parameter model: The keras model variable

	:type activation: string
	:parameter activation: The activation function used in the output layer of the neural network

	:type loss: string
	:parameter loss: The loss function on which optimization is done

	:type optimizer: string
	:parameter optimizer: The optimizing function which acts on the loss to assign weights

	:type epochs: int
	:parameter epochs: The number of iterations that the neural network runs to assign weights to the nodes

	:type batch_size: int
	:parameter batch_size: The size of each batch of data being passed to the neural network for each iteration or epoch

	:type kernel_init: string
	:parameter kernel_init: The function used to assign the initial weights to the nodes
	"""

    
    def __init__(self,X,Y,nodes,model=model,activation='softmax',loss='categorical_crossentropy',optimizer='adam',epochs=10,batch_size=200,kernel_init='normal'):#constructor
        self.X              = [[X[i][j] for j in range(len(X[i]))] for i in range(len(X))]       
        self.Y              = [Y[i] for i in range(len(Y))]                                      
        self.input_dim      = nodes[0]                                                           
        self.nodes          = [nodes[i] for i in range(len(nodes))]                              
        self.model          = model                                                              
        self.layers         = len(nodes)                                                                                      
        self.activation     = activation                                                                                     
        self.loss           = loss                                                               
        self.optimizer      = optimizer                                                     
        self.epochs         = epochs                                                             
        self.batch_size     = batch_size                                                         
        self.kernel_init    = kernel_init                                                       
                                                                    
        
        self.model.add(Dense(self.nodes[0],input_dim=self.nodes[0], kernel_initializer=self.kernel_init,activation='relu'))  
        
       
        if(self.layers>2):
            for i in range(1,self.layers-1):
                self.model.add(Dense(self.nodes[i], kernel_initializer=self.kernel_init,activation='relu'))            
        
        self.model.add(Dense(self.nodes[(self.layers)-1], kernel_initializer=self.kernel_init,activation=self.activation))  
        
        self.model.compile(loss=self.loss,optimizer=self.optimizer,metrics=['accuracy'])
        print(model.summary())
        
        np.random.seed(7)
        self.model.fit(np.array(self.X),np.array(self.Y),epochs= self.epochs,batch_size=self.batch_size,verbose=2)
        
        
        scores               = self.model.evaluate(np.array(self.X), np.array(self.Y),verbose=0)
        print("\n%s: %.2f%%" % ("accuracy of the classifier on the training dataset ", scores[1]*100))
            
     
    def check_accuracy(self,P,Q):
    	"""
    	The check accuracy function is used to check the accuracy of the generated neural network on validation data

    	:type P: int
    	:parameter P: P is the feature data of the cross validation dataset

    	:type Q: int
    	:parameter Q: Q is the label data of the cross validation dataset
    	"""	
        scores               = self.model.evaluate(np.array(P), np.array(Q),verbose=0)
        print("\n%s: %.2f%%" % ("accuracy of the classifier on the validation dataset ", scores[1]*100))
        
    def test_code(self,P,Q):
    	"""
    	The test_code function is used to test whether the neural network generated on some standard data. If the accuracy of the 
    	neural net is greater than 97%, the network passes the test.
    	:type P: int
    	:parameter P: P is the feature data of the standard data

    	:type Q: int
    	:parameter Q: Q is the label data of the standard data
    	"""
        np.random.seed(7)
        scores               = self.model.evaluate(np.array(P), np.array(Q),verbose=0)
        if(scores[1]>0.97):
            print("\n%s: %.2f%%" % ("accuracy of the classifier on the validation dataset ", scores[1]*100))
            print("code passes the test")
            
        else:
            print("code has failed the test")
            
        
        
    def predict(self,X_new):
   		"""
    	The predict function is used to generate predictions for data whose target variables are unknown.
    	:type X_new: int
    	:parameter X_new: Contains the feature data for which the target variables values are unknown
   	 	"""
        predictions 	 = self.model.predict(X_new)
        df 				 = X_new[:,0]
        df['predictions']=predictions
        df.to_csv('predictions.csv')
        
    
        
    def save_model_and_weights(self):
    	"""
    	The save_model_and_weights function saves the model as a json file, and the weights as a .h5 file
    	"""
        #saving model and weights
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")
        
   

class Retrain:
	"""
	The Retrain class is used to fine tune pre-trained neural networks and make predictions using these tuned nets.
	:type model: keras
	:parameter model: The keras model variable

	:type json_file: json
	:paramete json_file: The json file which contains the structure of the pre-trained model

	:type h5_file: h5
	:parameter h5_file: The .h5 file which contains the weights of the pre-trained model

	:type output_dim: int
	:parameter output: The number of classes of the incremental training dataset

	:type X: int
	:parameter X: The feature data of the incremental training dataset

	:type Y: int
	:parameter Y: The label data of the incremental training dataset

	:type activation: string
	:parameter activation: The activation function of the output layer of the neural network.

	:type epochs: int
	:parameter epochs: Number of iterations that the neural network must run per batch

	:type batch_size: int
	:parameter batch_size: Number of training examples being fed in to the net for each batch
	"""
    
    def __init__(self,json_file,h5_file,output_dim,X,Y,activation,epochs=10,batch_size=200):
        self.model          = model
        self.json_file      = json_file
        self.h5_file 		= h5_file
        self.output_dim     = output_dim
        self.X              = [[X[i][j] for j in range(len(X[i]))] for i in range(len(X))]
        self.Y              = [Y[i] for i in range(len(Y))] 
        self.activation 	= activation
        self.epochs         = epochs
        self.batch_size     = batch_size
        
        #loading the new model
        json_file = open(self.json_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model2= model_from_json(loaded_model_json)
        # load weights into new model
        model2.load_weights(self.h5_file)
        print("Loaded model from disk")
        
        self.model2.layers.pop()
        self.model2.add(Dense(self.output_dim,activation=self.activation))
        
        self.model.fit(np.array(self.X),np.array(self.Y),epochs= self.epochs,batch_size=self.batch_size,verbose=2)
        
        scores               = self.model.evaluate(np.array(self.X), np.array(self.Y))
        print("\n%s: %.2f%%" % ("accuracy of the incrementally trained classifier on the training dataset ", scores[1]*100))
            
        
    def check_accuracy(self,choice,P,Q):
    	"""
    	The check accuracy function is used to check the accuracy of fine tuned neural network on validation data

    	:type P: int
    	:parameter P: P is the feature data of the cross validation dataset

    	:type Q: int
    	:parameter Q: Q is the label data of the cross validation dataset
    	""" 
        scores               = self.model2.evaluate(np.array(P), np.array(Q))
        print("\n%s: %.2f%%" % ("accuracy of the incrementally trained classifier on the validation dataset ", scores[1]*100))
        
       
    def predict(self,X_new):
    	"""
    	The predict function is used to generate predictions for data whose target variables are unknown.
    	:type X_new: int
    	:parameter X_new: Contains the feature data for which the target variables values are unknown
   	 	"""
        predictions      = self.model2.predict(X_new)
        df 				 = X_new[:,0]
        df['predictions']=predictions
        df.to_csv('predictions2.csv')    


# In[4]:


def test(): 
	"""
	The test function uses the test_code module from the ANN class to test whether the neural network generator works properly.
	"""
    from keras.datasets import mnist
    np.random.seed(7)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    num_pixels              			 = X_train.shape[1] * X_train.shape[2]
    X_train								 = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
    X_test								 = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
    X_train 							 = X_train.astype('float32')
    X_test 								 = X_test.astype('float32')
    X_train 							/= 255
    X_test 								/= 255
    y_train 							 = np_utils.to_categorical(y_train,10)
    y_test								 = np_utils.to_categorical(y_test,10)
    num_classes 						 = y_test.shape[0]
    c 									 = ANN(X_train,y_train,[784,10])
    proceed								 = c.test_code(X_test,y_test)
    return proceed


# In[5]:


print('''Firstly, open a terminal and launch python.
		 
		 Then, import ANN. This will allow you to use the neural network generator.

		 X,Y and input_dim are the essential parameters.The remaining parameters are all optional.
         
         An example of a call to the constructor is :      c = ANN.ANN(X,Y,20)
         
         However, if we want to replace certain defaults,  c = ANN.ANN(X,Y,20,optimixer='RMSProp')''' )

