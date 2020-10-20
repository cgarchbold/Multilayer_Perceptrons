import numpy as np

# input any array-like, pass each through sigmoids
def sigmoid_F(X):
    X = np.array(X)
    return 1/(1 + np.exp(-X))

def sigmoid_derivative(X):
    X = np.array(X)
    return sigmoid_F(X) * (1 - sigmoid_F(X))


# Class to represent a nueral network 
class NueralNetwork():

    def __init__(self, hidden_layers):
        np.random.seed(1)
        
        #initialize weights 

    def __str__(self):
        return 'hello'

    def predict(self, x):
        return x

    def train(self, inputs , epochs):
        

        for epoch in range(epochs):
            # declarem inputs
            inputs = inputs # FIX

            #forward input 




