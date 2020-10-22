import numpy as np
import pandas as pd
import matplotlib as plt

def sigmoid_F(X):
    """
    Sigmoid Function - Preform the sigmoid function on array_like X
    """
    X = np.array(X)
    return 1/(1 + np.exp(-X))

def sigmoidDerivative_F(X):
    """
    Sigmoid Derivative Function - Preform the sigmoid function on array_like X
    """
    X = np.array(X)
    return sigmoid_F(X) * (1 - sigmoid_F(X))

def softmax_F(X):
    """
    Sofmax Function - Preform the softmax function on array_like X
    """
    X = np.array(X)
    return np.exp(X)/sum(np.exp(X))

def softmaxDerivative_F(X):
    """
    Sofmax Derivative Function - Preform the softmax derivative function on array_like X
    """
    X = np.array(X)
    return softmax_F(X) * ( np.identity(X.shape[0]) - softmax_F(X) ) 

class MNIST_dataset():

    def __init__(self, csv_path):
        """
        MNIST_dataset - (csv_path)
            - Loads from comma separated values file
        """
        df = pd.read_csv(csv_path)
        data = df.to_numpy()

        self.classes = data[:,0]
        self.images = data[:,1:] / 255.0

    def __len__(self):
        return len(self.images)

    def out_feat(self):
        return len(np.unique(self.classes))

    def in_feat(self):
        return 784

    def __getitem__(self,idx):
        return {'image': self.images[idx],'class': self.classes[idx]}

# Class to represent a nueral network 
class NueralNetwork():

    def __init__(self,sizes):
        self.sizes = sizes

        self.params = []

        for i in range(len(sizes)-1):
            weights = np.random.rand( sizes[i+1], sizes[i] )
            self.params.append(weights)
            
    def __str__(self):
        return str(self.params)

    def forward(self, x):
        params = self.params

        activation = x

        for layer in params:
            inputs = np.dot(layer, activation )
            activation = sigmoid_F(inputs)

        return activation

    def backward(self, y, activation):
        #for each set of weights
        delta = []

        return y # Need to watch lecture
        
    def update(self, something):
        return something

    def predict(self, x):
        return self.forward(x)

    def train(self, epochs, dataset):

        for epoch in range(epochs):
            
            # could to random indices batches, OOOOOO NORMAL or gaussian here?
            for index in range(len(dataset)):

                output = self.forward(dataset[index]['image'])

                delta = self.backward(dataset[index]['class'], output)

                self.update(delta)