import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x, derivative=False):
        if derivative:
            return sigmoid(x) * (1 - sigmoid(x))
        return 1/(1 + np.exp(-x))

def softmax(x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)

def ReLU(X, derivative=False):
    if derivative:
        return np.greater(X, 0).astype(int)
    return np.maximum(0,X)

class MNIST_dataset():

    def __init__(self, csv_path):
        """
        MNIST_dataset - (csv_path)
            - Loads from comma separated values file
        """
        df = pd.read_csv(csv_path)
        data = df.to_numpy()

        a = data[:,0]

        one_hot = np.zeros((a.size, a.max()+1))
        one_hot[np.arange(a.size),a] = 1
        self.classes = one_hot
        #Normalize
        self.images = data[:,1:]
        self.images = self.images.astype('float64')/255
    def __len__(self):
        return len(self.images)

    def out_feat(self):
        return len(np.unique(self.classes))

    def in_feat(self):
        return 784

    def __getitem__(self,idx):
        return {'image': self.images[idx],'class': self.classes[idx]}

    def plot(self, idx):
        f = plt.figure()
        sp = f.add_subplot(111)

        #gather data from row
        data = self.__getitem__(idx)
        label = data['class']
        pixels = data['image']

        #interpret one dimensional array as image
        pixels = pixels.reshape((28,28))

        #set label, show image
        sp.set_title(f'Label is {label}')
        sp.imshow(pixels)#cmap ='gray')

        return f


# Class to represent a nueral network 
class NueralNetwork():

    def __init__(self,sizes):
        np.random.seed(22)
        self.sizes = sizes
        self.biases = []

        self.params = []

        for i in range(len(sizes)-1):
            weights = np.random.rand( sizes[i+1], sizes[i] )
            bias = np.random.rand(sizes[i+1])
            self.params.append(weights)
            self.biases.append(bias)

        print(self.params)
        print(self.biases)
            
    def __str__(self):
        return str(self.params)

    def forward(self, x):
        '''
        Forward Pass of Nueral network, returns activation or "output"
        '''
        params = self.params
        self.inputs = []
        self.activations = []

        activation = x
        self.activations.append(x)

        for index, layer in enumerate(params[:-1]):
            _input = np.dot( layer, activation) + self.biases[index]
            self.inputs.append( _input )
            activation = sigmoid(_input)
            self.activations.append(activation)

        final_input = np.dot( params[-1] , activation) + self.biases[-1]
        self.inputs.append(final_input)
        activation = softmax( final_input, derivative=False )
        self.activations.append(activation)

        return activation

    def backward(self, y, output):
        '''
        Backward Pass of Nueral network, returns change to params
        '''
        #for each set of weights, calculate error BACKWARDS
        deltas = []

        #print(y, output)

        error = (output - y) 

        #print(error)

        #print(softmax(self.inputs[-1], derivative=True))

        error = error * softmax(self.inputs[-1], derivative=True)

        #print(error)

        deltas.append( np.outer( error , self.activations[-2]))

        #print('final delta')
        #print(np.outer( error , self.activations[-2]).shape)

        #print('Activations')
        #for act in self.activations:
         #   print(act.shape)


        for index in reversed(range(len(self.params[1:]))):
            #print(index)
            #print(np.dot(self.params[index + 1].T, error))
            error = np.dot(self.params[index + 1].T, error) * ReLU( self.inputs[ index ], derivative=True)
            #print("is this small")
            #print(np.outer( error , self.activations[index]).shape)
            deltas.insert(0, np.outer( error , self.activations[index]))

        #print('deltas')
        #for de in deltas:
        #    print(de.shape)

        #print(error)
           
        return deltas 
        
    def update(self, deltas, lr):
        '''
        Update weights of nueral network, no return 
        '''
        for index,layer in enumerate(self.params):
            self.params[index] = layer - lr * deltas[index]


    def predict(self, x):
        '''
        Make a prediction 
        '''
        return  np.argmax( self.forward(x) )

    def train(self, epochs, lr, dataset):

        for epoch in range(epochs):

            #print(self.params[0])
            
            # could to random indices batches, OOOOOO NORMAL or gaussian here?
            for index in range(len(dataset)):

                output = self.forward(dataset[index]['image'])

                delta = self.backward(dataset[index]['class'], output)

                self.update(delta, lr)

                #print(delta)

            acc = self.accuracy(dataset)

            print(acc)

            #print(self.params[1])

            #break

    def accuracy(self, dataset):

        acc = []

        for index in range(len(dataset)):
            #gather data from row
            label = dataset[index]['class']
            pixels = dataset[index]['image']

            pred = self.predict(pixels)
            acc.append( np.argmax(label) == pred )

        return np.mean(acc)