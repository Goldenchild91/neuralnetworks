import pandas as pd
import math
import random
import numpy as np

#models a simple ANN to solve the Titanic problem
#Ella Mohanram
#May 13, 2023

#models a simple ANN to solve the Titanic problem
class NeuralNetwork:
    #number of features in the input file
    num_feats_read = 13

    #number of features in the model
    num_feats = 9

    #multiple of inputs to use for number of hidden nodes
    hidden_pct = 4.0

    #number of output nodes
    num_outputs = 1

    #number of training epochs
    num_epochs = 1000

    #rate at which weights are adjusted
    learn_rate = 1.0

    #inputs nodes
    inputs = []

    #weights of input nodes to hidden nodes
    weights_input = []

    #hidden nodes
    hidden = [0] * int(num_feats * hidden_pct)

    #weights of hidden nodes to output nodes
    weights_hidden = []

    #output nodes
    outputs = []

    #runs sigmoid function for forward propagation
    #@param x the input value
    #@return sigmoid(x)
    def sigmoid(self, x):
        return (1 / (1 + math.exp(x * -1)))

    #runs sigmoid derivative function for backward propagation
    #@param x the input value
    #@return sigmoid derivative of x
    def sigmoid_derivative(self, x):
        return (x * (1 - x))

    #converts passenger data to a string
    #param to_string: all properties of passenger
    #@return string containing all properties
    def to_string(self, to_string):
        string = '#%s: ' % str(to_string[0])
        string += '%s (%s/%s)' % (str(to_string[3]),str(to_string[4]),str(to_string[5]))
        string += '(%sC, T = %s F = %s C = %s E = %s)' % (str(to_string[2]), str(to_string[8]),str(to_string[9]),str(to_string[10]),str(to_string[11]))
        string += ' (S = %s, P = %s) ' % (str(to_string[6]),str(to_string[7]))

        return string

    #reads data from input file
    #@param filename: the file to read
    #@return features, labels: a list of scaled features and labels
    def read_passengers(self, filename):
        file = pd.read_csv(filename)

        file['Sex_m'] = [1 if sex == 'male' else 0 for sex in file['Sex']]
        file['Sex_f'] = [1 if sex == 'female' else 0 for sex in file['Sex']]
        file['Emb_s'] = [1 if loc == 'S' else 0 for loc in file['Embarked']]
        file['Emb_c'] = [1 if loc == 'C' else 0 for loc in file['Embarked']]
        file['Emb_q'] = [1 if loc == 'Q' else 0 for loc in file['Embarked']]

        labels = file.Survived.values.tolist()

        my_maxes = file[['Age', 'Parch', 'SibSp']].max().tolist()
        mean_age = file['Age'].mean()

        file['Age'] = file['Age'].fillna(mean_age)

        to_string_features = file.copy().values.tolist()

        file['Age'] = [(age/my_maxes[0]) for age in file['Age']]
        file['Pclass'] =[(1/pclass) for pclass in file['Pclass']]
        file['Parch'] = [(parch/my_maxes[1]) for parch in file['Parch']]
        file['SibSp'] = [(sibsp/my_maxes[2]) for sibsp in file['SibSp']]

        file.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Sex', 'Embarked', 'Survived'], inplace=True,
                        axis=1)

        features = file.values.tolist()

        return features, labels, to_string_features

    #sets up network structure
    def setup_network(self):
        for i in range(int(NeuralNetwork.num_feats)):
            input_array = [random.random() for p in range(int(NeuralNetwork.num_feats * NeuralNetwork.hidden_pct))]
            NeuralNetwork.weights_input.append(input_array)

        for i in range(int(NeuralNetwork.num_feats * NeuralNetwork.hidden_pct)):
            input_array = [random.random() for p in range(NeuralNetwork.num_outputs)]
            NeuralNetwork.weights_hidden.append(input_array)

    #runs forward propagation algorithm for a given element
    #returns: the vector of output values
    def forward_propagation(self, feature_set):
        hidden_weighted_sum = np.matmul(feature_set, NeuralNetwork.weights_input).tolist()
        for i in range(len(NeuralNetwork.hidden)):
            NeuralNetwork.hidden[i] = NeuralNetwork.sigmoid(self, hidden_weighted_sum[i])
        output_weighted_sum = np.matmul(NeuralNetwork.hidden, NeuralNetwork.weights_hidden).tolist()
        NeuralNetwork.outputs = [NeuralNetwork.sigmoid(self, output_weighted_sum[0])]

    #runs back propagation algorithm to update weights
    #param label_set: the labels to use for cost calculation
    def back_propagation(self, label_set):
        output_error = NeuralNetwork.sigmoid_derivative(self, NeuralNetwork.outputs[0]) * (label_set - NeuralNetwork.outputs[0])

        hidden_errors = []
        for hidldx in range(len(NeuralNetwork.hidden)):
            arr = NeuralNetwork.weights_hidden[hidldx][0] * output_error * NeuralNetwork.sigmoid_derivative(self, NeuralNetwork.hidden[hidldx])
            hidden_errors.append(arr)

        for hidldx1 in range(len(NeuralNetwork.hidden)):
            NeuralNetwork.weights_hidden[hidldx1][0] += (NeuralNetwork.learn_rate * NeuralNetwork.hidden[hidldx1] * output_error)

        for hidldx2 in range(len(NeuralNetwork.hidden)):
            for inldx in range(len(NeuralNetwork.inputs)):
                NeuralNetwork.weights_input[inldx][hidldx2] += (NeuralNetwork.learn_rate * NeuralNetwork.inputs[inldx] * hidden_errors[hidldx2])

    #runs the training algorithm on a network
    #param filename: the input file of training data
    def train_neural_network(self, filename):
        NeuralNetwork.setup_network(self)
        features, labels, to_string = NeuralNetwork.read_passengers(self, filename)
        for i in range(1, NeuralNetwork.num_epochs):
            #count = 0
            for idx in range(len(features)):
                NeuralNetwork.forward_propagation(self, features[idx])
                NeuralNetwork.back_propagation(self, labels[idx])
                #if round(NeuralNetwork.outputs[0]) == labels[idx]:
                    #count += 1
            #if (i != 0) and (i % 100 == 0):
                #print(count/(len(features)))

    #runs the test algorithm on a network
    #param filename: the input file of testing data
    def test_neural_network(self, filename):
        features, labels, to_string_features = NeuralNetwork.read_passengers(self, filename)
        count = 0
        for idx in range(len(features)):
            NeuralNetwork.forward_propagation(self, features[idx])
            NeuralNetwork.back_propagation(self, labels[idx])
            output_string = NeuralNetwork.to_string(self, to_string_features[idx])

            if labels[idx] == 1:
                actual = 'Survived'
            else:
                actual = 'Did Not Survive'
            if round(NeuralNetwork.outputs[0]) == 1:
                expected = 'Survived'
            else:
                expected = 'Did Not Survive'
            output_string += 'Actual: ' + actual + ', Expected: ' + expected

            if round(NeuralNetwork.outputs[0]) == labels[idx]:
                count += 1
                output_string += ' MATCH'
            print(output_string)
        print(count / (len(features)))

network = NeuralNetwork()
network.train_neural_network("/Users/student/Desktop/NeuralNetwork/titanic/train.csv")
network.test_neural_network("/Users/student/Desktop/NeuralNetwork/titanic/test.csv")