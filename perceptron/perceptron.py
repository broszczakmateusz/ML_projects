import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self, weights = None, bias = 0, learning_rate = 0.2):
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate

    def train(self, X, y):
        # liczba próbek, liczba cech pojedynczej próbki
        n_samples, n_features = X.shape[0], X.shape[1]

        # wpisanie do 0 kolumny: wektora wag - biasu, macierzy próbek - wartości 1
        self.weights = np.hstack((self.bias, np.zeros(n_features)))
        X = np.c_[np.ones(n_samples), X]

        i_max = 1000
        for i in range(n_samples):
            for j in range(i_max):
                # obliczenie sumy ważonej wejść
                weighted_sum = np.dot(X[i], self.weights) 
                
                # przypisanie klasy na podstawie funkcji aktywacji
                y_pred = np.sign(weighted_sum)
                
                # jeżeli predykcja jest poprawna przejdź do kolejnej próbki
                if y_pred == y[i]:
                    break                

                delta_weights = self.learning_rate * (y[i] - y_pred) * X[i]

                # jeżeli wartości wag i biasu nie będą się bardzo zmieniać
                if max(abs(delta_weights)) < 0.1:
                    break

                # aktualizacja wag i biasu
                if y_pred != y[i]:        
                    self.weights += delta_weights
                    self.bias = self.weights[0]


    def predict(self, sample):
        sample = np.hstack((1, sample))
        weighted_sum = np.dot(sample, self.weights)
        y_pred = np.sign(weighted_sum)
        return y_pred