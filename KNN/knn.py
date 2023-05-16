import pandas as pd 
import numpy as np
class KNN:
    def __init__(self, k = 2):
        self.X = None
        self.y = None
        self.k = k

    def _calculate_distances(self, point):
        distances = np.linalg.norm(self.X - point, axis=1)
        return distances

    def train(self, X, y):
        self.X = X
        self.y = y       

    def predict(self, sample):
        distances = self._calculate_distances(sample)

        # distances.argsort() - soruje rosnąco, zamienia wartość na jej index w liscie
        nearest_neighbor_indxs = distances.argsort()[:self.k] #zwraca liste k id sąsiadów

        class_neightbors = pd.DataFrame(self.y[nearest_neighbor_indxs]) # zwraca liste indexów klas k sąsiadów

        flower_class = class_neightbors.mode().values # znajduje najczęściej występującą klasę
        return flower_class[0][0]

    def confusion_matrix(self, predicted, target):
        m_indx = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
        }
        n = len(m_indx)
        M = np.zeros((n,n), dtype=int)
        # dla każdej próbki zmień klasę przewidywaną i rzeczywistą 
        # na odpowieni indeks macierzy(komórkę) i zwiększ wartość w tej komórce o 1
        for i in range(len(target)):
            M[m_indx[target[i]]][m_indx[predicted[i]]] += 1
        
        # ładne opisy dla dataframe
        pred_labels = []
        target_labels = []
        for i in range(M.shape[0]):
            pred_labels.append(f"pred_{i}")
            target_labels.append(f"target_{i}")
        M = pd.DataFrame(data = M, columns = pred_labels, index = target_labels)

        return M

    def calculate_accuracy(self, M):
        # wyciągnięcie wartości bez opisów
        M = M.values
        correct = np.sum(np.diag(M))
        all = np.sum(M)
        return correct / all