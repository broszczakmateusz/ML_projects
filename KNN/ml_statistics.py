import pandas as pd
import numpy as np

def confusion_matrix(target, predicted, n, index_dict):
        M = np.zeros((n,n), dtype=int)
        
        # dla każdej próbki zmień klasę przewidywaną i rzeczywistą 
        # na odpowieni indeks macierzy(komórkę) i zwiększ wartość w tej komórce o 1
        for i in range(len(target)):
            M[index_dict[target[i]]][index_dict[predicted[i]]] += 1
        
        # ładne opisy
        pred_labels = []
        target_labels = []
        for i in range(M.shape[0]):
            pred_labels.append(f"pred_{i}")
            target_labels.append(f"target_{i}")
        M = pd.DataFrame(data = M, columns = pred_labels, index = target_labels)

        return M

def calculate_accuracy(target, predicted):
    correct = 0
    all = 0
    for a, p in zip(target, predicted):
        if a == p:
            correct += 1
        all += 1
    
    if all == 0:
        return "Error - devided by 0"
    return correct/all