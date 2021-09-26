import numpy as np
import torch
import random
import math
from scipy.spatial import distance
from torch import nn, optim
from torchvision import datasets, transforms
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB


#N is the number of folds
#dictionaries are used to divide the data in N folds in a righteous way
N = 10
dictionary_1 = {}
dictionary_2 = {}

def crossValidation(pos):
    pos = np.mod(pos, N)
    return(pos)

def sortIt(filename):
    credit_infile2 = open(filename)
    credit_infile3 = open(filename)
    sort_line = credit_infile3.readline()
    sort_line = credit_infile3.readline()
    sort_line = credit_infile3.readline()
    lines_credit = credit_infile2.readlines()
    lista  = []

    for i in range(0, len(lines_credit)):
        if sort_line != "":
            lista2 = []
            length_line = sort_line.split(",")
            for j in range (0, len(length_line)):
                    lista2.append(float(length_line[j]))
            lista.append(lista2)
            sort_line = credit_infile3.readline()
    output = sorted(lista, key=lambda x: x[-1])
    return output
            

def organiseFolds(fileName):
    
    if(fileName == "spambase.data"):
        infile = open("spambase.data")
        infile2 = open("spambase.data")
        lines = infile.readlines()
        line = infile2.readline()
        length = line.split(",")
        lista = []
        
        for i in range(0, len(lines)):
            if line != "":
                lista = []
                pos = i
                pos = crossValidation(pos)
                length = line.split(",")

                for j in range (0, len(length)):
                    lista.append(float(length[j]))

                if pos in dictionary_1:
                    values = dictionary_1.get(pos)
                    values.append(lista)
                    dictionary_1[pos] = values
                else:
                    dictionary_1[pos] = [lista]
                line = infile2.readline()

            
    elif (fileName == "default of credit card clients.csv"):
        
        credit_infile = open("default of credit card clients.csv")
        credit_infile2 = open("default of credit card clients.csv")
        
        lines_credit = credit_infile.readlines()
        line_credit = credit_infile2.readline()
        line_credit = credit_infile2.readline()
        line_credit = credit_infile2.readline()
        length_credit = line_credit.split(",")
        lista  = []
        sorted_list = sortIt("default of credit card clients.csv")

        for i in range(0, len(sorted_list)):
            if line_credit != "":
                lista = []
                pos = i
                pos = crossValidation(pos)
                length_credit = line_credit.split(",")
                del sorted_list[i][0]
                del sorted_list[i][1]
                del sorted_list[i][1]
                del sorted_list[i][1]
                del sorted_list[i][1]
                del sorted_list[i][1]
                del sorted_list[i][1]
                del sorted_list[i][1]
                del sorted_list[i][1]
                del sorted_list[i][1]
                del sorted_list[i][1]

                for j in range (0, len(sorted_list[i])):
                    lista.append(float(sorted_list[i][j]))
            
                if pos in dictionary_2:
                    values = dictionary_2.get(pos)
                    values.append(lista)
                    dictionary_2[pos] = values
                else:
                    dictionary_2[pos] = [lista]
                line_credit = credit_infile2.readline()


organiseFolds("spambase.data")
organiseFolds("default of credit card clients.csv")

def normalize(credit, array):
    temp = np.amax(credit, axis =0)
    temp = temp.tolist()
    for i in range(0, len(array)):
        for j in range(0, len(array[0])-1):
            array[i][j] = np.around(array[i][j] / temp[j], decimals =5)
    return array



#spambase folds
fold0 = dictionary_1.get(0)
fold1 = dictionary_1.get(1)
fold2 = dictionary_1.get(2)
fold3 = dictionary_1.get(3)
fold4 = dictionary_1.get(4)
fold5 = dictionary_1.get(5)
fold6 = dictionary_1.get(6)
fold7 = dictionary_1.get(7)
fold8 = dictionary_1.get(8)
fold9 = dictionary_1.get(9)

#credit cards folds
credit_fold0 = dictionary_2.get(0)
credit_fold1 = dictionary_2.get(1)
credit_fold2 = dictionary_2.get(2)
credit_fold3 = dictionary_2.get(3)
credit_fold4 = dictionary_2.get(4)
credit_fold5 = dictionary_2.get(5)
credit_fold6 = dictionary_2.get(6)
credit_fold7 = dictionary_2.get(7)
credit_fold8 = dictionary_2.get(8)
credit_fold9 = dictionary_2.get(9)
'''
credit = credit_fold0 + credit_fold1 + credit_fold2 + credit_fold3 + credit_fold4 + credit_fold5 + credit_fold6 + credit_fold7 + credit_fold8 + credit_fold9
credit_fold0 = normalize(credit, credit_fold0)
credit_fold1 = normalize(credit, credit_fold1)
credit_fold2 = normalize(credit, credit_fold2)
credit_fold3 = normalize(credit, credit_fold3)
credit_fold4 = normalize(credit, credit_fold4)
credit_fold5 = normalize(credit, credit_fold5)
credit_fold6 = normalize(credit, credit_fold6)
credit_fold7 = normalize(credit, credit_fold7)
credit_fold8 = normalize(credit, credit_fold8)
credit_fold9 = normalize(credit, credit_fold9)
'''
def LVQ(fileName):
    training_set = []
    test_set = []
    centers = []
    radius = []
    category0 = []
    category1 = []
    learning_rate = 0.01
    negative = 0
    positive = 0
    true_negative = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0
    accuracy = 0
    precision = 0
    recall = 0
    F1_score = 0
    
    if (fileName == "spambase.data"): 
        training_set = fold0 #+ fold1 + fold2 + fold3 + fold4 + fold5 + fold6 + fold7 + fold8
        test_set = fold9
    elif (fileName == "default of credit card clients.csv"):
        training_set = credit_fold0 #+ credit_fold1 + credit_fold2 + credit_fold3 + credit_fold4 + credit_fold5 + credit_fold6 + credit_fold7 + credit_fold8
        test_set = credit_fold9
    
    
    for i in range(0, len(training_set)):
        if training_set[i][len(training_set[i])-1] == 0:
            category0.append(training_set[i])
        elif training_set[i][len(training_set[i])-1] == 1:
            category1.append(training_set[i])

    
    
    temp = []
    temp = np.mean(category0, axis = 0)
    temp = temp.tolist()
    centers.append(temp)

    temp = np.mean(category1, axis = 0)
    temp = temp.tolist()
    centers.append(temp)
    
    for i in range(0, len(centers)):
        for j in range(0, len(centers[i])): 
            centers[i][j] = np.around(centers[i][j], decimals =5)
    
    
    var0 = np.var(category0) 
    var0 = np.around(var0, decimals =5)
    radius.append(var0)
    var1 = np.var(category1)
    var1 = np.around(var1, decimals = 5)
    radius.append(var1)
    

    distances = []
    K = 2
    M = 10
    N = len(training_set)
    for epoch in range(0, M):
        print("LVQ Training- Epoch", epoch+1)
        for i in range(0, N):
            distances = np.zeros(K)
            random_example = random.randint(0, len(training_set)-1)
            right_category = training_set[random_example][len(training_set[random_example])-1]
            
            
            for j in range(0, len(training_set[i])-1):
                for k in range(0, K):
                     distances[k] =  (distances[k] + math.exp(-pow(abs(training_set[random_example][j] - centers[k][j]), 2) / (2 * radius[k])))#''', decimals = 15)'''np.around( ''''''
            winner_radius = np.amax(distances)
            
            for l in range (0, len(distances)):
                if distances[l] == winner_radius:
                    winner_position = l
                    
            winner_category = centers[winner_position][len(centers[winner_position])-1]
            
            if winner_category == right_category: #Reward
                for j in range(0, len(centers[winner_position])-1):
                    centers[winner_position][j] = (1 - learning_rate) * centers[winner_position][j] + learning_rate * training_set[random_example][j]
                    radius[winner_position] = radius[winner_position] + learning_rate * distance.euclidean(training_set[random_example][j], centers[winner_position][j])
                
            else: #Penalty
                for j in range(0, len(centers[winner_position])-1):
                    centers[winner_position][j] =(1 + learning_rate) * centers[winner_position][j] - learning_rate * training_set[random_example][j]
                    radius[winner_position] = radius[winner_position] - learning_rate * distance.euclidean(training_set[random_example][j], centers[winner_position][j])
              
                K = K + 1
                centers.append(training_set[random_example])
                radius_init = radius[winner_position] / 5
                radius.append(radius_init)
               
                print("New Area, The number of areas are:", K)
                        
    print("LVQ Testing Starting")
    for i in range(0, len(test_set)):
        distances = np.zeros(K)
        for j in range(0, len(test_set[i])):
            for k in range(0, K):
                distances[k] = np.around(distances[k] + math.exp(-pow(abs(test_set[i][j] - centers[k][j]), 2) / (2 * radius[k])), decimals = 15)
            
        winner_radius = np.amax(distances)
            
        for l in range (0, len(distances)):
            if distances[l] == winner_radius:
                winner_position = l
                    
        winner_category = centers[winner_position][len(centers[winner_position])-1]

        if test_set[i][len(test_set[i])-1] == 0:
            negative = negative + 1
        elif test_set[i][len(test_set[i])-1] == 1:
            positive = positive + 1
            
            
        if winner_category == test_set[i][len(test_set[i])-1] and winner_category == 0:
            true_negative = true_negative + 1
            print("TrueNegative", true_negative)
        elif winner_category == test_set[i][len(test_set[i])-1] and winner_category == 1:
            true_positive = true_positive + 1
            print("TruePositive", true_positive)
        elif winner_category != test_set[i][len(test_set[i])-1] and winner_category == 0:
            false_negative = false_negative + 1
            print("FalseNegative", false_negative)
        elif winner_category != test_set[i][len(test_set[i])-1] and winner_category == 1:
            false_positive = false_positive + 1
            print("FalsePositive", false_positive)
            
    accuracy = ((true_positive + true_negative) / (positive + negative)) * 100
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    F1_score = 2 * (precision * recall) / (precision + recall)
    print("Accuracy is ", accuracy)
    print("F1_score is ", F1_score)
        
            
            
#LVQ("spambase.data")
#LVQ("default of credit card clients.csv")
