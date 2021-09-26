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

def SVM(fileName, kernel):
    training_set = []
    test_set = []
    right_categories = []
    test_right_categories = []
    negative = 0
    positive = 0
    if fileName == "spambase.data":
        training_set = fold0 + fold1 + fold2 + fold3 + fold4 + fold5 + fold6 + fold7 + fold8
        test_set = fold9
    elif (fileName == "default of credit card clients.csv"):
        training_set = credit_fold0 + credit_fold1 + credit_fold2 + credit_fold3 + credit_fold4 + credit_fold5 + credit_fold6 + credit_fold7 + credit_fold8
        test_set = credit_fold9
    
    #Converting Data
    for i in range(0, len(training_set)):
        right_categories.append(training_set[i][len(training_set[i])-1])
    for i in range(0, len(test_set)):
        test_right_categories.append(test_set[i][len(test_set[i])-1])
        
    
    trainval = []
    testval = []
    for i in range(0, len(training_set)):
        temp = []
        for j in range(0, len(training_set[i])-1):
            temp.append(training_set[i][j])
        trainval.append(temp)
        
    for i in range(0, len(test_set)):
        temp = []
        for j in range(0, len(test_set[i])-1):
            temp.append(test_set[i][j])
        testval.append(temp)
    
    #Training
    if kernel == "Linear":
        svc = SVC(kernel = 'linear')
        svc.fit(trainval, right_categories)
        output = svc.predict(testval)
        print(confusion_matrix(test_right_categories, output))
        
        #Testing
        matrix = confusion_matrix(test_right_categories, output)
        for i in range(0, len(test_right_categories)):
            if test_right_categories[i] == 0:
                negative = negative + 1
            elif test_right_categories[i] == 1:
                positive = positive + 1
        true_positive = matrix[0][0]
        false_positive = matrix[0][1]
        false_negative = matrix[1][0]
        true_negative = matrix[1][1]
        accuracy = ((true_positive + true_negative) / (positive + negative)) * 100
        print("Accuracy is %.2lf" %accuracy)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        F1_score = 2 * (precision * recall) / (precision + recall)
        print("F1_score is %.2lf" %F1_score)
        
        
        print(classification_report(test_right_categories, output))
    
    elif kernel == "Gaussian":
        svc = SVC(kernel = 'rbf')
        svc.fit(trainval, right_categories)
        output = svc.predict(testval)
        print(confusion_matrix(test_right_categories, output))
        
        matrix = confusion_matrix(test_right_categories, output)
        for i in range(0, len(test_right_categories)):
            if test_right_categories[i] == 0:
                negative = negative + 1
            elif test_right_categories[i] == 1:
                positive = positive + 1
        true_positive = matrix[0][0]
        false_positive = matrix[0][1]
        false_negative = matrix[1][0]
        true_negative = matrix[1][1]
        accuracy = ((true_positive + true_negative) / (positive + negative)) * 100
        print("Accuracy is %.2lf" %accuracy)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        F1_score = 2 * (precision * recall) / (precision + recall)
        print("F1_score is %.2lf" %F1_score)
        
        
        print(classification_report(test_right_categories, output))
    
        
#SVM("spambase.data", "Linear")
#SVM("spambase.data", "Gaussian")
#SVM("default of credit card clients.csv", "Linear")
#SVM("default of credit card clients.csv", "Gaussian")
