#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import csv
import numpy as np
from skimage import color
from skimage import io
from skimage import transform
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

class Galaxy:

    X = [];
    y = [];

    def getAllGalaxies(self):
        galaxy_types = {}
        with open('training_solutions_rev1.csv', 'rb') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            data = list(data)

        data = np.array(data)
        data = data[1:,0:4]
        for row in data:
           galaxy_types[row[0]] = row.tolist().index(max(row[1:4]))

        galaxy_types = { k:v for k, v in galaxy_types.items() if v !=3 }
        return galaxy_types

    def fillGalaxiesArray(self, N=0):
        X_aux = [];
        i = 1;
        galaxies = Galaxy.getAllGalaxies(self);
        for key, value in galaxies.items():
            X_aux.append(key)
            self.y.append(value)
        if N != 0:
            X_aux = X_aux[0:N]
            self.y = self.y[0:N]

        for galaxy in X_aux:
            i += 1
            print i,"of",str(len(X_aux))
            print "Working in galaxies"
            print "--------------------"
            im = io.imread('images_training_rev1/' + galaxy + '.jpg');
            im = color.rgb2grey(im)
            im = transform.resize(im, (64, 64), mode='reflect')
            im_vector = np.array(im).ravel();
            self.X.append(im_vector);
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        return galaxies

    def neuralNetwork(self):
        X_train, X_test, y_train, y_test = \
            train_test_split(self.X, self.y, test_size=0.25, random_state=0)

        #downsampling training
        X_train = X_train[0:-1:100,:]
        y_train = y_train[0:-1:100]
        learning_rate = np.logspace(-2, 3, 70)
        alpha = np.logspace(-4, 4, 70)

        tuned_parameters = {'hidden_layer_sizes': [[2],[3],[3,3,3],
                                                   [4,5,3],
                                                   [10,10]],
                            'learning_rate_init': learning_rate,
                            'alpha': alpha,
                            'activation': ['logistic', 'relu']}

        mlp = MLPClassifier(solver='sgd');
        grid = GridSearchCV(mlp, param_grid=tuned_parameters, cv=2, n_jobs = -1,verbose = 4)
        grid.fit(X_train, y_train)
        joblib.dump(grid, 'neuralNetwork.pkl')


    def naiveBayes(self):
        X = self.X
        y = self.y
        print X.shape
        print y.shape
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.25, random_state=0)
        #Downsampling training set

        naive_bayes = GaussianNB()
        naive_bayes.fit(X_train,y_train)
        joblib.dump(naive_bayes, 'naiveBayes.pkl')

if __name__ == "__main__":
    print 'Ejecutando como programa principal'

    galaxies = Galaxy()
    f = raw_input('Load saved images [yes]/no: ')

    if f == 'yes' or f== '':
        print "Loading data ..."
        X = np.load('x.npy')
        y = np.load('y.npy')
        print "Data loaded"
        galaxies.X = X
        galaxies.y = y
    elif f == 'no':
        z = galaxies.fillGalaxiesArray();
        X = galaxies.X;
        y = galaxies.y;
        print "saving numpy images as x.npy and y.npy"
        np.save('x.npy',X)
        np.save('y.npy',y)


    #galaxies.y = y[0:119]
    print " "
    print "-----------------"
    print "Training Navie Bayes"
    galaxies.naiveBayes();
    print " "
    print "-----------------"
    print "Training Neural Network"
    galaxies.neuralNetwork();
