#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage import io


if __name__ == "__main__":
    print 'Ejecutando como programa principal'


    X = np.load('x.npy')
    y = np.load('y.npy')


    X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.25, random_state=0)

    ###RED NEURONAL###
    acierto_espiral_nn = X_test[0].reshape(64,64);
    fig1 = plt.figure()
    fig1.suptitle("Acierto Espiral", fontsize=10)
    io.imshow(acierto_espiral_nn)

    acierto_red_nn = X_test[6].reshape(64,64);
    fig2 = plt.figure()
    fig2.suptitle("Acierto Redondeada", fontsize=10)
    io.imshow(acierto_red_nn)

    fallo_red_espiral_nn = X_test[11].reshape(64,64);
    fig3 = plt.figure()
    fig3.suptitle("Fallo es Redondeada predice Espiral", fontsize=10)
    io.imshow(fallo_red_espiral_nn)

    fallo_espiral_red_nn = X_test[18].reshape(64,64);
    fig4 = plt.figure()
    fig4.suptitle("Fallo es espiral predice redondeada", fontsize=10)
    io.imshow(fallo_espiral_red_nn)


    ###NAIVE BAYES###
    acierto_espiral_nb = X_test[7].reshape(64,64);
    fig1 = plt.figure()
    fig1.suptitle("Acierto Espiral", fontsize=10)
    io.imshow(acierto_espiral_nb)

    acierto_red_nb = X_test[5].reshape(64,64);
    fig2 = plt.figure()
    fig2.suptitle("Acierto Redondeada", fontsize=10)
    io.imshow(acierto_red_nb)

    fallo_red_espiral_nb = X_test[8].reshape(64,64);
    fig3 = plt.figure()
    fig3.suptitle("Fallo es Redondeada predice Espiral", fontsize=10)
    io.imshow(fallo_red_espiral_nb)

    fallo_espiral_red_nb = X_test[9].reshape(64,64);
    fig4 = plt.figure()
    fig4.suptitle("Fallo es espiral predice redondeada", fontsize=10)
    io.imshow(fallo_espiral_red_nb)
