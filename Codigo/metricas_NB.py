#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, \
                            f1_score, log_loss, precision_score, \
                            recall_score, roc_auc_score, confusion_matrix

from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib

class Metrics:

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def plot_roc_curve(self, classifier, X_test, y_test):
        n_classes = 2
        y_score = classifier.predict_proba(X_test)[:, 1];
        print y_score
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test, y_score, pos_label=2)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test, y_score, pos_label=2)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 2
        plt.plot(fpr[1], tpr[1], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

if __name__ == "__main__":
    print 'Ejecutando como programa principal'

    metrics = Metrics()

    X = np.load('x.npy')
    y = np.load('y.npy')


    X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.25, random_state=0)

    naive_bayes = joblib.load('resultados/naiveBayes.pkl')


    ###ACCURACY_SCORE###
    y_pred_nn = naive_bayes.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_nn)
    print "accuracy_score: " + str(accuracy)


    ###F1_SCORE###
    f1 = f1_score(y_test, y_pred_nn)
    print "f1: " + str(f1);


    ###PRECISION_SCORE###
    precision = precision_score(y_test, y_pred_nn)
    print "precision_score: " + str(precision)


    ###RECALL_SCORE###
    recall = recall_score(y_test, y_pred_nn)
    print "recall_score: " + str(recall)


    ###CONFUSION_MATRIX###
    cnf_matrix = confusion_matrix(y_test, y_pred_nn)
    print "confusion_matrix: " + str(cnf_matrix)
    plt.figure()
    class_names = ['rounded', 'spiral']
    class_names = np.array(class_names)
    metrics.plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')


#    ###ROC_AUC_SCORE###
#    y_pred_prob_nn = mlp.predict_proba(X_test)
#    y_pred_prob_positive_nn = y_pred_prob_nn[:,1]
#    roc_auc = roc_auc_score(y_test - 1, y_pred_prob_positive_nn - 1)
#    print "roc_auc_score: " + str(roc_auc)
#
#
#    ###ROC_CURVE###
#    metrics.plot_roc_curve(mlp, X_test, y_test);
