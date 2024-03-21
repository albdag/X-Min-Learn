# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 21:54:40 2022

@author: albdag
"""

from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.metrics import accuracy_score, silhouette_score, silhouette_samples


class MultiTaskThread(QThread):
    taskInitialized = pyqtSignal(str)
    workFinished = pyqtSignal(tuple, bool)

    def __init__(self):
        super(MultiTaskThread, self).__init__()

    def run(self):
        # To be reimplemented in each child
        return


class MineralClassificationThread(MultiTaskThread):

    def __init__(self):
        super(MineralClassificationThread, self).__init__()

        self.classifier = None
        self.algorithm = None

    def set_classifier(self, classifier):
        self.classifier = classifier
        self.algorithm = classifier.algorithm

    def reset_classifier(self):
        self.classifier = None
        self.algorithm = None

    def run(self):
        if self.classifier is None or self.algorithm is None: return
        # Reimplement in each child



class SilhouetteThread(QThread):
    subtaskCompleted = pyqtSignal() # to increment progressBar by 1 unit
    taskFinished = pyqtSignal(tuple) # (True/False, outputs/exception)

    def __init__(self):
        super(SilhouetteThread, self).__init__()

        self.data = None
        self.pred = None

    def set_params(self, data, pred):
        self.data = data
        self.pred = pred

    def silhouette_metric(data, pred, type):
        if type == 'avg':
            return silhouette_score(data, pred, metric='euclidean')
        elif type == 'all':
            return silhouette_samples(data, pred, metric='euclidean')
        else:
            raise NameError(f'{type} is not a valid silhouette score type.')

    def run(self):
        try:
        # Compute the overall average silhouette score
            mask = self.pred != '_ND_' # exclude ND data for the average prediction
            sil_avg = self.silhouette_metric(self.data[mask, :], self.pred[mask], type='avg')
            self.subtaskCompleted.emit()

        # Compute the silhouette score for each sample
            sil_sam = self.silhouette_metric(self.data, self.pred, type='all')
            self.subtaskCompleted.emit()

            success = True
            out = (sil_avg, sil_sam, self.pred)

        except Exception as e:
            success = False
            out = (e,)

        finally:
            self.taskFinished.emit((success, out))



class BalanceThread(QThread): # ??? can be made a generic thread?
    taskFinished = pyqtSignal(tuple)

    def __init__(self):
        super(BalanceThread, self).__init__()

        self.func = lambda: None

    def set_func(self, func):
        self.func = func

    def run(self):
        try:
            out = self.func()
        except Exception as e:
            out = (e,)
        finally:
            self.taskFinished.emit(out)




class LearningThread(QThread):
    epochCompleted = pyqtSignal(tuple) # (epoch, losses, acc)
    updateRequested = pyqtSignal()
    taskFinished = pyqtSignal(tuple) # (True/False, epoch/exception)

    def __init__(self):
        super(LearningThread, self).__init__()

        self.epochs = range(0)
        self.upRate = 0
        self.Y_tr = None
        self.Y_vd = None
        self.func = lambda: None

    def setParameters(self, func, GT, e_range, upRate):
        self.set_func(func)
        self.set_groundTruth(*GT)
        self.set_epochs(*e_range)
        self.set_upRate(upRate)

    def set_func(self, func):
        self.func = func

    def set_groundTruth(self, Y_tr, Y_vd):
        self.Y_tr = Y_tr
        self.Y_vd = Y_vd

    def set_epochs(self, e_min, e_max):
        self.epochs = range(e_min, e_max)

    def set_upRate(self, value):
        self.upRate = value

    def run(self):

        try:
            for e in self.epochs:
            # Check for user cancel request
                if self.isInterruptionRequested():
                    e -= 1
                    break

            # Learn
                tr_loss, vd_loss, tr_pred, vd_pred = self.func()

            # Compute accuracy
                tr_acc = accuracy_score(self.Y_tr, tr_pred)
                vd_acc = accuracy_score(self.Y_vd, vd_pred)

            # Update progress bar and scores
                self.epochCompleted.emit((e, (tr_loss, vd_loss),
                                             (tr_acc,  vd_acc)))

            # Update loss and accuracy plots and labels
                if (e+1) % self.upRate == 0:
                    self.updateRequested.emit()


        # Exit with success
            self.taskFinished.emit((True, None))

        except Exception as exc:
        # Exit with error
            self.taskFinished.emit((False, exc))

        finally:
        # Reset parameters for safety measures
            self.setParameters(lambda: None, (None, None), (0, 0), 0)



