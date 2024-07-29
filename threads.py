# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 21:54:40 2022

@author: albdag
"""
from typing import Callable

from PyQt5.QtCore import QThread, pyqtSignal

import numpy as np
from scipy import ndimage
import sklearn.metrics

from _base import RoiMap


class MultiTaskThread(QThread):
    '''
    Base function for threads defined by a fixed known amount of tasks. It is
    equipped with custom signals to inform when each task is initialized and
    when the entire procedure is completed or interrupted.
    '''
    taskInitialized = pyqtSignal(str)
    workInterrupted = pyqtSignal()
    workFinished = pyqtSignal(tuple, bool)

    def __init__(self):
        '''
        Constructor.
        '''
        super(MultiTaskThread, self).__init__()

    
    def isInterruptionRequested(self):
        '''
        Add a custom signal to the default isInterruptionRequested function.

        Returns
        -------
        interrupt : bool
            If the thread interruption has been requested.

        '''
        interrupt = super(MultiTaskThread, self).isInterruptionRequested()
        if interrupt:
            self.workInterrupted.emit()
        return interrupt


    def run(self):
        '''
        Main function of the thread. Defines how the worker should perform its
        tasks. To be reimplemented in each child.

        '''
        return


class MineralClassificationThread(MultiTaskThread):
    '''
    A special type of multitask thread, tailored for all mineral classification
    procedures. 
    '''
    def __init__(self):
        '''
        Constructor.

        '''
        super(MineralClassificationThread, self).__init__()
    
    # Set main attributes
        self.classifier = None
        self.algorithm = None


    def set_classifier(self, classifier):
        '''
        Set the classifier that is requesting this worker. Useful to access to
        the classifiers attributes and methods.

        Parameters
        ----------
        classifier : _ClassifierBase
            Classifier.
        '''
        self.classifier = classifier
        self.algorithm = classifier.algorithm


    def reset_classifier(self):
        '''
        Reset the classifier attributes for safety measures.

        '''
        self.classifier = None
        self.algorithm = None


    def run(self):
        '''
        Main function of the thread. Defines how the worker should perform its
        tasks. To be reimplemented in each child.

        '''     
        if self.classifier is None or self.algorithm is None: 
            return



class ModelBasedClassificationThread(MineralClassificationThread):
    '''
    A mineral classification thread specialized for supervised ML model based
    classifers.
    '''
    def __init__(self):
        '''
        Constructor.

        '''
        super().__init__()


    def run(self):
        '''
        Main function of the thread. Defines how the worker should perform its
        tasks. 

        '''
        super().run()

        try:
        # Pre-process feature data
            self.taskInitialized.emit('Pre-processing data')
            feat_data = self.classifier.preProcessFeatureData()

        # Predict unknown data and calculate probability scores
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Identifying mineral classes')
            prob, pred = self.algorithm.predict(feat_data.float())

        # Reconstruct the result
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Reconstructing mineral map')
            prob = prob.detach().numpy().round(2)
            pred = self.classifier.decodeLabels(pred)

        # Send the workFinished signal with success
            self.workFinished.emit((pred, prob), True)

        except Exception as e:
        # Send the workFinished signal with error
            self.workFinished.emit((e,), False)

        finally:
        # Reset parameters for safety measures
            self.reset_classifier()


class RoiBasedClassificationThread(MineralClassificationThread):
    '''
    A mineral classification thread specialized for ROI based classifers.
    '''
    def __init__(self):
        '''
        Constructor.

        '''
        super().__init__()


    def run(self):
        '''
        Main function of the thread. Defines how the worker should perform its
        tasks. 

        '''
        super().run()

        try:
        # Extract training data
            self.taskInitialized.emit('Collecting training data')
            x_train, y_train, in_data = self.classifier.getTrainingData()

        # "Train" the classifier
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Training classifier')
            self.algorithm.fit(x_train, y_train)

        # Predict unknown data
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Identifying mineral classes')
            pred = self.algorithm.predict(in_data)

        # Compute probability score
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Calculating probability map')
            prob = self.algorithm.predict_proba(in_data).max(axis=1)

        # Send the workFinished signal with success
            self.workFinished.emit((pred, prob), True)

        except Exception as e:
        # Send the workFinished signal with error
            self.workFinished.emit((e,), False)

        finally:
        # Reset parameters for safety measures
            self.reset_classifier()



class UnsupervisedClassificationThread(MineralClassificationThread):
    '''
    A mineral classification thread specialized for unsupervised classifiers.
    '''
    def __init__(self):
        '''
        Constructor.

        '''
        super().__init__()


    def run(self):
        '''
        Main function of the thread. Defines how the worker should perform its
        tasks. 
        
        '''
        super().run()

        try:
        # Pre-process input data
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Pre-processing data')
            in_data = self.classifier.preProcessFeatureData()
            
        # Fit data
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Fitting data')
            self.algorithm.fit(in_data)

        # Cluster data
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Clustering data')
            pred = self.algorithm.predict(in_data)

        # Compute probability score
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Computing probability score')
            dist = self.algorithm.transform(in_data).min(axis=1)
            prob = 1 - dist/dist.max()

        # Compute silhouette score (by cluster)
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Computing silhouette score by cluster')
            sil_clust = None
            if self.classifier.do_silhouette_score:
                sample_size = int(self.classifier.silhouette_ratio * pred.size)
                rng = np.random.default_rng(self.classifier.seed)
                subset_idx = rng.permutation(pred.size)[:sample_size]
                _in_data, _pred = in_data[subset_idx, :], pred[subset_idx]
                sil_sam = sklearn.metrics.silhouette_samples(_in_data, _pred)
                unq_val = np.unique(_pred)
                sil_clust = {u: np.sort(sil_sam[_pred == u]) for u in unq_val}

        # Compute silhouette score (average)
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Computing average silhouette score')
            sil_avg = None
            if sil_clust:
                sil_avg = np.mean(sil_sam)

        # Compute Calinski-Harabasz Index (CHI) score
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Computing Calinski-Harabasz Index')
            chi = None
            if self.classifier.do_chi_score:
                chi = sklearn.metrics.calinski_harabasz_score(in_data, pred)

        # Compute Davies-Bouldin Index (DBI) score
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Computing Davies-Bouldin Index')
            dbi = None
            if self.classifier.do_dbi_score:
                dbi = sklearn.metrics.davies_bouldin_score(in_data, pred)

        # Send the workFinished signal with success
            out = (pred.astype('U8'), prob, sil_avg, sil_clust, chi, dbi)
            self.workFinished.emit(out, True)

        except Exception as e:
        # Send the workFinished signal with error
            self.workFinished.emit((e,), False)

        finally:
        # Reset parameters for safety measures
            self.reset_classifier()



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
            return sklearn.metrics.silhouette_score(data, pred, metric='euclidean')
        elif type == 'all':
            return sklearn.metrics.silhouette_samples(data, pred, metric='euclidean')
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



class NpvThread(MultiTaskThread):
    '''
    Multi task thread that computes the cumulative Neighborhood Pixel Variance
    (NPV) map.
    '''
    def __init__(self):
        '''
        Constructor.

        '''
        super().__init__()

    # Set main attributes
        self.arrays = list()
        self.map_names = list()
        self.size = 0
        self.npv_func = None

    def set_params(self, arrays: list[np.ndarray], map_names: list[str], 
                   size:int, npv_func: Callable):
        '''
        Set worker's required parameters.

        Parameters
        ----------
        arrays : list of ndarrays
            Input maps arrays.
        map_names : list[str]
            Maps names.
        size : int
            Size of the squared kernel.
        npv_func : Callable
            NPV type function. Can be one of np.sum, np.median, np.mean.
        '''
        self.arrays = arrays
        self.map_names = map_names
        self.size = size
        self.npv_func = npv_func


    def run(self):
        '''
        Main function of the thread. Defines how the worker should perform its
        tasks. 

        '''
        try:
        # Set derived parameters
            c = self.size**2 // 2  # central pixel index of NPV kernel

        # Compute NPV for each map
            npv_list = []
            for arr, name in zip(self.arrays, self.map_names):
                if self.isInterruptionRequested(): return

            # Neighborhood pixel variance (NPV)
                self.taskInitialized.emit(f'Computing NPV for {name} map')
                npv = ndimage.generic_filter(arr, lambda a: self.npv_func(
                                             np.abs(a-a[c])), self.size)
            # NPV rescaling
                npv = npv/(arr.max() - arr.min())
            # Append npv map to npv_list
                npv_list.append(npv)

        # Send the workFinished signal with success and return the 
        # (normalized) cumulative NPV map
            npv_sum = sum(npv_list)
            npv_sum = npv_sum/npv_sum.max() # normalize to range [0, 1]
            self.workFinished.emit((npv_sum,), True)

        except Exception as e:
        # Send the workFinished signal with error
            self.workFinished.emit((e,), False)



class RoiDetectionThread(MultiTaskThread):
    '''
    Multi task thread that automatically identifies ROIs from an NPV map.

    '''
    def __init__(self):
        '''
        Constructor.

        '''
        super().__init__()

    # Set main attributes
        self.n_roi = 0
        self.size = 0
        self.distance = 0
        self.npv_map = None
        self.current_roimap = None


    def set_params(self, n_roi: int, size: int, distance: int, 
                   npv_map: np.ndarray, current_roimap: RoiMap):
        '''
        Set worker's required parameters.

        Parameters
        ----------
        n_roi : int
            Number of requested ROIs
        size : int
            Size of each (squared) ROI.
        distance : int
            Minimum tolerated distance between each ROI.
        npv_map : np.ndarray
            Neighborhood Pixel Variance map.
        current_roimap : RoiMap
            Currently existent ROI map.

        '''
        self.n_roi = n_roi
        self.size = size
        self.distance = distance
        self.npv_map = npv_map
        self.current_roimap = current_roimap


    def run(self):
        '''
        Main function of the thread. Defines how the worker should perform its
        tasks. 

        '''
        try:
        # Set derived parameters
            r = self.size // 2    # ROI radius
            d = r + self.distance
            shape = self.npv_map.shape

        # Exclude borders
            mask = np.ones(shape)
            mask[r:-r, r:-r] = 0
            self.npv_map[mask==1] = np.nan

        # Sort flattened cumulative NPV map by values (smaller values first)
            idx = np.argsort(self.npv_map, axis=None) 

        # Create a new empty ROI map
            auto_roimap = RoiMap.from_shape(shape)

        # Detect best ROIs
            rois_found = 0 
            self.taskInitialized.emit(f'Detecting ROI (1/{self.n_roi})')
            for i in idx:
                if self.isInterruptionRequested(): return

            # Get the original row-col coords from the index of the pixel
                row, col = np.unravel_index(i, shape)
            # Skip NaN pixels (border of NPV cumulative map)
                if np.isnan(self.npv_map[row, col]):
                    continue
            # Fix out of range ROI extents
                y0, y1, x0, x1 = row-d, row+d, col-d, col+d
                if y0 < 0: y0 = 0
                if y1 > shape[0]: y1 = shape[0]
                if x0 < 0: x0 = 0
                if x1 > shape[1]: x1 = shape[1] 
            # Allow only ROIs that do not overlap with existent ROIs
                if (not self.current_roimap.extents_overlaps((x0,x1,y0,y1)) 
                    and not auto_roimap.extents_overlaps((x0,x1,y0,y1))):
                    bbox = auto_roimap.extents_to_bbox((col-r, col+r+1,
                                                        row-r, row+r+1))
                    auto_roimap.add_roi('-', bbox)
                    rois_found += 1
                # Kill the loop if the requested number of ROIs are found
                    if rois_found == self.n_roi:
                        break
                    else:
                        txt = f'Detecting ROI ({rois_found+1}/{self.n_roi})'
                        self.taskInitialized.emit(txt)

        # Send the workFinished signal with success and return the auto 
        # detected roimap
            self.workFinished.emit((auto_roimap,), True)

        except Exception as e:
        # Send the workFinished signal with error
            self.workFinished.emit((e,), False)



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
                tr_acc = sklearn.metrics.accuracy_score(self.Y_tr, tr_pred)
                vd_acc = sklearn.metrics.accuracy_score(self.Y_vd, vd_pred)

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



