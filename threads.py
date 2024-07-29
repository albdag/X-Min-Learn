# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 21:54:40 2022

@author: albdag
"""

from typing import Callable

from PyQt5.QtCore import QThread, pyqtSignal

import imblearn.over_sampling as OS
import imblearn.under_sampling as US
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

# class BalanceThread(QThread): # ??? can be made a generic thread?
#     taskFinished = pyqtSignal(tuple)

#     def __init__(self):
#         super(BalanceThread, self).__init__()

#         self.func = lambda: None

#     def set_func(self, func):
#         self.func = func

#     def run(self):
#         try:
#             out = self.func()
#         except Exception as e:
#             out = (e,)
#         finally:
#             self.taskFinished.emit(out)


class BalancingThread(MultiTaskThread):
    '''
    Multi task thread for balancing imbalanced training sets using different
    over-sampling and/or under-sampling algorithms. 
    '''
    def __init__(self):
        '''
        Constructor.

        '''
        super(BalancingThread, self).__init__()

        self._init_params()

    
    def _init_params(self):
        '''
        Initialize (or reset) all the required balancing parameters.

        '''
        self.x = None         # input dataset features
        self.y = None         # input dataset labels
        self.strategy = None  # balancing strategy
        self.seed = None      # random seed
        self.os = None        # over-sampling algorithm
        self.us = None        # under-sampling algorithm
        self.kos = None       # k-neighbours (oversampling parameter)
        self.mos = None       # m-neighbours (oversampling parameter)
        self.nus = None       # n-neighbours (undersampling parameter)


    def set_params(self, x: np.ndarray, y: np.ndarray, strategy: dict,
                   seed: int, osa: str|None=None, usa: str|None=None, kos=5, 
                   mos=10, nus=3, njobs=1):
        '''
        Set balancing parameters.

        Parameters
        ----------
        x : np.ndarray
            Input training features from unbalanced dataset.
        y : np.ndarray
            Output training labels from unbalanced dataset.
        strategy : dict
            Dataset balancing strategy. It must indicate the exact value of 
            resampling for each class.
        seed : int 
            Deterministic random state.
        osa : str | None, optional
            Over-sampling algorithm. If None, over-sampling is prevented. The 
            default is None.
        usa : str | None, optional
            Under-sampling algorithm. If None, under-sampling is prevented. The 
            default is None.
        kos : int, optional
            Number of nearest neighbours used to construct synthetic samples 
            during over-sampling. The default is 5.
        mos : int, optional
            Number of nearest neighbours used to determine if a minority sample
            is in danger during over-sampling. The default is 10.
        nus : int, optional
            Number of nearest neighbours used during under-sampling. The 
            default is 3.
        njobs : int, optional
            Number of parallel CPU threads. If -1, all processors are used. The
            default is 1.

        '''
        self.x = x
        self.y = y
        self.strategy = strategy
        self.seed = seed
        self.os = osa
        self.us = usa
        self.kos = kos
        self.mos = mos
        self.nus = nus
        self.njobs = njobs


    def get_oversampler(self, algm: str, strat: dict, kos: int, mos: int,
                        seed: int):
        '''
        Get over-sampling method.

        Parameters
        ----------
        algm : str
            Over-sampling algorithm as a valid string.
        strat : dict
            Over-sampling strategy for each class.
        kos : int
            Number of nearest neighbours used to construct synthetic samples 
            during over-sampling.
        mos : int
            Number of nearest neighbours used by some algorithms to determine 
            if a minority sample is in danger during over-sampling.
        seed : int
            Deterministic random state.

        Returns
        -------
        imblearn.BaseOverSampler
            Over-sampling class.

        Raises
        ------
        KeyError
            Algorithm must be one of ['SMOTE', 'BorderlineSMOTE', 'SVMSMOTE',
            'ADASYN'].

        '''
        if algm == 'SMOTE':
            return OS.SMOTE(sampling_strategy = strat,
                            random_state = seed,
                            k_neighbors = kos)
        
        elif algm == 'BorderlineSMOTE':
            return OS.BorderlineSMOTE(sampling_strategy = strat,
                                      random_state = seed,
                                      k_neighbors = kos,
                                      m_neighbors = mos)
        elif algm == 'SVMSMOTE':
            return OS.SVMSMOTE(sampling_strategy = strat,
                               random_state = seed,
                               k_neighbors = kos,
                               m_neighbors = mos)
        
        elif algm == 'ADASYN':
            return OS.ADASYN(sampling_strategy = strat,
                             random_state = seed,
                             n_neighbors = kos)

        else:
            err = f"Unknown over-sampling algorithm: {algm}. Must be one of "\
                  "the following: ['SMOTE', 'BorderlineSMOTE', 'SVMSMOTE', "\
                  "'ADASYN']"
            raise KeyError(err)


    def get_undersampler(self, algm: str, strat: dict, nus: int, seed: int, 
                         njobs: int):
        '''
        Get under-sampling method.

        Parameters
        ----------
        algm : str
            Under-sampling algorithm as a valid string.
        strat : dict
            Under-sampling strategy for each class.
        nus : int
            Number of nearest neighbours used by some algorithms.
        seed : int
            Deterministic random state.
        njobs : int
             Number of parallel CPU threads. If -1, all processors are used.

        Returns
        -------
        imblearn.BaseUnderSampler
            Under-sampling class.

        Raises
        ------
        KeyError
            Algorithm must be one of ['RandUS', 'NearMiss', 'ClusterCentroids',
            'TomekLinks', 'ENN-all', 'ENN-mode', 'NCR'].

        '''
        if algm == 'RandUS':
            return US.RandomUnderSampler(sampling_strategy = strat,
                                         random_state = seed)
        
        elif algm == 'NearMiss':
            return US.NearMiss(sampling_strategy = strat,
                               n_neighbors = nus,
                               n_jobs = njobs)
        
        elif algm == 'ClusterCentroids':
            return US.ClusterCentroids(sampling_strategy = strat,
                                       random_state = seed)
        
        elif algm == 'TomekLinks':
            return US.TomekLinks(sampling_strategy = list(strat.keys()),
                                 n_jobs = njobs)
        
        elif algm in ('ENN-all', 'ENN-mode'):
            return US.EditedNearestNeighbours(sampling_strategy = list(strat.keys()),
                                              n_neighbors = nus,
                                              kind_sel = algm.split('-')[-1],
                                              n_jobs = njobs)

        elif algm == 'NCR':
            return US.NeighbourhoodCleaningRule(sampling_strategy = list(strat.keys()),
                                                n_neighbors = nus,
                                                n_jobs = njobs)

        else:
            err = f"Unknown under-sampling algorithm: {algm}. Must be one of "\
                  "the following: ['RandUS', 'NearMiss', 'ClusterCentroids', "\
                  "'TomekLinks', 'ENN-all', 'ENN-mode', 'NCR']"
            raise KeyError(err)


    def run(self):
        '''
        Main function of the thread. Defines how the worker should perform its
        tasks.

        '''
    # Store balancing parameters in a dictionary
        info = {'Strategy': self.strategy, 
                'OS': self.os, 
                'US': self.us,
                'n-neigh_US': self.nus, 
                'k-neigh_OS': self.kos, 
                'm-neigh_OS': self.mos, 
                'Seed': self.seed
                }
        
    # S T R A T E G Y
        if self.isInterruptionRequested(): return
        self.taskInitialized.emit('Parsing strategy')
        
    # Update strategy in balancing info dictionary
        unq, cnt = np.unique(self.y, return_counts=True)
        num = list(self.strategy.values())
        strat_by_class = zip(unq, [f'{c} -> {n}' for c, n in zip(cnt, num)])
        info['Strategy'] = dict(strat_by_class)

    # Set over-sampling and under-sampling strategies
        os_strat, us_strat = {}, {}
        for u, c, n in zip(unq, cnt, num):
            if n >= c:
                os_strat[u] = n
            else:
                us_strat[u] = n

        try:
        # U N D E R - S A M P L I N G
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Under-sampling dataset')

            if self.us:
                us_method = self.get_undersampler(self.us, us_strat, self.nus,
                                                self.seed, self.njobs)
                self.x, self.y = us_method.fit_resample(self.x, self.y)

        # O V E R - S A M P L I N G
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Over-sampling dataset')

            if self.os:
                os_method = self.get_oversampler(self.os, os_strat, self.kos,
                                                 self.mos, self.seed)
                self.x, self.y = os_method.fit_resample(self.x, self.y)

        # D A T A S E T  P E R M U T A T I O N
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Applying permutation')

            if self.os or self.us:
                perm = np.random.default_rng(self.seed).permutation(len(self.x))
                x_balanced, y_balanced = self.x[perm], self.y[perm]
            else:
                x_balanced, y_balanced = self.x, self.y


            self.workFinished.emit((x_balanced, y_balanced, info), True)

        except Exception as e:
            self.workFinished((e,), False)

        finally:
            self._init_params()



class LearningThread(QThread):
    epochCompleted = pyqtSignal(tuple) # (epoch, losses, acc)
    renderRequested = pyqtSignal()
    taskFinished = pyqtSignal(tuple, bool) # (result/exception, True/False)

    def __init__(self):
        super(LearningThread, self).__init__()

        self.epochs = range(0)
        self.upRate = 0
        self.y_tr = None
        self.y_vd = None
        self.func = lambda: None


    def setParameters(self, func: Callable, y_tr: np.ndarray, y_vd: np.ndarray, 
                      epoch_range: tuple[int], uprate: int):
        self.set_func(func)
        self.set_ground_truth(y_tr, y_vd)
        self.set_epochs(*epoch_range)
        self.set_uprate(uprate)


    def set_func(self, func):
        self.func = func


    def set_ground_truth(self, y_tr, y_vd):
        self.y_tr = y_tr
        self.y_vd = y_vd


    def set_epochs(self, e_min, e_max):
        self.epochs = range(e_min, e_max)


    def set_uprate(self, value):
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
                tr_acc = sklearn.metrics.accuracy_score(self.y_tr, tr_pred)
                vd_acc = sklearn.metrics.accuracy_score(self.y_vd, vd_pred)

            # Update progress bar and scores
                scores = (e, (tr_loss, vd_loss), (tr_acc,  vd_acc))
                self.epochCompleted.emit(scores)

            # Update loss and accuracy plots and labels
                if (e+1) % self.upRate == 0:
                    self.renderRequested.emit()

        # Exit with success
            self.renderRequested.emit() # force last plot and labels rendering
            self.taskFinished.emit((tr_pred, vd_pred), True)

        except Exception as e:
        # Exit with error
            self.taskFinished.emit((e,), False)

        finally:
        # Reset parameters for safety measures
            self.setParameters(lambda: None, None, None, (0, 0), 0)



