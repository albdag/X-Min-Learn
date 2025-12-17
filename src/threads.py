# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 21:54:40 2022

@author: albdag
"""

from collections.abc import Callable

from PyQt5.QtCore import QThread, pyqtSignal

import numpy as np
from scipy import ndimage

from _base import RoiMap


class DynamicStepsThread(QThread):

    iterCompleted = pyqtSignal(int)
    taskInterrupted = pyqtSignal()
    taskFinished = pyqtSignal(tuple, bool)

    def __init__(self) -> None:
        '''
        Base class for working threads operating a main task with a variable
        amount of iterations. It is equipped with custom signals to inform when
        each iteration is completed and when the entire procedure is completed
        or interrupted.

        '''
        super().__init__()
        self.task = None
        self.params = ()


    def set_task(self, function: Callable) -> None:
        '''
        Set the main task of the thread. This method must be called before
        starting the thread.

        Parameters
        ----------
        function : Callable
            The main task of the thread.

        '''
        self.task = function


    def set_params(self, *args) -> None:
        '''
        Set optional parameters for the task.

        Parameters
        ----------
        args
            Sequence of parameters.

        '''
        self.params = args


    def isInterruptionRequested(self) -> bool:
        '''
        Add a custom signal to the default 'isInterruptionRequested' method.

        Returns
        -------
        interrupt : bool
            If the thread interruption has been requested.

        '''
        interrupt = super().isInterruptionRequested()
        if interrupt:
            self.taskInterrupted.emit()
            self.reset()
        return interrupt


    def run(self) -> None:
        '''
        Main method of the thread. Defines how the worker should perform its
        task. To be reimplemented in each child.

        Raises
        ------
        ValueError
            Raised if the thread's task is not set.

        '''
        if self.task is None:
            raise ValueError('Task is not set.')
    

    def reset(self) -> None:
        '''
        Reset the thread parameters for safety measures. This method should be
        internally called by the thread when terminated.

        '''
        self.task = None
        self.params = ()



class FixedStepsThread(QThread):

    taskInitialized = pyqtSignal(str)
    workInterrupted = pyqtSignal()
    workFinished = pyqtSignal(tuple, bool)

    def __init__(self) -> None:
        '''
        Base class for working threads operating a fixed amount of tasks, which
        are provided as a pipeline of functions. It is equipped with custom
        signals to inform when each task is initialized and when the entire 
        procedure is completed or interrupted.

        '''
        super().__init__()
        self.pipeline = ()
        self.args = ()
    

    def set_pipeline(self, pipeline: tuple[Callable, ...], *args):
        '''
        Set the tasks pipeline. This method must be called before starting the
        thread.

        Parameters
        ----------
        pipeline : tuple[Callable]
            Sequence of functions to be performed by the thread.
        args 
            Sequence of extra parameters required by pipeline's functions. 

        '''
        self.pipeline = pipeline
        self.args = args

    
    def isInterruptionRequested(self) -> bool:
        '''
        Add a custom signal to the default 'isInterruptionRequested' method.

        Returns
        -------
        interrupt : bool
            If the thread interruption has been requested.

        '''
        interrupt = super().isInterruptionRequested()
        if interrupt:
            self.workInterrupted.emit()
            self.reset()
        return interrupt


    def run(self) -> None:
        '''
        Main method of the thread. Defines how the worker should perform its
        task. To be reimplemented in each child.

        Raises
        ------
        ValueError
            Raised if the thread's pipeline is not set.

        '''
        if not self.pipeline:
            raise ValueError('Tasks pipeline is not set.')
    

    def reset(self) -> None:
        '''
        Reset the thread parameters for safety measures. This method should 
        be called by the thread when terminated.

        '''
        self.pipeline = ()
        self.args = ()



class LearningThread(DynamicStepsThread):

    renderRequested = pyqtSignal(tuple)

    def __init__(self) -> None:
        '''
        Dynamic steps thread specialized for supervised learning sessions. This
        thread expects a task (see 'set_task' method from parent class) that
        returns the following four outputs:
            o1. Training loss -> float
            o2. Testing loss -> float
            o3. Training accuracy -> float
            o4. Testing accuracy -> float
        Moreover, the following parameters must be provided (see 'set_params'
        method from parent class):
            p1. Starting epoch -> int
            p2. Ending epoch -> int
            p3. Graphics update rate -> int
            p4. Training losses -> list[float]
            p5. Testing losses -> list[float]
            p6. Training accuracies -> list[float]
            p7. Testing accuracies -> list[float]

        '''
        super().__init__()


    def run(self) -> None:
        '''
        Main method of the thread. Defines how the worker should perform its
        task. 

        '''
        super().run()

        (e_min, e_max, uprate, tr_loss_list, ts_loss_list, tr_acc_list, 
         ts_acc_list) = self.params

        try:
            for e in range(e_min, e_max):
            # Check for user cancel request
                if self.isInterruptionRequested():
                    e -= 1
                    break

            # Learn
                tr_loss, ts_loss, tr_acc, ts_acc = self.task()
            
            # Store loss and accuracy values
                tr_loss_list.append(tr_loss)
                ts_loss_list.append(ts_loss)
                tr_acc_list.append(tr_acc)
                ts_acc_list.append(ts_acc)

            # Update progress bar
                self.iterCompleted.emit(e)

            # Update loss and accuracy plots and labels
                if (e + 1) % uprate == 0:
                    self.renderRequested.emit((
                        tr_loss_list, ts_loss_list, tr_acc_list, ts_acc_list))

        # Force last plot and labels rendering and exit with success 
            self.renderRequested.emit((
                tr_loss_list, ts_loss_list, tr_acc_list, ts_acc_list)) 
            self.taskFinished.emit((
                tr_loss_list, ts_loss_list, tr_acc_list, ts_acc_list), True)

        except Exception as exc:
        # Exit with error
            self.taskFinished.emit((exc, ), False)

        finally:
        # Reset parameters for safety measures
            self.reset()



class CsvChunkReadingThread(DynamicStepsThread):

    def __init__(self) -> None:
        '''
        Dynamic steps thread specialized for CSV chunk reading. This thread 
        expects the 'read_csv' function from pandas to be set as task (see
        'set_task' method from parent class). No extra parameters are required.

        '''
        super().__init__()


    def run(self) -> None:
        '''
        Main method of the thread. Defines how the worker should perform its
        task. 

        '''
        super().run()
        chunk_list = []
        
        try:
        # Read CSV and return chunks
            with self.task() as reader:
                for i, chunk in enumerate(reader, start=1):
                    if self.isInterruptionRequested(): 
                        return
                    chunk_list.append(chunk)
                    self.iterCompleted.emit(i)

            self.taskFinished.emit(tuple(chunk_list), True)
        
        except Exception as exc:
        # Exit with error
            self.taskFinished.emit((exc, ), False)

        finally:
        # Reset parameters for safety measures
            self.reset()



class ModelBasedClassificationThread(FixedStepsThread):

    def __init__(self) -> None:
        '''
        A fixed steps thread specialized for supervised model-based
        classifications. This thread expects a pipeline (see 'set_pipeline'
        method from parent class) that consists of the following functions:
            f1. Pre-process features (None) -> o1 (Tensor)
            f2. Predict targets (o1) -> o2.1 (Tensor), o2.2 (Tensor)
            f3. Post-process targets (o2.1, o2.2) -> o3.1 (array), o3.2 (array)
        No extra arguments are required.

        '''
        super().__init__()


    def run(self) -> None:
        '''
        Main method of the thread. Defines how the worker should perform its
        tasks. 

        '''
        super().run()
        pre_process, predict, post_process = self.pipeline

        try:
        # Pre-process feature data
            self.taskInitialized.emit('Pre-processing data')
            feat_data = pre_process()

        # Predict unknown data and calculate probability scores
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Identifying mineral classes')
            prob, pred = predict(feat_data)

        # Reconstruct the result
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Reconstructing mineral map')
            prob, pred = post_process(prob, pred)

        # Send the workFinished signal with success
            self.workFinished.emit((pred, prob), True)

        except Exception as e:
        # Send the workFinished signal with error
            self.workFinished.emit((e, ), False)

        finally:
        # Reset parameters for safety measures
            self.reset()



class RoiBasedClassificationThread(FixedStepsThread):

    def __init__(self) -> None:
        '''
        A fixed steps thread specialized for ROI-based classifications. This
        thread expects a pipeline (see 'set_pipeline' method from parent class)
        that consists of the following functions:
            f1. Get training data (None) -> o1.1 (array), o1.2 (array), o1.3 (array)
            f2. Fit classifier (o1.1, o1.2) -> None
            f3. Predict targets (o1.3) -> o3 (array)
            f4. Compute probability score (o1.3) -> o4 (array)
        No extra arguments are required.

        '''
        super().__init__()


    def run(self) -> None:
        '''
        Main method of the thread. Defines how the worker should perform its
        tasks. 

        '''
        super().run()
        get_train_data, fit, predict, compute_prob = self.pipeline

        try:
        # Extract training data
            self.taskInitialized.emit('Collecting training data')
            x_train, y_train, in_data = get_train_data()

        # "Train" the classifier
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Training classifier')
            fit(x_train, y_train)

        # Predict unknown data
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Identifying mineral classes')
            pred = predict(in_data)

        # Compute probability score
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Calculating probability map')
            prob = compute_prob(in_data)

        # Send the workFinished signal with success
            self.workFinished.emit((pred, prob), True)

        except Exception as e:
        # Send the workFinished signal with error
            self.workFinished.emit((e, ), False)

        finally:
        # Reset parameters for safety measures
            self.reset()



class UnsupervisedClassificationThread(FixedStepsThread):

    def __init__(self) -> None:
        '''
        A fixed steps thread specialized for unsupervised classifications. This
        thread expects a pipeline (see 'set_pipeline' method from parent class)
        that consists of the following functions:
            f1. Pre-process data (None) -> o1 (array)
            f2. Fit classifier (o1) -> None
            f3. Cluster data (o1) -> o3 (array)
            f4. Compute probability score (o1) -> o4 (array)
            f5. Compute Silhouette score (o1, o3) -> o5.1 (dict), o5.2 (float)
            f6. Compute Calinski-Harabasz Index (o1, o3) -> o6 (float)
            f7. Compute Davies-Bouldin Index (o1, o3) -> o7 (float)
        No extra arguments are required.

        '''
        super().__init__()


    def run(self) -> None:
        '''
        Main method of the thread. Defines how the worker should perform its
        tasks. 
        
        '''
        super().run()
        (pre_process, fit, predict, compute_prob, sil_score, chi_score, 
         dbi_score) = self.pipeline

        try:
        # Pre-process input data
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Pre-processing data')
            in_data = pre_process()
            
        # Fit data
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Fitting data')
            fit(in_data)

        # Cluster data
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Clustering data')
            pred = predict(in_data)

        # Compute probability score
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Computing probability score')
            prob = compute_prob(in_data)

        # Compute silhouette score
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Computing silhouette score')
            sil_clust, sil_avg = sil_score(in_data, pred)

        # Compute Calinski-Harabasz Index (CHI) score
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Computing Calinski-Harabasz Index')
            chi = chi_score(in_data, pred)

        # Compute Davies-Bouldin Index (DBI) score
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Computing Davies-Bouldin Index')
            dbi = dbi_score(in_data, pred)

        # Send the workFinished signal with success
            out = (pred, prob, sil_avg, sil_clust, chi, dbi)
            self.workFinished.emit(out, True)

        except Exception as e:
        # Send the workFinished signal with error
            self.workFinished.emit((e,), False)

        finally:
        # Reset parameters for safety measures
            self.reset()



class BalancingThread(FixedStepsThread):

    def __init__(self) -> None:
        '''
        Fixed steps thread specialized for balancing training sets. This thread
        expects a pipeline (see 'set_pipeline' method from parent class) that
        consists of the following functions:
            f1. Parse strategy (a1) -> o1.1 (dict), o1.2 (dict), o1.3 (dict)
            f2. Undersample (o1.2, a4, a2, a7, a8) -> o2.1 (array), o2.2 (array)
            f3. Oversample (o1.1, a3, a2, a5, a6, o2.1, o2.2) -> o3.1 (array), o3.2 (array)
            f4. Shuffle (o3.1, o3.2, a2) -> o4.1 (array), o4.2 (array)
        Moreover, the following extra arguments (see 'set_pipeline' method from
        parent class) are required:
            a1. Balancing strategy -> int | str | dict
            a2. Seed -> int
            a3. Oversampling algorithm -> str | None
            a4. Undersampling algorithm -> str | None
            a5. K-neighbors for oversampler -> int
            a6. M-neighbors for oversampler -> int
            a7. N-neigbors for oversampler -> int
            a8. Number of CPU cores -> int

        '''
        super().__init__()


    def run(self) -> None:
        '''
        Main method of the thread. Defines how the worker should perform its
        tasks. 

        '''
        super().run()

        parse_strat, oversample, undersample, shuffle = self.pipeline
        strategy, seed, osa, usa, kos, mos, nus, njobs = self.args
        balancing_info = {}

        try:
        # Parse over-sampling and under-sampling strategies
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Parsing strategy')
            os_strat, us_strat, strat_info = parse_strat(strategy, verbose=True)
            balancing_info['Strategy'] = strat_info
            balancing_info['OS'] = osa
            balancing_info['US'] = usa
            balancing_info['n-neigh_US'] = nus
            balancing_info['k-neigh_OS'] = kos
            balancing_info['m-neigh_OS'] = mos
            balancing_info['Seed'] = seed

        # Compute under-sampling
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Under-sampling dataset')
            x_train, y_train = undersample(us_strat, usa, seed, nus, njobs)

        # Compute over-sampling
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Over-sampling dataset')
            x_train, y_train = oversample(os_strat, osa, seed, kos, mos,
                                          x=x_train, y=y_train)

        # Shuffle dataset
            if self.isInterruptionRequested(): return
            self.taskInitialized.emit('Shuffling dataset')
            if osa or usa:
                x_train, y_train = shuffle(x_train, y_train, seed=seed)

        # Send the workFinished signal with success
            self.workFinished.emit((x_train, y_train, balancing_info), True)

        except Exception as e:
        # Send the workFinished signal with error
            self.workFinished.emit((e,), False)

        finally:
        # Reset parameters for safety measures
            self.reset()



class NpvThread(FixedStepsThread): 
    # Should NOT inherit from FixedStepsThread, as the n. of possible steps is
    # data driven. Should be moved to DynamicStepsThread. Also move the logic
    # of the function to master code, as thread is just a worker and should
    # just execute commands.

    def __init__(self) -> None:
        '''
        Multi task thread specialized for computing a cumulative Neighborhood
        Pixel Variance (NPV) map.

        '''
        super().__init__()

    def set_params(
        self,
        arrays: list[np.ndarray],
        map_names: list[str], 
        size: int,
        npv_func: Callable
    ) -> None:
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


    def reset(self) -> None:
        '''
        Reset the thread parameters for safety measures. This method is 
        automatically called by the thread when terminated.

        '''
        self.arrays = []
        self.map_names = []
        self.size = 0
        self.npv_func = None


    def run(self) -> None:
        '''
        Main method of the thread. Defines how the worker should perform its
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
            self.workFinished.emit((npv_sum, ), True)

        except Exception as e:
        # Send the workFinished signal with error
            self.workFinished.emit((e, ), False)

        finally:
        # Reset parameters for safety measures
            self.reset()



class RoiDetectionThread(FixedStepsThread):
    # Should NOT inherit from FixedStepsThread, as the n. of possible steps is
    # data driven. Should be moved to DynamicStepsThread. Also move the logic
    # of the function to master code, as thread is just a worker and should
    # just execute commands.

    def __init__(self) -> None:
        '''
        Multi task thread that automatically identifies ROIs from an NPV map.

        '''
        super().__init__()

    # Set main attributes
        self.n_roi = 0
        self.size = 0
        self.distance = 0
        self.npv_map = None
        self.current_roimap = None


    def set_params(
        self,
        n_roi: int,
        size: int,
        distance: int, 
        npv_map: np.ndarray,
        current_roimap: RoiMap
    ) -> None:
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


    def reset(self) -> None:
        '''
        Reset the thread parameters for safety measures. This method is 
        automatically called by the thread when terminated.

        '''
        self.n_roi = 0
        self.size = 0
        self.distance = 0
        self.npv_map = None
        self.current_roimap = None


    def run(self) -> None:
        '''
        Main method of the thread. Defines how the worker should perform its
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

        finally:
        # Reset parameters for safety measures
            self.reset()