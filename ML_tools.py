# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:06:51 2021

@author: dagos
"""

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             silhouette_score, silhouette_samples,
                             calinski_harabasz_score, davies_bouldin_score)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import torch

from _base import InputMap, MineralMap, RoiMap
from ExternalThreads import MineralClassificationThread
import conv_functions as CF



class SoftMaxRegressor(torch.nn.Module):
    def __init__(self, in_features, out_classes, seed=None):
        super(SoftMaxRegressor, self).__init__()

        self._name = 'Softmax Regression'
        self._loss = 'Cross-Entropy loss'

        if seed is not None:
            torch.random.manual_seed(seed)

        self.linear = torch.nn.Linear(in_features, out_classes) #il regressore softmax restituisce
        #distribuzioni di probabilità, quindi il numero di feature di output coincide con il
        #numero di classi
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = torch.nn.CrossEntropyLoss() # Set the Cross-Entropy as loss

    def get_weight(self):
        return self.linear.weight

    def get_bias(self):
        return self.linear.bias

    def forward(self, x):
        """Definisce come processare l'input x"""
        scores = self.linear(x)
        return scores

    def predict(self, x):
        '''Definisce come applicare la funzione softmax ai logit (z) ottenuti da self.linear'''
        self.eval()
        z = self.linear(x)
        probs, classes = self.softmax(z).max(1)
        return probs, classes

    def learn(self, X_train, Y_train, X_test, Y_test, optimizer, device):
        '''Definisce una singola iterazione di learning'''

        self.train()
        out = self.forward(X_train.to(device))
        l = self.loss(out, Y_train.long().to(device))
        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss = l.cpu().detach().numpy().item()
        train_preds = out.max(1)[1].cpu()

        self.eval()
        with torch.set_grad_enabled(False):
            out = self.forward(X_test.to(device))
            l = self.loss(out, Y_test.long().to(device))

            test_loss = l.cpu().detach().numpy().item()
            test_preds = out.max(1)[1].cpu()

        return (train_loss, test_loss, train_preds, test_preds)



class EagerModel():

    def __init__(self, vrb_dict, model_path=None): # variables dict

        self.vrb_dict = vrb_dict
        self.filepath = model_path

        self._base_vrb = ['algm_name', 'loss_name', 'optim_name',
                          'ordered_Xfeat', 'Y_dict', 'device', 'seed',
                          'parentModel_path', 'GT_dataset_path', 'TVT_rateos',
                          'balancing_info', 'regressorDegree', 'epochs', 'lr',
                          'wd', 'mtm', 'accuracy', 'loss', 'F1_scores']
        self._extended_vrb = ['accuracy_list', 'loss_list', 'standards',
                              'optim_state_dict', 'model_state_dict']

        if len(mv := self.missingVariables()):
            print(f'Warning, missing variables {mv}')

    @classmethod
    def load(cls, model_path):
        vrb_dict = torch.load(model_path)
        return cls(vrb_dict, model_path)

    @property
    def algorithm(self):
        return self.vrb_dict.get('algm_name')

    @property
    def inFeat(self):
        return self.vrb_dict.get('ordered_Xfeat')

    @property
    def encoder(self):
        return self.vrb_dict.get('Y_dict')

    @property
    def xMean(self):
        return self.vrb_dict.get('standards')[0]

    @property
    def xStd(self):
        return self.vrb_dict.get('standards')[1]

    @property
    def stateDict(self):
        return self.vrb_dict.get('model_state_dict')

    @property
    def regrDegree(self):
        return self.vrb_dict.get('regressorDegree')

    def getTrainedModel(self):
        infeat = self.inFeatPoly() if self.regrDegree > 1 else len(self.inFeat)
        outcls = len(self.encoder)
        model = getNetworkArchitecture(self.algorithm, infeat, outcls)

        if model is not None:
            model.load_state_dict(self.stateDict)
            return model

# TODO find a more elegant solution (equation?)
    def inFeatPoly(self):
        infeat = len(self.inFeat)
        dummy_arr = np.arange(infeat).reshape(1, infeat)
        _, n_poly_feat = map2Polynomial(dummy_arr, self.regrDegree,
                                        return_n_features = True)
        return n_poly_feat

    def save(self, outpath, log_path=False, extended_log=False):
        torch.save(self.vrb_dict, outpath)
        if log_path:
            self.saveLog(log_path, extended_log)

    def saveLog(self, outpath, extended=False):
        with open(outpath, 'w') as log:
            for k, v in self.vrb_dict.items():
                if not extended and k in self._extended_vrb:
                    continue
                log.write(f'{k.upper()}\n{repr(v)}\n\n\n')

    def generateLogPath(self, path):
        if path is None: return
        logpath = CF.extendFileName(path, '_log', ext='.txt')
        return logpath

    def missingVariables(self):
        required_vrb = self._base_vrb + self._extended_vrb
        missing = set(required_vrb) - set(self.vrb_dict.keys())
        return missing












class ModelBasedClassifier():

    def __init__(self, model, inmaps, mask=None): # model = EagerModel

        self.model = model
        self.inmaps = inmaps
        self.mask = mask
        self.map_shape = inmaps[0].map.shape

        self.algorithm = model.getTrainedModel()
        self.classification_steps = 4
        self.thread = self.ClassificationThread()

    def preProcessInputData(self):
        in_data = [i.map for i in self.inmaps]

    # Apply mask if present
        if self.mask is not None:
            in_data = [np.ma.masked_where(self.mask, i) for i in in_data]

    # Merge input data in a classifier-friendly reshaped array (npix x nmaps)
        in_data = mergeMaps(in_data, masked=self.mask is not None)

    # Map features from linear to polynomial if required
        if (regr_degree := self.model.regrDegree) > 1:
            in_data = map2Polynomial(in_data, regr_degree)

    # Standardize data
        in_data = torch.tensor(in_data)
        in_data = norm_data(in_data, self.model.xMean, self.model.xStd,
                            return_standards=False)

        return in_data

    def encodeLabels(self, array, dtype='int16'):
        '''From labels to IDs.'''
        res = np.copy(array)
        for k, v in self.model.encoder.items(): res[array==k] = v
        return res.astype(dtype)

    def decodeLabels(self, array, dtype='U8'):
        '''From IDs to labels.'''
        res = np.copy(array).astype(dtype)
        for k, v in self.model.encoder.items(): res[array==v] = k
        return res

    def startThreadedClassification(self):
        self.thread.set_classifier(self)
        self.thread.start()



    class ClassificationThread(MineralClassificationThread):

        def __init__(self):
            super().__init__()

        def run(self):
            super().run()

            try:
            # Pre-process input data
                self.taskInitialized.emit('Pre-processing data')
                in_data = self.classifier.preProcessInputData()

            # Predict unknown data and calculate probability scores
                if self.isInterruptionRequested(): return
                self.taskInitialized.emit('Identifying mineral classes')
                prob, pred = self.algorithm.predict(in_data.float())

            # Reconstruct the result
                if self.isInterruptionRequested(): return
                self.taskInitialized.emit('Reconstructing mineral map')
                shape = self.classifier.map_shape
                prob = prob.reshape(shape)
                pred = self.classifier.decodeLabels(pred).reshape(shape)

            # Send the workFinished signal with success
                self.workFinished.emit((pred, prob), True)

            except Exception as e:
            # Send the workFinished signal with error
                self.workFinished.emit((e,), False)

            finally:
            # Reset parameters for safety measures
                self.reset_classifier()



class _RoiBasedClassifier():

    def __init__(self, inmaps, roimap, mask=None, pixel_proximity=False):
        self.inmaps = inmaps
        self.roimap = roimap
        self.mask = mask
        self.map_shape = inmaps[0].shape
        self.proximity = pixel_proximity

        self.algorithm = None  # to be reimplemented in each child class
        self.classification_steps = 6
        self.thread = self.ClassificationThread()

    def getCoordMaps(self):
        shape = self.map_shape
        xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        return xx, yy


    def preProcessFeatureData(self):
    # Include pixel coordinate maps if required
        feat_data = [i.map for i in self.inmaps]
        if self.proximity:
            feat_data.extend(self.getCoordMaps())

    # Apply mask if required:
        if self.mask is not None:
            feat_data = [np.ma.masked_where(self.mask, i) for i in feat_data]

    # Merge input data in a classifier-friendly reshaped array (npix x nmaps)
        feat_data = mergeMaps(feat_data, masked=self.mask is not None)

    # Normalize the data
        feat_data = norm_data(feat_data, return_standards=False, engine='numpy')

        return feat_data


    def preProcessRoiData(self):
        roimap = self.roimap.map

    # Apply mask if required and flatten/compress the map array
        if self.mask is not None:
            roimap = np.ma.masked_where(self.mask, roimap).compressed()
        else:
            roimap = roimap.flatten()

        return roimap


    def getTrainingData(self, return_full_input=True):
    # Get pre-processed X (feature) and Y (label) data
        X = self.preProcessFeatureData()
        Y = self.preProcessRoiData()

    # Extract indices where Y (ROI map data) is actually populated with labels
        labeled_indices = (Y != self.roimap._ND).nonzero()[0]

    # Use the indices to extract training data from both features and labels
        x_train = X[labeled_indices, :]
        y_train = Y[labeled_indices]

        tr_data = [x_train, y_train]
        if return_full_input: tr_data.append(X)
        return tr_data

    def startThreadedClassification(self):
        self.thread.set_classifier(self)
        self.thread.start()

    class ClassificationThread(MineralClassificationThread):

        def __init__(self):
            super().__init__()

        def run(self):
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

            # Calculate probability score
                if self.isInterruptionRequested(): return
                self.taskInitialized.emit('Calculating probability map')
                prob = self.algorithm.predict_proba(in_data).max(axis=1)

            # Reconstruct the result
                if self.isInterruptionRequested(): return
                self.taskInitialized.emit('Reconstructing mineral map')
                shape = self.classifier.map_shape
                pred = pred.reshape(shape)
                prob = prob.reshape(shape)

            # Send the workFinished signal with success
                self.workFinished.emit((pred, prob), True)

            except Exception as e:
            # Send the workFinished signal with error
                self.workFinished.emit((e,), False)

            finally:
            # Reset parameters for safety measures
                self.reset_classifier()



class KNearestNeighbors(_RoiBasedClassifier):
    def __init__(self, inmaps, roimap, n_neigh, weights, n_jobs=None,
                 **kwargs):
        super(KNearestNeighbors, self).__init__(inmaps, roimap, **kwargs)

        self.n_neigh = n_neigh
        self.weights = weights
        self.n_jobs = n_jobs
        self.algorithm = KNeighborsClassifier(self.n_neigh, self.weights,
                                              n_jobs = self.n_jobs)






    # def classify(self):
    #     knc = KNeighborsClassifier(self.n_neigh, self.weights) # !!!  check for more params

    # # "Train" the KNN classifier
    #     x_train, y_train, input_data = self.getTrainingData()
    #     knc.fit(x_train, y_train)

    # # Classify unknown data and calculate probability score
    #     pred = knc.predict(input_data)
    #     prob = knc.predict_proba(input_data).max(axis=1)

    # # Reconstruct the result
    #     pred = pred.reshape(self.map_shape)
    #     prob = prob.reshape(self.map_shape)
    #     return pred, prob








def getNetworkArchitecture(network_id, in_feat, out_cls, seed=None):
    if network_id == 'Softmax Regression':
        network = SoftMaxRegressor(in_feat, out_cls, seed)
    else:
        network = None

    return network


def doMapsFit(maps):
    fit = all([m.shape == maps[0].shape for m in maps])
    return fit


def mergeMaps(maps, masked=False): # !!! change name to smt like prepareMapsforClassifier
# Raise error if maps do not share the same shape
    if not doMapsFit(maps):
        raise ValueError('Input maps do not share the same size.')
# If maps are masked, make them 1D with compressed() otherwise use flatten()
    if masked:
        merged_maps = np.vstack([m.compressed() for m in maps]).T
    else:
        merged_maps = np.vstack([m.flatten() for m in maps]).T

    return merged_maps


# def KNN(data, x_train, y_train, n_neigh, wgts):
# # Build KNN classifier
#     KNC = KNeighborsClassifier(n_neighbors=n_neigh, weights=wgts)
#     KNC.fit(x_train, y_train)
# # Predict data classification
#     pred = KNC.predict(data)
#     prob = KNC.predict_proba(data).max(axis=1)
#     return pred, prob

def K_Means(data, k, seed):
# Build K-Means classifier
    kmeans = KMeans(n_clusters=k, random_state=seed)
    kmeans.fit(data)
# Cluster the data
    pred = kmeans.predict(data)
    dist = kmeans.transform(data).min(axis=1)
    prob = 1 - dist/dist.max()
    return pred, prob

def silhouette_metric(data, pred, type):
    if type == 'avg':
        return silhouette_score(data, pred, metric='euclidean')
    elif type == 'all':
        return silhouette_samples(data, pred, metric='euclidean')
    else:
        raise NameError(f'{type} is not a valid silhouette score type.')

def CHIscore(data, pred):
    return calinski_harabasz_score(data, pred)

def DBIscore(data, pred):
    return davies_bouldin_score(data, pred)


def array2Tensor(arr, dtype='float64'):
    return torch.Tensor(arr.astype(dtype))

def norm_data(data, mean=None, std=None, return_standards=True,
              engine='pytorch'):
    if engine == 'pytorch':
        assert type(data) == torch.Tensor
    mean = data.mean(0) if mean==None else mean
    std = data.std(0) if std==None else std
    data_norm = (data - mean)/std
    if return_standards:
        return (data_norm, mean, std)
    else:
        return data_norm

def map2Polynomial(arr, degree, return_n_features=False):
    kernel = PolynomialFeatures(degree, include_bias=False)
    out = kernel.fit_transform(arr)
# workaround to know how many in_feat there will be after polynomial mapping
    if return_n_features:
        out = (out, kernel.n_output_features_)
    return out

def cuda_available():
    return torch.cuda.is_available()

# def saveModel(var_dict, path, log_path=False, extendedLog=False):
#     torch.save(var_dict, path)
#     if log_path:
#         saveModelLog(var_dict, log_path, extendedLog)

# def saveModelLog(var_dict, log_path, extendedLog=False):
#     with open(log_path, 'w') as log:
#         for k, v in var_dict.items():
#             if not extendedLog:
#                 if k in ('accuracy_list', 'loss_list', 'standards',
#                           'optim_state_dict', 'model_state_dict'):
#                     continue
#             log.write(f'{k.upper()}\n{repr(v)}\n\n\n')

# def loadModel(self, path):
#     #include all variables (ex pickle) https://pytorch.org/tutorials/beginner/saving_loading_models.html
#     var_dict = torch.load(path)
#     return var_dict

# def missingVariables(var_dict):
#     required = set(['algm_name', 'loss_name', 'optim_name',
#                     'optim_state_dict', 'model_state_dict',
#                     'regressorDegree', 'standards', 'parentModel_path',
#                     'GT_dataset_path', 'TVT_rateos', 'balancing_info',
#                     'device', 'seed', 'epochs', 'lr', 'wd', 'mtm',
#                     'accuracy_list', 'loss_list', 'accuracy', 'loss',
#                     'F1_scores', 'ordered_Xfeat', 'Y_dict'])
#     featured = set(var_dict.keys())
#     # returns an iterable with lenght 0 if all required variables are present
#     return required - featured

def embed_modelParameters(old_state_dict, new_model):
# Iterate through parent model state dict to extract weights and biases
    parent_weights, parent_biases = None, None
    for k, v in old_state_dict.items():
        if 'weight' in k:
            parent_weights = v
        elif 'bias' in k:
            parent_biases = v
# Check that parent weights and biases were identified correctly
    assert parent_weights != None and parent_biases != None
# Replace parent weights and biases into new model by using
# the tensor size of parent model output (parent_output size)
    parent_output_size = parent_biases.size(0)
# The weights tensor is a ixJ (2D) tensor (i=n_class, j=n_Xfeat)
    new_model.get_weight().data[:parent_output_size, :] = parent_weights
# The biases tensor is a j (1D) tensor (j=n_Xfeat)
    new_model.get_bias().data[:parent_output_size] = parent_biases



def splitTrainValidTest(X, Y, trRateo, vdRateo=None, seed=None, axis=0):
    '''
    Split X features and Y targets into train, (validation) and test sets.

    Parameters
    ----------
    X : numpy.ndarray
        X features.
    Y : numpy.ndarray
        Y targets.
    trRateo : FLOAT
        Percentage of data to be included in training set.
    vdRateo : FLOAT or None, optional
        Percentage of data to be included in validation set. If None, no
        validation set will be produced. The default is None.
    seed : INT, optional
        Random seed for reproducibility. The default is None.
    axis : INT, optional
        The array axis along which to split. The default is 0.

    Returns
    -------
    X_split : LIST
        Train, (validation), test sets of X features.
    Y_split : LIST
        Train, (validation), test sets of Y targets.

    '''
    lenDS = X.shape[axis]
# Apply permutations to dataset
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    X = X[idx]
    Y = Y[idx]
# Define split index/indices
    split_idx = [int(lenDS * trRateo)]
    if vdRateo is not None:
        split_idx.append(int(lenDS * (trRateo + vdRateo)))
# Split X and Y into training, (validation) & test sets
    X_split = np.split(X, split_idx, axis=axis)
    Y_split = np.split(Y, split_idx, axis=axis)

    return X_split, Y_split


def splitXFeat_YTarget(dataset, split_idx=-1, xtype='int64', ytype='str', spliton='cols'):
    '''
    Split X features from Y targets from given dataset.

    Parameters
    ----------
    dataset : numpy.ndarray
        The ground-truth dataset.
    split_idx : INT, optional
        The splitting index. The default is -1.
    xtype : numpy.dtype -> STR, optional
        X features dtype. The default is 'int64'.
    ytype : numpy.dtype -> STR, optional
        Y targets dtype. The default is 'str'.
    spliton : STR, optional
        Whether to split dataset along columns ('cols') or rows ('rows').
        The default is 'cols'.

    Returns
    -------
    X : numpy.ndarray
        X features.
    Y : numpy.ndarray
        Y targets.

    '''
    if spliton == 'rows':
        dataset = dataset.T
    X = dataset[:, :split_idx].astype(xtype)
    Y = dataset[:, split_idx].astype(ytype)
    return X, Y


def balance_TrainSet(X, Y, strategy, over_sampl='SMOTE', under_sampl=None,
                     kOS=5, mOS=10, nUS=3, seed=None, progressBar=False):
    '''
    A function to balance training datasets with over-sample and/or under-sample algorithms.

    Parameters
    ----------
    X : array-like
        Input training features from unbalanced dataset.
    Y : array-like
        Output training labels from unbalanced dataset.
    strategy : int OR str OR dict
        Data balancing strategy.
         - int: classes will be resampled to this specific value.
         - str: a predefined function to resample to a computed value. Accepted keywords are ['Min', 'Max', 'Mean', 'Median'].
         - dict: a dictionary indicating the exact value of resampling for each class.
    over_sampl : str, optional
        Select over-sampling algorithm. Set None to not allow over-sampling. The default is 'SMOTE'.
    under_sampl : str, optional
        Select under-sampling algorithm. Set None to not allow under-sampling. The default is None.
    kOS : int, optional
        Number of k-neighbours to consider in over-sampling algorithms. The default is 5.
    mOS : int, optional
        Number of m-neighbours to consider in over-sampling algorithms. The default is 10.
    nUS : int, optional
        Number of n-neighbours to consider in under-sampling algorithms. The default is 3.
    seed : int, optional
        Control the randomization of the algorithms. The default is None.

    Returns
    -------
    X_bal : array-like
        Balanced training features.
    Y_bal : array-like
        Balanced training labels.
    args : dictionary
        Convenient dictionary storing all the balancing session information.

    '''
    args = {'Strategy':strategy, 'OS':over_sampl, 'US':under_sampl,
            'n-neigh_US':nUS, 'k-neigh_OS':kOS, 'm-neigh_OS':mOS, 'Seed':seed}
    unq, cnt = np.unique(Y, return_counts=True)

    if type(strategy) == int:
        num = [strategy] * len(cnt)

    elif type(strategy) == str:
        if strategy == 'Min':
            num = [cnt.min()] * len(cnt)
        elif strategy == 'Max':
            num = [cnt.max()] * len(cnt)
        elif strategy == 'Mean':
            num = [int(np.mean(cnt))] * len(cnt)
        elif strategy == 'Median':
            num = [int(np.median(cnt))] * len(cnt)
        else:
            raise KeyError(f'Unknown function: {strategy}')
        args['Strategy'] = num[0]

    elif type(strategy) == dict:
        num = list(strategy.values())

    else:
        raise TypeError('sample_num parameter can only be of type int, str or'\
                       f' dict, not{type(strategy)}')

    # Update strategy in args dictionary
    args['Strategy'] = dict(zip(unq, [f'{c} -> {n}' for c, n in zip(cnt, num)]))

    # Splitting over-sampling and under-sampling strategies
    OS_strat, US_strat = {}, {}
    for u, c, n in zip(unq, cnt, num):
        if n >= c:
            OS_strat[u] = n
        else:
            US_strat[u] = n




    # U N D E R - S A M P L I N G
    if under_sampl is not None:
        import imblearn.under_sampling as US
        warn = 'Warning: {0} under-sampling algorithm ignores the sample'\
               ' numbers required by the user'

        # Setting under-sampling algorithm
        if under_sampl == 'RandUS':
            US_method = US.RandomUnderSampler(sampling_strategy = US_strat,
                                              random_state = seed)
        elif under_sampl == 'NearMiss':
            US_method = US.NearMiss(sampling_strategy = US_strat,
                                    n_neighbors = nUS,
                                    n_jobs = -2)
        elif under_sampl == 'ClusterCentroids':
            US_method = US.ClusterCentroids(sampling_strategy = US_strat,
                                            random_state = seed)
        elif under_sampl == 'TomekLinks':
            US_method = US.TomekLinks(sampling_strategy = list(US_strat.keys()),
                                      n_jobs = -2)
            # print(warn.format('TomekLinks'))
        elif under_sampl in ('ENN-all', 'ENN-mode'):
            US_method = US.EditedNearestNeighbours(sampling_strategy = list(US_strat.keys()),
                                                   n_neighbors = nUS,
                                                   kind_sel = under_sampl.split('-')[-1],
                                                   n_jobs = -2)
            # print(warn.format('EditedNearestNeighbours'))
        elif under_sampl in ('NCR-all', 'NCR-mode'):
            US_method = US.NeighbourhoodCleaningRule(sampling_strategy = list(US_strat.keys()),
                                                     n_neighbors = nUS,
                                                     kind_sel = under_sampl.split('-')[-1],
                                                     n_jobs = -2)
            # print(warn.format('NeighbourhoodCleaningRule'))
        else:
            accepted_US_methods = ['RandUS', 'NearMiss', 'ClusterCentroids', 'TomekLinks',
                                   'ENN-all', 'ENN-mode', 'NCR-all', 'NCR-mode']
            raise KeyError(f'Unknown under-sampling algorithm: {under_sampl}.'\
                            ' under_sampl keyword must be one of the following:'\
                           f' {sorted(accepted_US_methods)}. More info at'\
                            ' https://imbalanced-learn.org/stable/index.html')

        X, Y = US_method.fit_resample(X, Y)
        if progressBar:
            progressBar.setValue(progressBar.value() + 1)

    # O V E R - S A M P L I N G
    if over_sampl is not None:
        import imblearn.over_sampling as OS

        # Setting over-sampling algorithm
        if over_sampl == 'SMOTE':
            OS_method = OS.SMOTE(sampling_strategy = OS_strat,
                                 random_state = seed,
                                 k_neighbors = kOS,
                                 n_jobs = -2)
        elif over_sampl == 'BorderlineSMOTE':
            OS_method = OS.BorderlineSMOTE(sampling_strategy = OS_strat,
                                           random_state = seed,
                                           k_neighbors = kOS,
                                           m_neighbors = mOS,
                                           n_jobs = -2)
        elif over_sampl == 'SVMSMOTE':
            OS_method = OS.SVMSMOTE(sampling_strategy = OS_strat,
                                    random_state = seed,
                                    k_neighbors = kOS,
                                    m_neighbors = mOS,
                                    n_jobs = -2)
        elif over_sampl == 'ADASYN':
            OS_method = OS.ADASYN(sampling_strategy = OS_strat,
                                  random_state = seed,
                                  n_neighbors = kOS,
                                  n_jobs = -2)

        else:
            accepted_OS_methods = ['SMOTE', 'BorderlineSMOTE', 'SVMSMOTE', 'ADASYN']
            raise KeyError(f'Unknown over-sampling algorithm: {over_sampl}.'\
                            ' over_sampl keyword must be one of the following:'\
                           f' {sorted(accepted_OS_methods)}. More info at'\
                            ' https://imbalanced-learn.org/stable/index.html')

        X, Y = OS_method.fit_resample(X, Y)
        if progressBar:
            progressBar.setValue(progressBar.value() + 1)

    np.random.seed(seed)
    perm = np.random.permutation(len(X))
    X_bal, Y_bal = X[perm], Y[perm]

    return X_bal, Y_bal, args



# !!! deprecated. Use getNetworkArchitecture() instead
# def getModel(model_key, in_features, out_classes, seed=None):
#     if model_key == 'Softmax Regression':
#         model = SoftMaxRegressor(in_features, out_classes, seed)
#     else: raise KeyError(f'Invalid model {model_key}')
#     return model

# def getLoss(loss_key):
#     loss_dict = {'Cross-Entropy': torch.nn.CrossEntropyLoss()}
#     return loss_dict[loss_key]

#!!! deprecated. moved to ModelBasedClassifier class
# def applyModel(modelVars, arr):
# # Get variables from model
#     algm = modelVars['algm_name']
#     in_feat = len(modelVars['ordered_Xfeat'])
#     out_class = len(modelVars['Y_dict'])
#     X_mean, X_std = modelVars['standards']
#     state_dict = modelVars['model_state_dict']
#     regrDegree = modelVars['regressorDegree']
# # Map features from linear to polynomial if required
#     if regrDegree > 1:
#         arr = map2Polynomial(arr, regrDegree)
#         in_feat = arr.shape[1]
# # Standardize data
#     data = torch.tensor(arr)
#     data_norm = norm_data(data, X_mean, X_std, return_standards=False)
# # Initialize model
#     model = getModel(algm, in_feat, out_class)
#     model.load_state_dict(state_dict)
# # Predict results
#     prob, lbl = model.predict(data_norm.float())
#     return (prob, lbl)

def getOptimizer(optimizer_key, model, lr, mtm, wd):
    if optimizer_key == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=mtm, weight_decay=wd)
    else: raise KeyError(f'Invalid optimizer {optimizer_key}')
    return optimizer

def accuracy(true, preds):
    return accuracy_score(true, preds)

def confusionMatrix(true, preds, IDs, percent=False):
    cm = confusion_matrix(true, preds, labels=IDs)
    if percent:
        perc = np.zeros(cm.shape)   # pre-init a zeroes matrix
        sums = cm.sum(1).reshape(-1, 1)
        # replace the pre-init matrix with the percentages (only where sum is not 0 to avoid NaNs)
        np.divide(cm, sums, out=perc, where = sums!=0)
        cm = np.round(perc, 2)
    return cm

def f1score(true, preds, avg):
    f1 = f1_score(true, preds, average=avg)
    return f1

