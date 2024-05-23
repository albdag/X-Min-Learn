# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:06:51 2021

@author: albdag
"""

import numpy as np
import sklearn.cluster
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing
import torch

from _base import InputMap, RoiMap, InputMapStack
import conv_functions as CF
import threads




class SoftMaxRegressor(torch.nn.Module):
    # Some part of this class should probably be enhanced once the Model Learner tool is overhauled
    '''
    Softmax Regressor Neural Network.
    '''
    def __init__(self, in_features: int, out_classes: int, seed:int|None=None):
        '''
        Constructor.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_classes : int
            Number of required output classes.
        seed : int | None, optional
            Random seed. If None it will be automatically generated. The 
            default is None.

        '''
        super(SoftMaxRegressor, self).__init__()

    # Set random seed
        if seed is not None:
            torch.random.manual_seed(seed)

    # Set main attributes
        self._name = 'Softmax Regression'
        self._loss = 'Cross-Entropy loss'
        self.linear = torch.nn.Linear(in_features, out_classes) 
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = torch.nn.CrossEntropyLoss() 


    def get_weight(self):
        '''
        Return model weights.

        Returns
        -------
        Tensor
            Model weights.

        '''
        return self.linear.weight


    def get_bias(self):
        '''
        Return model bias.

        Returns
        -------
        Tensor
            Model bias.

        '''
        return self.linear.bias


    def forward(self, x: torch.Tensor):
        '''
        Defines how to process the input x.

        Parameters
        ----------
        x : Tensor
            Input data.

        Returns
        -------
        scores : Tensor
            Linear scores (logits).

        '''
        scores = self.linear(x)
        return scores


    def predict(self, x: torch.Tensor):
        '''
        Defines how to apply the softmax function to the logits (z) obtained
        from self.linear.

        Parameters
        ----------
        x : torch.Tensor
            Input data.

        Returns
        -------
        probs : Tensor
            Probability scores.
        classes : Tensor
            Output classes.

        '''
        self.eval()
        z = self.linear(x)
        probs, classes = self.softmax(z).max(1)
        return probs, classes


    def learn(self, X_train: torch.Tensor, Y_train: torch.Tensor, 
              X_test: torch.Tensor, Y_test: torch.Tensor, 
              optimizer: torch.optim.Optimizer, device: str):
        
        '''
        Defines a single learning iteration.

        Parameters
        ----------
        X_train : Tensor
            Train set feature data.   
        Y_train : Tensor
            Train set label data.
        X_test : Tensor
            Test set feature data.   
        Y_test : Tensor
            Test set label data.
        optimizer : Optimizer
            Optimizer.
        device : str
            Where to compute the learning iteration. Can be either 'cpu' or 
            'cuda'.

        Returns
        -------
        train_loss : Tensor
            Train set loss.
        test_loss : Tensor
            Test set loss.
        train_preds : Tensor
            Predictions on train set.
        test_prefs : Tensor
            Predictions on test set

        '''
    # Predict train data and compute train loss
        self.train()
        out = self.forward(X_train.to(device))
        l = self.loss(out, Y_train.long().to(device))
 
        l.backward()
        optimizer.step()
        optimizer.zero_grad() # Should this be called before l.backward()?

        train_loss = l.cpu().detach().numpy().item()
        train_preds = out.max(1)[1].cpu()

    # Predict test data and compute test loss
        self.eval()
        with torch.set_grad_enabled(False):

            out = self.forward(X_test.to(device))
            l = self.loss(out, Y_test.long().to(device))

            test_loss = l.cpu().detach().numpy().item()
            test_preds = out.max(1)[1].cpu()

        return (train_loss, test_loss, train_preds, test_preds)



class EagerModel():
    '''
    A class that allows to manage eager ML models and their variables.
    '''
    def __init__(self, variables: dict, model_path: str|None=None):
        '''
        Constructor.

        Parameters
        ----------
        variables : dict
            Model variables dictionary.
        model_path : str | None, optional
            Model filepath. The default is None.

        '''
    # Set main attributes
        self.variables = variables
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
    def load(cls, model_path: str):
        '''
        Load model from filepath.

        Parameters
        ----------
        model_path : str
            Model filepath

        Returns
        -------
        EagerModel
            A new instance of EagerModel.

        '''
        variables = torch.load(model_path)
        return cls(variables, model_path)

    @property
    def algorithm(self):
        return self.variables.get('algm_name')

    @property
    def inFeat(self):
        return self.variables.get('ordered_Xfeat')

    @property
    def encoder(self):
        return self.variables.get('Y_dict')

    @property
    def xMean(self):
        return self.variables.get('standards')[0]

    @property
    def xStd(self):
        return self.variables.get('standards')[1]

    @property
    def stateDict(self):
        return self.variables.get('model_state_dict')

    @property
    def regrDegree(self):
        return self.variables.get('regressorDegree')

    def getTrainedModel(self):
        '''
        Return the trained ML model.

        Returns
        -------
        model
            Trained ML network architecture.

        '''
        infeat = self.inFeatPoly() if self.regrDegree > 1 else len(self.inFeat)
        outcls = len(self.encoder)
        model = getNetworkArchitecture(self.algorithm, infeat, outcls)

        if model is not None:
            model.load_state_dict(self.stateDict)
            return model

# find a more elegant solution (equation?)
    def inFeatPoly(self):
        infeat = len(self.inFeat)
        dummy_arr = np.arange(infeat).reshape(1, infeat)
        _, n_poly_feat = map2Polynomial(dummy_arr, self.regrDegree,
                                        return_n_features = True)
        return n_poly_feat

    def save(self, outpath: str, log_path: str|None=None, extended_log=False):
        '''
        Save model variables to file.

        Parameters
        ----------
        outpath : str
            Output filepath.
        log_path : str | None, optional
            Log file output. If None, no log file will be compiled. The default
            is None.
        extended_log : bool, optional
            Whether the log file should include extended information. This is
            ignored if <log_path> is None. The default is False.

        '''
        torch.save(self.variables, outpath)
        if log_path is not None:
            self.saveLog(log_path, extended_log)


    def saveLog(self, outpath: str, extended=False):
        '''
        Save model log file.

        Parameters
        ----------
        outpath : str
            Log filepath.
        extended : bool, optional
            Whether the log file should include extended information. The 
            default is False.

        '''
        with open(outpath, 'w') as log:
            for k, v in self.variables.items():
                if not extended and k in self._extended_vrb:
                    continue
                log.write(f'{k.upper()}\n{repr(v)}\n\n\n')


    def generateLogPath(self, path: str|None):
        '''
        Automatically generate a log filepath from the given path.

        Parameters
        ----------
        path : str | None
            Reference path. Usually is the model variables path.

        Returns
        -------
        logpath : str
            Generated log filepath.

        '''
        if path is None: return
        logpath = CF.extend_filename(path, '_log', ext='.txt')
        return logpath


    def missingVariables(self):
        '''
        Check if any model variable is missing.

        Returns
        -------
        missing : set
            Missing variables.

        '''
        required_vrb = self._base_vrb + self._extended_vrb
        missing = set(required_vrb) - set(self.variables.keys())
        return missing




class ModelBasedClassifier():
    '''
    Base class for all ML model based classifiers.
    '''
    def __init__(self, model: EagerModel, input_stack: InputMapStack):
        '''
        Constuctor.

        Parameters
        ----------
        model : EagerModel
            ML supervised model.
        input_stack : InputMapStack
            Stack of input maps.

        '''
    # Set main attributes
        self.name = CF.path2filename(model.filepath)
        self.model = model
        self.input_stack = input_stack
        self.map_shape = input_stack.maps_shape
        self.algorithm = model.getTrainedModel()
        self.classification_steps = 4
        self.thread = threads.ModelBasedClassificationThread()

    def preProcessFeatureData(self):
        '''
        Perform several pre-processing operations on input feature data.

        Returns
        -------
        feat_data : Tensor
            Pre-processed input data.

        '''
    # Get a 2D features array suited for classification (n_pix x n_maps)
        feat_data = self.input_stack.get_feature_array()

    # Map features from linear to polynomial if required
        if (regr_degree := self.model.regrDegree) > 1:
            feat_data = map2Polynomial(feat_data, regr_degree)

    # Standardize data
        feat_data = torch.tensor(feat_data)
        feat_data = norm_data(feat_data, self.model.xMean, self.model.xStd,
                              return_standards=False)

        return feat_data


    def encodeLabels(self, array: np.ndarray, dtype='int16'):
        '''
        Encode labels from text names to class IDs.

        Parameters
        ----------
        array : np.ndarray
            Labels array.
        dtype : str, optional
            Encoded array dtype. The default is 'int16'.

        Returns
        -------
        res : ndarray
            Encoded labels array.

        '''
        res = np.copy(array)
        for k, v in self.model.encoder.items(): res[array==k] = v
        return res.astype(dtype)

    def decodeLabels(self, array: np.ndarray, dtype='U8'):
        '''
        Decode labels from class IDs to text names.

        Parameters
        ----------
        array : np.ndarray
            Labels array.
        dtype : str, optional
            Decoded array dtype. The default is 'U8'.

        Returns
        -------
        res : ndarray
            Decoded labels array.
            
        '''
        res = np.copy(array).astype(dtype)
        for k, v in self.model.encoder.items(): res[array==v] = k
        return res


    def startThreadedClassification(self):
        '''
        Launch the classification external thread.

        '''
        self.thread.set_classifier(self)
        self.thread.start()



class RoiBasedClassifier():
    '''
    Base class for all ROI based classifiers.
    '''
    def __init__(self, input_stack: InputMapStack, roimap: RoiMap, n_jobs=1, 
                 pixel_proximity=False):
        '''
        Constructor.

        Parameters
        ----------
        input_stack : InputMapStack
            Stack of input maps.
        roimap : RoiMap
            ROI map containing training data.
        n_jobs : int, optional
            Number of parallel CPU threads. If -1, all processors are used. The
            default is 1.
        pixel_proximity : bool, optional
            Add x,y pixel indices maps to input features. The default is False.

        '''
    # Set main attributes
        self.input_stack = input_stack
        self.roimap = roimap
        self.map_shape = input_stack.maps_shape
        self.n_jobs = n_jobs
        self.proximity = pixel_proximity
        self.algorithm = None  # to be reimplemented in each child class
        self.classification_steps = 5
        self.thread = threads.RoiBasedClassificationThread()


    def getCoordMaps(self):
        '''
        Return x, y pixel indices (coordinates) maps.

        Returns
        -------
        coord_maps : list of ndarrays
            X, Y coordinates maps.

        '''
        shape = self.map_shape
        xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        coord_maps = [InputMap(xx), InputMap(yy)]
        return coord_maps


    def preProcessFeatureData(self):
        '''
        Perform several pre-processing operations on input feature data.

        Returns
        -------
        feat_data : ndarray
            Pre-processed input data.

        '''
    # Include pixel coordinate maps if required
        if self.proximity:
            self.input_stack.add_maps(self.getCoordMaps())

    # Get a 2D features array suited for classification (n_pix x n_maps)
        feat_data = self.input_stack.get_feature_array()

    # Normalize the data
        feat_data = norm_data(feat_data, return_standards=False, 
                              engine='numpy')

        return feat_data


    def preProcessRoiData(self):
        '''
        Perform several pre-processing operations on ROI training data.

        Returns
        -------
        roidata : ndarray
            Pre-processed ROI training data.

        '''
        roidata = self.roimap.map

    # Apply mask if required and flatten/compress the map array
        if self.input_stack.mask is None:
            roidata = roidata.flatten()
        else:
            mask_arr = self.input_stack.mask.mask
            roidata = np.ma.masked_where(mask_arr, roidata).compressed()

        return roidata


    def getTrainingData(self, return_full_input=True):
        '''
        Return training data, splitted into features (X) and labels (Y).

        Parameters
        ----------
        return_full_input : bool, optional
            Whether to also return the full input data. The default is True.

        Returns
        -------
        tr_data : list of ndarrays
            Training data, splitted in feature and label data. The list also
            includes the full input feature data if <return_full_input> is 
            True.

        '''
    # Get pre-processed X (feature) and Y (label) data
        x = self.preProcessFeatureData()
        y = self.preProcessRoiData()

    # Extract indices where Y (ROI map data) is actually populated with labels
        labeled_indices = (y != self.roimap._ND).nonzero()[0]

    # Use the indices to extract training data from both features and labels
        x_train = x[labeled_indices, :]
        y_train = y[labeled_indices]

        tr_data = [x_train, y_train]
        if return_full_input: tr_data.append(x)
        return tr_data


    def startThreadedClassification(self):
        '''
        Launch the classification external thread.

        '''
        self.thread.set_classifier(self)
        self.thread.start()


class KNearestNeighbors(RoiBasedClassifier):
    '''
    K-Nearest Neighbors classifier.
    '''
    def __init__(self, input_stack: InputMapStack, roimap: RoiMap, neigh: int,
                 weights: str, **kwargs):
        '''
        Constructor.

        Parameters
        ----------
        input_stack : InputMapStack
            Stack of input maps.
        roimap : RoiMap
            ROI map containing training data.
        neigh : int
            Number of neighbors.
        weights : str
            Neighbors weight function. Can be either 'uniform' or 'distance'.
        **kwargs
            Parent class arguments (see RoiBasedClassifier class).

        '''
        super(KNearestNeighbors, self).__init__(input_stack, roimap, **kwargs)
    
    # Set main attributes
        self.name = 'KNN'
        self.n_neigh = neigh
        self.weights = weights
        kw = {'weights': self.weights, 'n_jobs': self.n_jobs}
        self.algorithm = sklearn.neighbors.KNeighborsClassifier(self.n_neigh,
                                                                **kw)



class UnsupervisedClassifier():
    '''
    Base class for all unsupervised classifiers.
    '''
    def __init__(self, input_stack: InputMapStack, seed: int, n_jobs=1, 
                 pixel_proximity=False):
        '''
        Constructor.

        Parameters
        ----------
        input_stack : InputMapStack
            Stack of input maps.
        seed : int
            Deterministic random state.
        n_jobs : int, optional
            Number of parallel CPU threads. If -1, all processors are used. The
            default is 1.
        pixel_proximity : bool, optional
            Add x,y pixel indices maps to input features. The default is False.

        '''
    # Set main attributes
        self.input_stack = input_stack
        self.map_shape = input_stack.maps_shape
        self.seed = seed
        self.n_jobs = n_jobs
        self.proximity = pixel_proximity
        self.algorithm = None  # to be reimplemented in each child class
        self.classification_steps = 4
        self.thread = threads.UnsupervisedClassificationThread()


    def getCoordMaps(self):
        '''
        Return x, y pixel indices (coordinates) maps.

        Returns
        -------
        coord_maps : list of ndarrays
            X, Y coordinates maps.

        '''
        shape = self.map_shape
        xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        coord_maps = [InputMap(xx), InputMap(yy)]
        return coord_maps
    

    def preProcessFeatureData(self):
        '''
        Perform several pre-processing operations on input feature data.

        Returns
        -------
        feat_data : ndarray
            Pre-processed input data.

        '''
    # Include pixel coordinate maps if required
        if self.proximity:
            self.input_stack.add_maps(self.getCoordMaps())

    # Get a 2D features array suited for classification (n_pix x n_maps)
        feat_data = self.input_stack.get_feature_array()

    # Normalize the data
        feat_data = norm_data(feat_data, return_standards=False, 
                              engine='numpy')

        return feat_data


    def startThreadedClassification(self):
        '''
        Launch the classification external thread.

        '''
        self.thread.set_classifier(self)
        self.thread.start()



class KMeans(UnsupervisedClassifier):
    '''
    K-Means classifier.
    '''
    def __init__(self, input_stack: InputMapStack, n_clust: int, seed: int,
                 **kwargs):
        '''
        Constructor.

        Parameters
        ----------
        input_stack : InputMapStack
            Stack of input maps.
        n_clust : int
            Number of clusters.
        seed : int
            Deterministic random state.

        '''
        super(KMeans, self).__init__(input_stack, seed, **kwargs)
    
    # Set main attributes
        self.name = 'K-Means'
        self.n_clust = n_clust
        self.algorithm = sklearn.cluster.KMeans(self.n_clust, 
                                                random_state=self.seed)





def getNetworkArchitecture(network_id, in_feat, out_cls, seed=None):
    if network_id == 'Softmax Regression':
        network = SoftMaxRegressor(in_feat, out_cls, seed)
    else:
        network = None

    return network


# !!! deprecated. Moved to _base.InpuMapStack class
# def doMapsFit(maps:list):
#     '''
#     Check if all maps in list have same shape.

#     Parameters
#     ----------
#     maps : list
#         List of maps. Accepts lists of ndarray, InputMap, MineralMap, RoiMap

#     Returns
#     -------
#     fit : bool
#         Whether the maps share the same shape.

#     '''
#     fit = all([m.shape == maps[0].shape for m in maps])
#     return fit

# !!! deprecated. Moved to _base.InpuMapStack class
# def mergeMaps(maps, masked=False): 
# # Raise error if maps do not share the same shape
#     if not doMapsFit(maps):
#         raise ValueError('Input maps do not share the same size.')
# # If maps are masked, make them 1D with compressed() otherwise use flatten()
#     if masked:
#         merged_maps = np.vstack([m.compressed() for m in maps]).T
#     else:
#         merged_maps = np.vstack([m.flatten() for m in maps]).T

#     return merged_maps




def CHIscore(data, pred):
    return sklearn.metrics.calinski_harabasz_score(data, pred)

def DBIscore(data, pred):
    return sklearn.metrics.davies_bouldin_score(data, pred)


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
    kern = sklearn.preprocessing.PolynomialFeatures(degree, include_bias=False)
    out = kern.fit_transform(arr)
# workaround to know how many in_feat there will be after polynomial mapping
    if return_n_features:
        out = (out, kern.n_output_features_)
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


def splitXFeat_YTarget(dataset, split_idx=-1, xtype='int64', ytype='str', 
                       spliton='cols'):
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
    x : numpy.ndarray
        X features.
    y : numpy.ndarray
        Y targets.

    '''
    if spliton == 'rows':
        dataset = dataset.T
    x = dataset[:, :split_idx].astype(xtype)
    y = dataset[:, split_idx].astype(ytype)
    return x, y


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

def confusionMatrix(true, preds, IDs, percent=False):
    cm = sklearn.metrics.confusion_matrix(true, preds, labels=IDs)
    if percent:
        perc = np.zeros(cm.shape)   # pre-init a zeroes matrix
        sums = cm.sum(1).reshape(-1, 1)
        # replace the pre-init matrix with the percentages (only where sum is not 0 to avoid NaNs)
        np.divide(cm, sums, out=perc, where = sums!=0)
        cm = np.round(perc, 2)
    return cm

def f1score(true, preds, avg):
    f1 = sklearn.metrics.f1_score(true, preds, average=avg)
    return f1

