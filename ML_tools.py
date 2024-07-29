# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:06:51 2021

@author: albdag
"""

import math
from typing import Any

import numpy as np
import pandas as pd
import sklearn.cluster
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing
import torch

from _base import InputMap, InputMapStack, MineralMap, RoiMap
import conv_functions as CF
import threads



class GroundTruthDataset():
    '''
    A base class to process and manipulate ground truth datasets.
    '''
    def __init__(self, dataframe: pd.DataFrame, filepath: str|None = None):
        '''
        Constructor.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Ground truth dataset.
        filepath : str | None, optional
            Filepath to the stored dataset file. The default is None.

        '''
    # Main attributes
        self.filepath = filepath
        self.dataframe = dataframe

    # X features and Y targets (as labels) and their encoder -> dict(lbl: id)
        self.features, self.targets = None, None
        self.encoder = {}

    # Original data, untouched by possible balancing operations on train set
        self.orig_subsets_ratios = (None, None, None)
        self.orig_x_train = None
        self.orig_y_train = None
        self.orig_train_counter = {}

    # Current X features and Y targets (as IDs) split into subsets
        self.x_train, self.x_valid, self.x_test = None, None, None
        self.y_train, self.y_valid, self.y_test = None, None, None

    # Current subsets counters -> dict(target_labels: count)
        self.train_counter = {}
        self.valid_counter = {}
        self.test_counter = {}

    
    def reset(self):
        '''
        Reset dataset derived attributes.

        '''
        self.features, self.targets = None, None
        self.encoder.clear()
        self.orig_subsets_ratios = (None, None, None)
        self.orig_x_train = None
        self.orig_y_train = None
        self.orig_train_counter.clear()
        self.x_train, self.x_valid, self.x_test = None, None, None
        self.y_train, self.y_valid, self.y_test = None, None, None
        self.train_counter.clear()
        self.valid_counter.clear()
        self.test_counter.clear()


    def are_subsets_split(self):
        '''
        Check if dataset has been split already into train, validation and test
        subsets.

        Returns
        -------
        bool
            Whether dataset has been split.

        '''
        return any(self.orig_subsets_ratios)


    def split_features_targets(self, split_idx=-1, xtype='int64', ytype='U8',
                               spliton='columns'):
        '''
        Split X features from Y targets.

        Parameters
        ----------
        split_idx : int, optional
            The splitting index. The default is -1.
        xtype : str, optional
            X features dtype. The default is 'int64'.
        ytype : str, optional
            Y targets dtype. The default is 'U8'.
        spliton : str, optional
            Whether to split dataset along columns ('columns') or rows ('rows').
            The default is 'columns'.

        Returns
        -------
        self.features : np.ndarray
            X features.
        self.targets : np.ndarray
            Y targets as labels.

        '''
        dataset = self.dataframe.to_numpy()
        if spliton == 'rows':
            dataset = dataset.T

        self.features = dataset[:, :split_idx].astype(xtype)
        self.targets = dataset[:, split_idx].astype(ytype)
        return self.features, self.targets
    

    def update_encoder(self, parent_encoder={}):
        '''
        Refresh the encoder. Can inherit from a parent encoder.

        Parameters
        ----------
        parent_encoder : dict, optional
            Existent encoder. The default is {}.

        Raises
        ------
        ValueError
            Features and targets must be split before calling this function.

        '''
        if self.targets is None:
            raise ValueError('Features and targets are not split yet.')
        
        self.encoder = parent_encoder
        for u in np.unique(self.targets):
            if not u in self.encoder.keys():
                self.encoder[u] = len(self.encoder)


    def split_subsets(self, train_ratio: float, valid_ratio: float, 
                      test_ratio: float, seed: int|None=None, axis=0):
        '''
        Split X features and Y targets into train, (validation) and test sets.

        Parameters
        ----------
        train_ratio : float
            Percentage of data to be included in training set.
        validation_ratio : float optional
            Percentage of data to be included in validation set. 
        test_ratio : float
            Percentage of data to be included in test set.
        seed : int, optional
            Random seed for reproducibility. The default is None.
        axis : int, optional
            The array axis along which to split. The default is 0.

        Returns
        -------
        feat_split : list
            Train, validation and test sets of X features.
        targ_split : list
            Train, validation and test sets of Y targets.

        Raises
        ------
        ValueError
            Encoder must be populated before calling this function.

        '''
        if not self.encoder:
            raise ValueError('Dataset encoder is empty.')
        
    # Store the current ratios as the original subsets ratios
        self.orig_subsets_ratios = (train_ratio, valid_ratio, test_ratio)

    # Encode targets' labels as IDs    
        targets_ids = np.empty(self.targets.shape, dtype='int16')
        for lbl, id in self.encoder.items():
            targets_ids[self.targets==lbl] = id

    # Apply permutations to dataset
        n_instances = self.features.shape[axis]
        perm_idx = np.random.default_rng(seed).permutation(n_instances)
        feat_perm = self.features[perm_idx]
        targ_perm = targets_ids[perm_idx]

    # Define split indices
        idx = [int(n_instances * train_ratio), 
               int(n_instances * (train_ratio + valid_ratio))]

    # Split permuted features and targets into train, validation and test sets
        feat_split = np.split(feat_perm, idx, axis)
        targ_split = np.split(targ_perm, idx, axis)
        self.x_train, self.x_valid, self.x_test = feat_split
        self.y_train, self.y_valid, self.y_test = targ_split

    # Populate train, validaton and test counters
        self.update_counters()

    # Store original x_train, y_train and train_counter
        self.orig_x_train = self.x_train.copy()
        self.orig_y_train = self.y_train.copy()
        self.orig_train_counter = self.train_counter.copy()

        return feat_split, targ_split
    

    def update_counters(self):
        '''
        Refresh train, validation and test counters.

        Raises
        ------
        ValueError
            Dataset must be split into subsets before calling this function.

        '''
        if not self.are_subsets_split():
            raise ValueError('Dataset is not split in subsets.')
        
        for lbl, id in self.encoder.items():
            self.train_counter[lbl] = np.count_nonzero(self.y_train==id)
            self.valid_counter[lbl] = np.count_nonzero(self.y_valid==id)
            self.test_counter[lbl] = np.count_nonzero(self.y_test==id)


    def apply_balancing(self, balanced_x: np.ndarray, balanced_y: np.ndarray):
        '''
        Update train subset and its counter after having performed balancing
        operations.

        Parameters
        ----------
        balanced_x : np.ndarray
            Balanced features.
        balanced_y : np.ndarray
            Balanced targets ad IDs.

        Raises
        ------
        ValueError
            Dataset must be split into subsets before calling this function.

        '''
        if not self.are_subsets_split():
            raise ValueError('Dataset is not split in subsets.')

        self.x_train, self.y_train = balanced_x, balanced_y
        for lbl, id in self.encoder.items():
            self.train_counter[lbl] = np.count_nonzero(self.y_train==id)


    def discard_balancing(self):
        '''
        Discard all balancing operations on train set by restoring original
        train subset and its counter.

        Raises
        ------
        ValueError
            Dataset must be split into subsets before calling this function.

        '''
        if not self.are_subsets_split():
            raise ValueError('Dataset is not split in subsets.')
        
        self.x_train = self.orig_x_train.copy()
        self.y_train = self.orig_y_train.copy()
        self.train_counter = self.orig_train_counter.copy()


    def counters(self):
        '''
        Return all counters.

        Returns
        -------
        tuple[dict]
            All counters.

        '''
        return (self.train_counter, self.valid_counter, self.test_counter)
    

    def current_subsets_ratios(self):
        '''
        Return the current train, validation and test ratios.

        Returns
        -------
        tuple[float]
            Current subset ratios.

        '''
        tr_size, vd_size, ts_size = [sum(c.values()) for c in self.counters()]
        tot_size = tr_size + vd_size + ts_size
        tr_ratio = tr_size / tot_size
        vd_ratio = vd_size / tot_size
        ts_ratio = ts_size / tot_size
        return (tr_ratio, vd_ratio, ts_ratio)
    

    def features_names(self):
        '''
        Return the names of input features. This function can be called when
        dataset has not been split yet, but assumes that the feature data is
        stored in all but the last column of the dataframe.

        Returns
        -------
        list
            Input features names
            
        '''
        return self.dataframe.columns.to_list()[:-1]
    

    def targets_names(self):
        '''
        Return the names of output target classes. This function can be called
        when dataset has not been split yet, but assumes that the target data
        is stored in the last column of the dataframe.

        Returns
        -------
        list
            List of sorted classes names.

        '''
        return sorted(self.dataframe.iloc[:, -1].unique().tolist())
    


class NeuralNetwork(torch.nn.Module):
    '''
    Base class for neural network architectures developed in X-Min Learn.
    '''
    def __init__(self, name='_name', loss='_loss', seed: int|None = None):
        '''
        Constructor.

        Parameters
        ----------
        name : str, optional
            Network name. To be defined by each child. The default is '_name'.
        loss : str, optional
            Network loss. To be defined by each child. The default is '_loss'.
        seed : int | None, optional
            Random seed. If None it will be automatically generated. The 
            default is None.

        '''
        super(NeuralNetwork, self).__init__()

    # Set main attributes
        self._name = name
        self._loss = loss

    # Set random seed
        if seed is not None:
            torch.random.manual_seed(seed)


    def get_weight(self):
        '''
        Return model weights. To be reimplemented in each child class.

        Returns
        -------
        Tensor
            Model weights.

        '''
        return torch.Tensor([])


    def get_bias(self):
        '''
        Return model bias. To be reimplemented in each child class.

        Returns
        -------
        Tensor
            Model bias.

        '''
        return torch.Tensor([])


# This function is depracated, since the same result can be achieved with 
# load_state_dict() function. This function may be useful when only certain
# weights / biases need to be retrieved from parent model. A practical example
# could be if one wants to add new layers/network to an already existent model.
# However this option is not viable in X-Min Learn and there are no current 
# plans to enable it.
    def embedParentNetworkStateDict(self, parent_state_dict: dict):
    # Get parent weights and biases
        parent_weights, parent_biases = None, None
        for k, v in parent_state_dict.items():
            if 'weight' in k:
                parent_weights = v
            elif 'bias' in k:
                parent_biases = v
    
    # Check that parent weights and biases were identified correctly
        if parent_weights is None or parent_biases is None:
            raise ValueError('Cannot parse parent weights and/or biases.')

    # Replace parent weights and biases into new network by using the tensor 
    # size of parent network output (parent_output size)
        parent_output_size = parent_biases.size(0)

    # The weights tensor is a ixJ (2D) tensor (i=n_class, j=n_Xfeat)
        self.get_weight().data[:parent_output_size, :] = parent_weights

    # The biases tensor is a j (1D) tensor (j=n_Xfeat)
        self.get_bias().data[:parent_output_size] = parent_biases



class SoftMaxRegressor(NeuralNetwork):
    '''
    Softmax Regressor Neural Network.
    '''
    def __init__(self, in_features: int, out_classes: int, **kwargs):
        '''
        Constructor.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_classes : int
            Number of required output classes.
        **kwargs
            Parent class keyword arguments (see NeuralNetwork for details).

        '''
        super(SoftMaxRegressor, self).__init__(name='Softmax Regression', 
                                               loss='Cross-Entropy loss',
                                               **kwargs)

    # Set main attributes
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
        from forward function.

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
        z = self.forward(x)
        probs, classes = self.softmax(z).max(1)
        return probs, classes


    def learn(self, X_train: torch.Tensor, Y_train: torch.Tensor, # maybe could be moved to parent class
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
        train_loss : float
            Train set loss.
        test_loss : float
            Test set loss.
        train_preds : Tensor
            Predictions on train set.
        test_prefs : Tensor
            Predictions on test set

        '''
    # Predict train data and compute train loss
        self.train()
        optimizer.zero_grad()

        out = self.forward(X_train.to(device))
        l = self.loss(out, Y_train.long().to(device))
 
        l.backward()
        optimizer.step()
        # optimizer.zero_grad() # Should this be called before l.backward()?

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
    A base class to manipulate and process eager ML models and their variables.
    '''
    _base_vrb = ['algm_name', 
                 'loss_name', 
                 'optim_name',
                 'ordered_Xfeat', 
                 'Y_dict', 
                 'device', 
                 'seed',
                 'parentModel_path', 
                 'GT_dataset_path', 
                 'TVT_rateos',
                 'balancing_info', 
                 'regressorDegree', 
                 'epochs', 
                 'lr',
                 'wd', 
                 'mtm', 
                 'accuracy', 
                 'loss', 
                 'F1_scores'
                 ]
    
    _extended_vrb = ['accuracy_list', 
                     'loss_list',
                     'standards',
                     'optim_state_dict', 
                     'model_state_dict'
                     ]

    def __init__(self, variables: dict, model_path: str|None = None):
        '''
        Constructor.

        Parameters
        ----------
        variables : dict
            Model variables dictionary.
        model_path : str | None, optional
            Model filepath. The default is None.

        Raise
        -----
        ValueError
            Raised when model is missing variables.

        '''
    # Set main attributes
        self.variables = variables
        if len(mv := self.missingVariables()):
            raise ValueError(f'Missing model variables: {mv}')

        self.filepath = model_path


    @classmethod
    def initialize_empty(cls):
        '''
        Build new model with empty variables.

        Returns
        -------
        EagerModel
            A new instance of EagerModel.

        '''
        keys = cls._base_vrb + cls._extended_vrb
        variables = dict().fromkeys(keys)
        return cls(variables)


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
    def optimizer(self):
        return self.variables.get('optim_name')
    
    @property
    def hyperparameters(self):
        lr = self.variables.get('lr')
        wd = self.variables.get('wd')
        mtm = self.variables.get('mtm')
        epochs = self.variables.get('epochs')
        return (lr, wd, mtm, epochs)

    @property
    def features(self):
        return self.variables.get('ordered_Xfeat')
    
    @property
    def targets(self):
        return list(self.encoder.keys())

    @property
    def encoder(self):
        return self.variables.get('Y_dict')

    @property
    def x_mean(self):
        return self.variables.get('standards')[0]

    @property
    def x_stdev(self):
        return self.variables.get('standards')[1]

    @property
    def network_state_dict(self):
        return self.variables.get('model_state_dict')

    @property
    def poly_degree(self):
        return self.variables.get('regressorDegree')
    
    @property
    def seed(self):
        return self.variables.get('seed')
    

    def get_network_architecture(self):
        '''
        Return the neural network architecture used in this model.

        Returns
        -------
        network : NeuralNetwork
            The neural network associated with this model.

        '''
        infeat = self.true_features_number()
        outcls = len(self.encoder)

        if self.algorithm == 'Softmax Regression':
            network = SoftMaxRegressor(infeat, outcls, seed=self.seed)
        else:
            network = None

        return network


    def get_trained_network(self):
        '''
        Return the trained neural network.

        Returns
        -------
        network : NeuralNetwork
            Trained neural network.

        '''
        network = self.get_network_architecture()
        if network is not None:
        # Use map_location arg if a pc tries to use a model trained on gpu but 
        # has no available gpu. However, maybe this is not even a problem.
            # network.load_state_dict(self.network_state_dict, map_location=torch.device('cpu'))
            network.load_state_dict(self.network_state_dict)
            return network
        

    def get_optimizer(self, network: NeuralNetwork):
        '''
        Return the optimizer used to train a network.

        Parameters
        ----------
        network : NeuralNetwork
            Trained neural network.

        Returns
        -------
        optimizer : torch optimizer
            The adopted optimizer.

        '''
        lr, wd, mtm, _ = self.hyperparameters

        if self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(network.parameters(), lr, momentum=mtm,
                                        weight_decay=wd)
        else: 
            optimizer = None

        return optimizer
        

    def true_features_number(self):
        '''
        Calculate the true number of input features, including the result of
        possible polynomial feature mapping.

        Returns
        -------
        int
            Number of input features.

        '''
        n = len(self.features)
        d = self.poly_degree
        return sum(math.comb(n + i - 1, i) for i in range(1, d + 1))


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
        self.filepath = outpath
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



class _ClassifierBase():
    '''
    Base class for all types of mineral classifiers.
    '''
    def __init__(self, type_: str, name: str, classification_steps: int,
                 thread: threads.MineralClassificationThread,
                 input_stack: InputMapStack):
        '''
        Constructor.

        Parameters
        ----------
        type_ : str
            Description of the type of classifier.
        name : str
            Descriptive name of the classifier.
        classification_steps : int
            Number of classification steps required.
        thread : threads.MineralClassificationThread
            Employed mineral classification worker.
        input_stack : InputMapStack
            Stack of input maps.

        '''        
    # Set main attributes
        self.type = type_
        self.name = name
        self.classification_steps = classification_steps
        self.thread = thread
        self.algorithm = None # to be reimplemented in each subclass
        self.input_stack = input_stack
        self.map_shape = input_stack.maps_shape


    @property
    def classification_pipeline(self):
        '''
        Defines the classification pipeline of this classifier. To reimplement
        in each child.

        Returns
        -------
        tuple
            Classification pipeline.

        '''
        return ()
        
        
    def startThreadedClassification(self):
        '''
        Launch the classification external thread (worker).

        '''
        self.thread.set_pipeline(self.classification_pipeline)
        self.thread.start()



class ModelBasedClassifier(_ClassifierBase):
    '''
    Base class for all ML model based classifiers.
    '''
    def __init__(self, input_stack: InputMapStack, model: EagerModel):
        '''
        Constuctor.

        Parameters
        ----------
        input_stack : InputMapStack
            Stack of input maps.
        model : EagerModel
            ML supervised model.

        '''
    # Set main attributes
        kwargs = {'type_': 'Model-based',
                  'name': CF.path2filename(model.filepath),
                  'classification_steps': 4,
                  'thread': threads.ModelBasedClassificationThread(),
                  'input_stack': input_stack
                  }
        super(ModelBasedClassifier, self).__init__(**kwargs)
        self.model = model
        self.algorithm = model.get_trained_network()


    @property
    def classification_pipeline(self):
        '''
        Classification pipeline for all model-based classifiers.

        Returns
        -------
        tuple
            Pipeline

        '''
        f1 = self.preProcessFeatureData
        f2 = self.predict
        f3 = self.postProcessOutputData
        return (f1, f2, f3)
    

    def classify(self):
        '''
        Run entire not-threaded classification process.

        Returns
        -------
        pred: ndarray
            Predictions.
        prob: ndarray
            Probability scores.

        '''
        feat_data = self.preProcessFeatureData()
        prob, pred = self.predict(feat_data)
        prob, pred = self.postProcessOutputData(prob, pred)
        return pred, prob


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

    # Map features from linear to polynomial (get original data if degree=1)
        feat_data = map_polinomial_features(feat_data, self.model.poly_degree)

    # Standardize data
        feat_data = array2tensor(feat_data, 'float32')
        feat_data = norm_data(feat_data, self.model.x_mean, self.model.x_stdev,
                              return_standards=False)

        return feat_data
    

    def predict(self, feat_data: torch.Tensor):
        '''
        Classify and compute probability scores.

        Parameters
        ----------
        feat_data : Tensor
            Input feature data.

        Returns
        -------
        prob : ndarray
            Probability scores.
        pred : ndarray
            Predictions.

        '''
        prob, pred = self.algorithm.predict(feat_data.float())
        return prob, pred
    

    def postProcessOutputData(self, prob: torch.Tensor, pred: torch.Tensor):
        '''
        Post-process probability scores and predictions for better readability.

        Parameters
        ----------
        prob : torch.Tensor
            Probability scores.
        pred : torch.Tensor
            Predictions.

        Returns
        -------
        prob : ndarray
            Rounded probability scores.
        pred : ndarray
            Decoded predictions.

        '''
        prob = prob.detach().numpy().round(2)
        pred = self.decodeLabels(pred)
        return prob, pred
    

    def encodeLabels(self, array: np.ndarray|torch.Tensor, dtype='int16'):
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



class RoiBasedClassifier(_ClassifierBase):
    '''
    Base class for all ROI based classifiers.
    '''
    def __init__(self, input_stack: InputMapStack, roimap: RoiMap, 
                 algorithm_name: str, n_jobs=1, pixel_proximity=False):
        '''
        Constructor.

        Parameters
        ----------
        input_stack : InputMapStack
            Stack of input maps.
        roimap : RoiMap
            ROI map containing training data.
        algorithm_name : str
            Name of the ROI-based algorithm.
        n_jobs : int, optional
            Number of parallel CPU threads. If -1, all processors are used. The
            default is 1.
        pixel_proximity : bool, optional
            Add x,y pixel indices maps to input features. The default is False.

        '''
    # Set main attributes
        kwargs = {'type_': 'ROI-based',
                  'name': algorithm_name,
                  'classification_steps': 5,
                  'thread': threads.RoiBasedClassificationThread(),
                  'input_stack': input_stack
                  }
        super(RoiBasedClassifier, self).__init__(**kwargs)
        self.roimap = roimap
        self.algorithm = None # to be reimplemented in each subclass
        self.n_jobs = n_jobs
        self.proximity = pixel_proximity

    
    @property
    def classification_pipeline(self):
        '''
        Classification pipeline for all ROI-based classifiers.

        Returns
        -------
        tuple
            Pipeline

        '''
        f1 = self.getTrainingData
        f2 = self.fit
        f3 = self.predict
        f4 = self.computeProbabilityScores
        return (f1, f2, f3, f4)
    

    def classify(self):
        '''
        Run entire not-threaded classification process.

        Returns
        -------
        pred: ndarray
            Predictions.
        prob: ndarray
            Probability scores.

        '''
        x_train, y_train, in_data = self.getTrainingData()
        self.fit(x_train, y_train)
        pred = self.predict(in_data)
        prob = self.computeProbabilityScores(in_data)
        return pred, prob


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
        feat_data = norm_data(feat_data, return_standards=False)

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
    

    def fit(self, x_train: np.ndarray, y_train: np.ndarray):
        '''
        Fit classifier to training data.

        Parameters
        ----------
        x_train : ndarray
            Training feature data. 
        y_train : ndarray
            Training target data.

        '''
        self.algorithm.fit(x_train, y_train)
    

    def predict(self, in_data: np.ndarray):
        '''
        Predict unknown data. This function must always be called after fit.

        Parameters
        ----------
        in_data : ndarray
            Input unknown data.

        Returns
        -------
        pred : ndarray
            Predictions.

        '''
        pred = self.algorithm.predict(in_data)
        return pred
    

    def computeProbabilityScores(self, in_data: np.ndarray):
        '''
        Compute confidence scores.

        Parameters
        ----------
        in_data : ndarray
            Input unknown data.

        Returns
        -------
        prob : ndarray
            Probability scores.
        '''
        prob = self.algorithm.predict_proba(in_data).max(axis=1)
        return prob



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
            Parent class keyword arguments (see RoiBasedClassifier class).

        '''
    # Set main attributes
        super(KNearestNeighbors, self).__init__(input_stack, roimap, 'KNN',
                                                **kwargs)
        self.n_neigh = neigh
        self.weights = weights
        kw = {'weights': weights, 'n_jobs': self.n_jobs}
        self.algorithm = sklearn.neighbors.KNeighborsClassifier(neigh, **kw)



class UnsupervisedClassifier(_ClassifierBase):
    '''
    Base class for all unsupervised classifiers.
    '''
    def __init__(self, input_stack: InputMapStack, seed: int, 
                 algorithm_name: str, n_jobs=1, pixel_proximity=False,
                 sil_score=False, sil_ratio=0.25, chi_score=False, 
                 dbi_score=False):
        '''
        Constructor.

        Parameters
        ----------
        input_stack : InputMapStack
            Stack of input maps.
        seed : int
            Deterministic random state.
        algorithm_name : str
            Name of the unsupervised algorithm.
        n_jobs : int, optional
            Number of parallel CPU threads. If -1, all processors are used. The
            default is 1.
        pixel_proximity : bool, optional
            Add x,y pixel indices maps to input features. The default is False.
        sil_score : bool, optional
            Whether to compute silhouette score after mineral classification. 
            The default is False.
        sil_ratio : float, optional
            Percentage of random data to process for silhouette score. The
            default is 0.25.
        chi_score : bool, optional
            Whether to compute Calinski-Harabasz Index score after mineral 
            classification. The default is False.
        dbi_score : bool, optional
            Whether to compute Davies-Bouldin Index score after mineral 
            classification. The default is False.

        '''
    # Set main attributes
        kwargs = {'type_': 'Unsupervised',
                  'name': algorithm_name,
                  'classification_steps': 8,
                  'thread': threads.UnsupervisedClassificationThread(),
                  'input_stack': input_stack
                  }
        super(UnsupervisedClassifier, self).__init__(**kwargs)
        self.seed = seed
        self.algorithm = None  # to be reimplemented in each child class
        self.n_jobs = n_jobs
        self.proximity = pixel_proximity
    
    # Set clustering score related attributes
        self.do_silhouette_score = sil_score
        self.silhouette_ratio = sil_ratio
        self.do_chi_score = chi_score
        self.do_dbi_score = dbi_score


    @property
    def classification_pipeline(self):
        '''
        Classification pipeline for all unsupervised classifiers.

        Returns
        -------
        tuple
            Pipeline

        '''
        f1 = self.preProcessFeatureData
        f2 = self.fit
        f3 = self.predict
        f4 = self.computeProbabilityScores
        f5 = self.computeSilhouetteScore
        f6 = self.computeChiScore
        f7 = self.computeDbiScore
        return (f1, f2, f3, f4, f5, f6, f7)
    

    def classify(self):
        '''
        Run entire not-threaded classification process. Warning: this function
        returns prediction labels as integer values and not as string like the
        threaded classification.

        Returns
        -------
        pred: ndarray
            Predictions.
        prob: ndarray
            Probability scores.

        '''
        in_data = self.preProcessFeatureData()
        self.fit(in_data)
        pred = self.predict(in_data)
        prob = self.computeProbabilityScores(in_data)
        return pred, prob


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
        feat_data = norm_data(feat_data, return_standards=False)

        return feat_data
    

    def fit(self, in_data: np.ndarray):
        '''
        Fit classifier to input data.

        Parameters
        ----------
        in_data : ndarray
            Input data.

        '''
        self.algorithm.fit(in_data)

    
    def predict(self, in_data: np.ndarray):
        '''
        Cluster input data.

        Parameters
        ----------
        in_data : ndarray
            Input data.

        Returns
        -------
        pred : ndarray
            Clustered data.

        '''
        pred = self.algorithm.predict(in_data)
        return pred
    

    def computeProbabilityScores(self, in_data: np.ndarray):
        '''
        Compute confidence scores.

        Parameters
        ----------
        in_data : ndarray
            Input data.

        Returns
        -------
        prob : ndarray
            Probability scores.

        '''
        dist = self.algorithm.transform(in_data).min(axis=1)
        prob = 1 - dist/dist.max()
        return prob
    

    def computeSilhouetteScore(self, in_data: np.ndarray, pred: np.ndarray):
        '''
        Compute silhouette score. The computation is ignored if the attribute
        'do_silhouette_score' is set to False.

        Parameters
        ----------
        in_data : ndarray
            Input data.
        pred : ndarray
            Clustered data.

        Returns
        -------
        sil_clust : ndarray | None
            Silhouette score by cluster.
        sil_avg : float | None
            Average silhouette score.

        '''
        if self.do_silhouette_score:
        # Define a random data sample of required size
            sample_size = int(self.silhouette_ratio * pred.size)
            rng = np.random.default_rng(self.seed)
            subset_idx = rng.permutation(pred.size)[:sample_size]
            data_slice, pred_slice = in_data[subset_idx, :], pred[subset_idx]
        # Compute silhouette score by cluster
            sil_sam = sklearn.metrics.silhouette_samples(data_slice, pred_slice)
            unq_val = np.unique(pred_slice)
            sil_clust = {u: np.sort(sil_sam[pred_slice == u]) for u in unq_val}
        # Compute average silhouette score
            sil_avg = np.mean(sil_sam)
        
        else:
            sil_clust, sil_avg = None, None

        return sil_clust, sil_avg
    

    def computeChiScore(self, in_data: np.ndarray, pred: np.ndarray):
        '''
        Compute Calinski-Harabasz Index. The computation is ignored if the 
        attribute 'do_chi_score' is set to False.

        Parameters
        ----------
        in_data : ndarray
            Input data.
        pred : ndarray
            Clustered data.

        Returns
        -------
        chi : float | None
            Calinski-Harabasz Index.

        '''
        if self.do_chi_score:
            chi = sklearn.metrics.calinski_harabasz_score(in_data, pred)
        else:
            chi = None
        return  chi
    

    def computeDbiScore(self, in_data: np.ndarray, pred: np.ndarray):
        '''
        Compute Davies-Bouldin Index. The computation is ignored if the 
        attribute 'do_dbi_score' is set to False.

        Parameters
        ----------
        in_data : ndarray
            Input data.
        pred : ndarray
            Clustered data.

        Returns
        -------
        dbi : float | None
            Davies-Bouldin Index.

        '''
        if self.do_dbi_score:
            dbi = sklearn.metrics.davies_bouldin_score(in_data, pred)
        else:
            dbi = None
        return dbi
        

       
class KMeans(UnsupervisedClassifier):
    '''
    K-Means classifier.
    '''
    def __init__(self, input_stack: InputMapStack, seed: int, nclust: int,
                 **kwargs):
        '''
        Constructor.

        Parameters
        ----------
        input_stack : InputMapStack
            Stack of input maps.
        seed : int
            Deterministic random state.
        nclust : int
            Number of clusters.

        '''
    # Set main attributes
        super(KMeans, self).__init__(input_stack, seed, 'K-Means', **kwargs)
        self.n_clust = nclust
        self.algorithm = sklearn.cluster.KMeans(nclust, random_state=self.seed)





def array2tensor(array: np.ndarray, dtype: np.dtype|None = None):
    '''
    Convert 

    Parameters
    ----------
    array : ndarray
        Input numpy array
    dtype : numpy dtype, optional
        Output dtype. If None it is inferred from array. The default is None.

    Returns
    -------
    Tensor
        Output torch Tensor.

    '''
    if dtype:
        array = array.astype(dtype)
    return torch.tensor(array)


def norm_data(data: np.ndarray|torch.Tensor, 
              mean: np.ndarray|torch.Tensor|None = None, 
              stdev: np.ndarray|torch.Tensor|None = None, 
              return_standards=True):
    '''
    Apply standard score data normalization to input array.

    Parameters
    ----------
    data : ndarray | Tensor
        Input data.
    mean : ndarray | Tensor | None, optional
        Mean scores per input feature. If None, it is computed. The default is
        None.
    stdev : ndarray | Tensor | None, optional
        Standard deviation scores per input feature. If None, it is computed. 
        The default is None.
    return_standards : bool, optional
        Whether to return means and standard deviations. The default is True.

    Returns
    -------
    data_norm : ndarray | Tensor 
        Normalized input data
    mean : ndarray | Tensor, optional
        Mean scores per input features.
    stdev : ndarray | Tensor, optional
        Standard deviation scores per input features.

    '''
    mean = data.mean(0) if mean is None else mean
    stdev = data.std(0) if stdev is None else stdev
    data_norm = (data - mean) / stdev
    if return_standards:
        return (data_norm, mean, stdev)
    else:
        return data_norm


def map_polinomial_features(array: np.ndarray, degree: int):
    '''
    Apply polynomial kernel to input features of array.  

    Parameters
    ----------
    array : np.ndarray
        Input array.
    degree : int
        Polynomial degree.

    Returns
    -------
    poly_features : np.ndarray
        Output polynomial features array.
    
    '''
    poly = sklearn.preprocessing.PolynomialFeatures(degree, include_bias=False)
    poly_features = poly.fit_transform(array)
    return poly_features


def cuda_available():
    '''
    Check if a cuda-compatible GPU is available on the local machine.

    Returns
    -------
    bool
        Whether cuda-GPU is available.

    '''
    return torch.cuda.is_available()


def confusion_matrix(true: np.ndarray|torch.Tensor, 
                     preds:np.ndarray|torch.Tensor, ids: list|tuple):
    '''
    Compute confusion matrix.

    Parameters
    ----------
    true : ndarray | Tensor
        True classes.
    preds : ndarray | Tensor
        Predicted classes.
    ids : list | tuple
        List of classes IDs.

    Returns
    -------
    cm : ndarray
        Confusion matrix of shape (n_classes, n_classes).

    '''
    cm = sklearn.metrics.confusion_matrix(true, preds, labels=ids)
    return cm


def f1_score(true: np.ndarray|torch.Tensor, preds: np.ndarray|torch.Tensor, 
             avg: str):
    '''
    Compute average F1 score.

    Parameters
    ----------
    true : ndarray | Tensor
        True classes.
    preds : ndarray | Tensor
        Predicted classes.
    avg : str
        Average type. Must be one of ('micro', 'macro', 'weighted').

    Returns
    -------
    f1 : float
        Average F1 score.

    '''
    f1 = sklearn.metrics.f1_score(true, preds, average=avg)
    return f1


def accuracy_score(true: np.ndarray|torch.Tensor, 
                   preds:np.ndarray|torch.Tensor):
    '''
    Compute accuracy score.

    Parameters
    ----------
    true : ndarray | Tensor
        True classes.
    preds : ndarray | Tensor
        Predicted classes.

    Returns
    -------
    accuracy : float
        Accuracy score.

    '''
    accuracy = sklearn.metrics.accuracy_score(true, preds)
    return accuracy


# def getNetworkArchitecture(network_id, in_feat, out_cls, seed=None): # deprecated. Moved to EagerModel
#     if network_id == 'Softmax Regression':
#         network = SoftMaxRegressor(in_feat, out_cls, seed=seed)
#     else:
#         network = None

#     return network


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

# def embed_modelParameters(old_state_dict, new_model): # deprecated, moved to new class NeuralNetwork
# # Iterate through parent model state dict to extract weights and biases
#     parent_weights, parent_biases = None, None
#     for k, v in old_state_dict.items():
#         if 'weight' in k:
#             parent_weights = v
#         elif 'bias' in k:
#             parent_biases = v
# # Check that parent weights and biases were identified correctly
#     assert parent_weights != None and parent_biases != None
# # Replace parent weights and biases into new model by using
# # the tensor size of parent model output (parent_output size)
#     parent_output_size = parent_biases.size(0)
# # The weights tensor is a ixJ (2D) tensor (i=n_class, j=n_Xfeat)
#     new_model.get_weight().data[:parent_output_size, :] = parent_weights
# # The biases tensor is a j (1D) tensor (j=n_Xfeat)
#     new_model.get_bias().data[:parent_output_size] = parent_biases



# def splitTrainValidTest(X, Y, trRateo, vdRateo=None, seed=None, axis=0): # deprecated. Moved to GroundTruthDataset
#     '''
#     Split X features and Y targets into train, (validation) and test sets.

#     Parameters
#     ----------
#     X : numpy.ndarray
#         X features.
#     Y : numpy.ndarray
#         Y targets.
#     trRateo : FLOAT
#         Percentage of data to be included in training set.
#     vdRateo : FLOAT or None, optional
#         Percentage of data to be included in validation set. If None, no
#         validation set will be produced. The default is None.
#     seed : INT, optional
#         Random seed for reproducibility. The default is None.
#     axis : INT, optional
#         The array axis along which to split. The default is 0.

#     Returns
#     -------
#     X_split : LIST
#         Train, (validation), test sets of X features.
#     Y_split : LIST
#         Train, (validation), test sets of Y targets.

#     '''
#     lenDS = X.shape[axis]
# # Apply permutations to dataset
#     idx = np.random.default_rng(seed).permutation(len(X))
#     X = X[idx]
#     Y = Y[idx]
# # Define split index/indices
#     split_idx = [int(lenDS * trRateo)]
#     if vdRateo is not None:
#         split_idx.append(int(lenDS * (trRateo + vdRateo)))
# # Split X and Y into training, (validation) & test sets
#     X_split = np.split(X, split_idx, axis=axis)
#     Y_split = np.split(Y, split_idx, axis=axis)

#     return X_split, Y_split


# def splitXFeat_YTarget(dataset, split_idx=-1, xtype='int64', ytype='str', 
#                        spliton='cols'): #deprecated moved do GroundTruthDataset
#     '''
#     Split X features from Y targets from given dataset.

#     Parameters
#     ----------
#     dataset : numpy.ndarray
#         The ground-truth dataset.
#     split_idx : INT, optional
#         The splitting index. The default is -1.
#     xtype : numpy.dtype -> STR, optional
#         X features dtype. The default is 'int64'.
#     ytype : numpy.dtype -> STR, optional
#         Y targets dtype. The default is 'str'.
#     spliton : STR, optional
#         Whether to split dataset along columns ('cols') or rows ('rows').
#         The default is 'cols'.

#     Returns
#     -------
#     x : numpy.ndarray
#         X features.
#     y : numpy.ndarray
#         Y targets.

#     '''
#     if spliton == 'rows':
#         dataset = dataset.T
#     x = dataset[:, :split_idx].astype(xtype)
#     y = dataset[:, split_idx].astype(ytype)
#     return x, y


# def balance_TrainSet(X, Y, strategy, over_sampl='SMOTE', under_sampl=None, # deprecated. Moved to separate thread
#                      kOS=5, mOS=10, nUS=3, seed=None, progressBar=False):
#     '''
#     A function to balance training datasets with over-sample and/or under-sample algorithms.

#     Parameters
#     ----------
#     X : array-like
#         Input training features from unbalanced dataset.
#     Y : array-like
#         Output training labels from unbalanced dataset.
#     strategy : int OR str OR dict
#         Data balancing strategy.
#          - int: classes will be resampled to this specific value.
#          - str: a predefined function to resample to a computed value. Accepted keywords are ['Min', 'Max', 'Mean', 'Median'].
#          - dict: a dictionary indicating the exact value of resampling for each class.
#     over_sampl : str, optional
#         Select over-sampling algorithm. Set None to not allow over-sampling. The default is 'SMOTE'.
#     under_sampl : str, optional
#         Select under-sampling algorithm. Set None to not allow under-sampling. The default is None.
#     kOS : int, optional
#         Number of k-neighbours to consider in over-sampling algorithms. The default is 5.
#     mOS : int, optional
#         Number of m-neighbours to consider in over-sampling algorithms. The default is 10.
#     nUS : int, optional
#         Number of n-neighbours to consider in under-sampling algorithms. The default is 3.
#     seed : int, optional
#         Control the randomization of the algorithms. The default is None.

#     Returns
#     -------
#     X_bal : array-like
#         Balanced training features.
#     Y_bal : array-like
#         Balanced training labels.
#     args : dictionary
#         Convenient dictionary storing all the balancing session information.

#     '''
#     args = {'Strategy':strategy, 'OS':over_sampl, 'US':under_sampl,
#             'n-neigh_US':nUS, 'k-neigh_OS':kOS, 'm-neigh_OS':mOS, 'Seed':seed}
#     unq, cnt = np.unique(Y, return_counts=True)

#     if type(strategy) == int:
#         num = [strategy] * len(cnt)

#     elif type(strategy) == str:
#         if strategy == 'Min':
#             num = [cnt.min()] * len(cnt)
#         elif strategy == 'Max':
#             num = [cnt.max()] * len(cnt)
#         elif strategy == 'Mean':
#             num = [int(np.mean(cnt))] * len(cnt)
#         elif strategy == 'Median':
#             num = [int(np.median(cnt))] * len(cnt)
#         else:
#             raise KeyError(f'Unknown function: {strategy}')
#         args['Strategy'] = num[0]

#     elif type(strategy) == dict:
#         num = list(strategy.values())

#     else:
#         raise TypeError('sample_num parameter can only be of type int, str or'\
#                        f' dict, not{type(strategy)}')

#     # Update strategy in args dictionary
#     args['Strategy'] = dict(zip(unq, [f'{c} -> {n}' for c, n in zip(cnt, num)]))

#     # Splitting over-sampling and under-sampling strategies
#     OS_strat, US_strat = {}, {}
#     for u, c, n in zip(unq, cnt, num):
#         if n >= c:
#             OS_strat[u] = n
#         else:
#             US_strat[u] = n




#     # U N D E R - S A M P L I N G
#     if under_sampl is not None:
#         import imblearn.under_sampling as US
#         warn = 'Warning: {0} under-sampling algorithm ignores the sample'\
#                ' numbers required by the user'

#         # Setting under-sampling algorithm
#         if under_sampl == 'RandUS':
#             US_method = US.RandomUnderSampler(sampling_strategy = US_strat,
#                                               random_state = seed)
#         elif under_sampl == 'NearMiss':
#             US_method = US.NearMiss(sampling_strategy = US_strat,
#                                     n_neighbors = nUS,
#                                     n_jobs = -2)
#         elif under_sampl == 'ClusterCentroids':
#             US_method = US.ClusterCentroids(sampling_strategy = US_strat,
#                                             random_state = seed)
#         elif under_sampl == 'TomekLinks':
#             US_method = US.TomekLinks(sampling_strategy = list(US_strat.keys()),
#                                       n_jobs = -2)
#             # print(warn.format('TomekLinks'))
#         elif under_sampl in ('ENN-all', 'ENN-mode'):
#             US_method = US.EditedNearestNeighbours(sampling_strategy = list(US_strat.keys()),
#                                                    n_neighbors = nUS,
#                                                    kind_sel = under_sampl.split('-')[-1],
#                                                    n_jobs = -2)
#             # print(warn.format('EditedNearestNeighbours'))
#         elif under_sampl in ('NCR-all', 'NCR-mode'):
#             US_method = US.NeighbourhoodCleaningRule(sampling_strategy = list(US_strat.keys()),
#                                                      n_neighbors = nUS,
#                                                      kind_sel = under_sampl.split('-')[-1],
#                                                      n_jobs = -2)
#             # print(warn.format('NeighbourhoodCleaningRule'))
#         else:
#             accepted_US_methods = ['RandUS', 'NearMiss', 'ClusterCentroids', 'TomekLinks',
#                                    'ENN-all', 'ENN-mode', 'NCR-all', 'NCR-mode']
#             raise KeyError(f'Unknown under-sampling algorithm: {under_sampl}.'\
#                             ' under_sampl keyword must be one of the following:'\
#                            f' {sorted(accepted_US_methods)}. More info at'\
#                             ' https://imbalanced-learn.org/stable/index.html')

#         X, Y = US_method.fit_resample(X, Y)
#         if progressBar:
#             progressBar.setValue(progressBar.value() + 1)

#     # O V E R - S A M P L I N G
#     if over_sampl is not None:
#         import imblearn.over_sampling as OS

#         # Setting over-sampling algorithm
#         if over_sampl == 'SMOTE':
#             OS_method = OS.SMOTE(sampling_strategy = OS_strat,
#                                  random_state = seed,
#                                  k_neighbors = kOS,
#                                  n_jobs = -2)
#         elif over_sampl == 'BorderlineSMOTE':
#             OS_method = OS.BorderlineSMOTE(sampling_strategy = OS_strat,
#                                            random_state = seed,
#                                            k_neighbors = kOS,
#                                            m_neighbors = mOS,
#                                            n_jobs = -2)
#         elif over_sampl == 'SVMSMOTE':
#             OS_method = OS.SVMSMOTE(sampling_strategy = OS_strat,
#                                     random_state = seed,
#                                     k_neighbors = kOS,
#                                     m_neighbors = mOS,
#                                     n_jobs = -2)
#         elif over_sampl == 'ADASYN':
#             OS_method = OS.ADASYN(sampling_strategy = OS_strat,
#                                   random_state = seed,
#                                   n_neighbors = kOS,
#                                   n_jobs = -2)

#         else:
#             accepted_OS_methods = ['SMOTE', 'BorderlineSMOTE', 'SVMSMOTE', 'ADASYN']
#             raise KeyError(f'Unknown over-sampling algorithm: {over_sampl}.'\
#                             ' over_sampl keyword must be one of the following:'\
#                            f' {sorted(accepted_OS_methods)}. More info at'\
#                             ' https://imbalanced-learn.org/stable/index.html')

#         X, Y = OS_method.fit_resample(X, Y)
#         if progressBar:
#             progressBar.setValue(progressBar.value() + 1)

#     perm = np.random.default_rng(seed).permutation(len(X))
#     X_bal, Y_bal = X[perm], Y[perm]

#     return X_bal, Y_bal, args



# !!! deprecated. Use getNetworkArchitecture() instead
# def getModel(model_key, in_features, out_classes, seed=None):
#     if model_key == 'Softmax Regression':
#         model = SoftMaxRegressor(in_features, out_classes, seed)
#     else: raise KeyError(f'Invalid model {model_key}')
#     return model

# def getLoss(loss_key):
#     loss_dict = {'Cross-Entropy': torch.nn.CrossEntropyLoss()}
#     return loss_dict[loss_key]

#!!! deprecated. moved to EagerModel class
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

# def getOptimizer(optimizer_key, model, lr, mtm, wd): # deprecated. Moved in EagerModel
#     if optimizer_key == 'SGD':
#         optimizer = torch.optim.SGD(model.parameters(), lr,
#                                     momentum=mtm, weight_decay=wd)
#     else: raise KeyError(f'Invalid optimizer {optimizer_key}')
#     return optimizer
