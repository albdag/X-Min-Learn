# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:06:51 2021

@author: albdag
"""
from collections.abc import Callable
import math
import multiprocessing

import numpy as np
from numpy.typing import DTypeLike
import sklearn.cluster
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing
import torch
import torch.utils.data as torch_data

from _base import *
import convenient_functions as cf
import preferences as pref
import threads


class NeuralNetwork(torch.nn.Module):

    def __init__(
        self,
        name: str = '_name',
        loss: str ='_loss',
        seed: int | None = None
    ) -> None:
        '''
        Base class for neural network architectures developed in X-Min Learn.

        Parameters
        ----------
        name : str, optional
            Network name. To be defined by each child. The default is '_name'.
        loss : str, optional
            Network loss. To be defined by each child. The default is '_loss'.
        seed : int or None, optional
            Random seed. If None it will be automatically generated. The 
            default is None.

        '''
        super().__init__()

    # Set main attributes
        self._name = name
        self._loss = loss

    # Set random seed
        torch.random.manual_seed(seed)


    def get_weight(self) -> torch.Tensor:
        '''
        Return model weights. To be reimplemented in each child class.

        Returns
        -------
        torch Tensor
            Model weights.

        '''
        return torch.Tensor([])


    def get_bias(self) -> torch.Tensor:
        '''
        Return model bias. To be reimplemented in each child class.

        Returns
        -------
        torch Tensor
            Model bias.

        '''
        return torch.Tensor([])


# This method is deprecated, since the same result can be achieved through the 
# 'load_state_dict' method. It may be useful when only certain weights / biases
# need to be retrieved from parent model. A practical example could be if one 
# wants to add new layers / network to an already existent model. However this
# option is not viable in X-Min Learn and there are no plans to enable it.
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



class TorchDataset(torch_data.Dataset):

    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        '''
        A custom torch dataset class, useful to properly access ground truth 
        datasets within a torch data loader. For more details, see 'DataLoader'
        class.

        Parameters
        ----------
        features : torch Tensor
            Input features.
        targets : torch Tensor
            Output targets.

        '''
        self.feat = features
        self.targ = targets
    

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        '''
        Dataset batch getter method.

        Parameters
        ----------
        idx : int
            Dataset slice index.

        Returns
        -------
        dict[str, torch.Tensor]
            Requested batch of features and targets.

        '''
        x = self.feat[idx, :]
        y = self.targ[idx]
        return {'features': x, 'target': y}
    

    def __len__(self) -> int:
        '''
        Return the number of entries in the dataset. 

        Returns
        -------
        int
            Dataset length.

        '''
        return len(self.targ)


class DataLoader(torch_data.DataLoader):

    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor, 
        batch_size: int,
        workers: int = 0
    ) -> None:
        '''
        Custom torch data loader, useful to access data during a batch learning
        session.

        Parameters
        ----------
        features : torch Tensor
            Input features. The tensor must NOT be loaded on GPU.
        targets : torch Tensor
            Output targets. The tensor must NOT be loaded on GPU.
        batch_size : int
            Number of entries for each batch of data.
        workers : int, optional
            Number of CPU cores used. If 0, no multiprocessing is performed. 
            The default is 0.

        Raises
        ------
        RuntimeError
            Raised if 'features' and/or 'targets' Tensors are loaded on GPU.

        '''
    # Check that Tensors are not loaded on GPU
        if features.is_cuda or targets.is_cuda:
            raise RuntimeError('"features" and/or "targets" are loaded on GPU.')
        
    # Set main attributes
        self.workers = workers
        self.dataset = TorchDataset(features, targets)
        self.batch_size = batch_size

        super().__init__(self.dataset, batch_size, num_workers=workers, pin_memory=True) 



class SoftMaxRegressor(NeuralNetwork):

    def __init__(self, in_features: int, out_classes: int, **kwargs) -> None:
        '''
        Softmax Regressor Neural Network.

        Parameters
        ----------
        in_features : int
            Number of input features.
        out_classes : int
            Number of required output classes.
        **kwargs
            Parent class keyword arguments (see 'NeuralNetwork' for details).

        '''
        name = 'Softmax Regression'
        loss = 'Cross-Entropy loss'
        super().__init__(name=name, loss=loss, **kwargs)

    # Set main attributes
        self.linear = torch.nn.Linear(in_features, out_classes) 
        self.softmax = torch.nn.Softmax(dim=1)
        self.loss = torch.nn.CrossEntropyLoss() 


    def get_weight(self) -> torch.Tensor:
        '''
        Return model weights.

        Returns
        -------
        torch Tensor
            Model weights.

        '''
        return self.linear.weight


    def get_bias(self) -> torch.Tensor:
        '''
        Return model bias.

        Returns
        -------
        torch Tensor
            Model bias.

        '''
        return self.linear.bias


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Defines how to process the input 'x'.

        Parameters
        ----------
        x : torch Tensor
            Input data.

        Returns
        -------
        scores : torch Tensor
            Linear scores (logits).

        '''
        scores = self.linear(x)
        return scores


    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: 
        '''
        Defines how to apply the softmax method to the logits (z) obtained from
        'forward' method.

        Parameters
        ----------
        x : torch Tensor
            Input data.

        Returns
        -------
        probs : torch Tensor
            Probability scores.
        classes : torch Tensor
            Output classes.

        '''
        self.eval()
        z = self.forward(x)
        probs, classes = self.softmax(z).max(1)
        return probs, classes


    def learn( # maybe could be moved to parent class
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor, 
        x_test: torch.Tensor,
        y_test: torch.Tensor, 
        optimizer: torch.optim.Optimizer,
        device: str
    ) -> tuple[float, float, float, float]:
        '''
        Defines a single learning iteration.

        Parameters
        ----------
        x_train : torch Tensor
            Train set feature data.   
        y_train : torch Tensor
            Train set target data.
        x_test : torch Tensor
            Test set feature data.   
        y_test : torch Tensor
            Test set target data.
        optimizer : torch Optimizer
            Model optimization function.
        device : str
            Where learning iteration is computed. Can be 'cpu' or 'cuda'.

        Returns
        -------
        train_loss : float
            Train set loss.
        test_loss : float
            Test set loss.
        train_acc : float
            Train set accuracy.
        test_acc : float
            Test set accuracy.

        '''
        x_train = x_train.to(device)
        y_train = y_train.long().to(device)
        x_test = x_test.to(device)
        y_test = y_test.long().to(device)

    # Predict train data and compute train loss
        self.train()
        optimizer.zero_grad()

        out = self.forward(x_train)
        loss = self.loss(out, y_train)
        acc = (out.argmax(1) == y_train).float().mean()
 
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        train_acc = acc.item()

    # Predict test data and compute test loss
        self.eval()
        with torch.set_grad_enabled(False):

            out = self.forward(x_test)
            loss = self.loss(out, y_test)
            acc = (out.argmax(1) == y_test).float().mean()

            test_loss = loss.item()
            test_acc = acc.item()

        return (train_loss, test_loss, train_acc, test_acc)


    def batch_learn( # maybe could be moved to parent class
        self,
        train_loader: DataLoader,
        test_loader: DataLoader, 
        optimizer: torch.optim.Optimizer,
        device: str
    ) -> tuple[float, float, float, float]:
        '''
        Defines a single batched learning iteration. 

        Parameters
        ----------
        train_loader : DataLoader
            Train set data loader. See 'DataLoader' class for details.   
        test_loader : DataLoader
            Test set data loader. See 'DataLoader' class for details. 
        optimizer : torch Optimizer
            Model optimization function.
        device : str
            Where learning iteration is computed. Can be 'cpu' or 'cuda'.

        Returns
        -------
        train_loss : float
            Train set loss.
        test_loss : float
            Test set loss.
        train_acc : float
            Train set accuracy.
        test_acc : float
            Test set accuracy.

        '''
        loaders = {'train': train_loader, 'test': test_loader}
        batch_tr_loss, batch_ts_loss = [], []
        batch_tr_acc, batch_ts_acc = [], []

        for mode in ('train', 'test'):
            self.train() if mode == 'train' else self.eval()
            
            with torch.set_grad_enabled(mode == 'train'):
                for batch in loaders[mode]:
                    x = batch['features'].to(device, non_blocking=True)
                    y = batch['target'].long().to(device, non_blocking=True)

                    if mode == 'train':
                        optimizer.zero_grad()

                    out = self.forward(x)
                    loss = self.loss(out, y)
                    # acc = accuracy_score(y, out.max(1)[1])
                    acc = (out.argmax(1) == y).float().mean()

                    if mode == 'train':
                        loss.backward()
                        optimizer.step()
                        batch_tr_loss.append(loss.item())
                        batch_tr_acc.append(acc.item())
                    else:
                        batch_ts_loss.append(loss.item())
                        batch_ts_acc.append(acc.item())

        train_loss = sum(batch_tr_loss) / len(batch_tr_loss)
        test_loss = sum(batch_ts_loss) / len(batch_ts_loss)
        train_accuracy = sum(batch_tr_acc) / len(batch_tr_acc)
        test_accuracy = sum(batch_ts_acc) / len(batch_ts_acc)

        return (train_loss, test_loss, train_accuracy, test_accuracy)



class EagerModel():

# Model versioning is used to keep compatibility with old models
    _current_version = 2 

    _base_vrb = [
        'version',
        'algorithm', 
        'loss', 
        'optimizer',
        'input_features', 
        'class_encoder', 
        'device', 
        'seed',
        'parent_model_path', 
        'dataset_path', 
        'tvt_ratios',
        'balancing_info', 
        'polynomial_degree', 
        'epochs', 
        'learning_rate',
        'weight_decay', 
        'momentum',
        'batch_size', 
        'accuracy', 
        'loss', 
        'f1_scores'
    ]
    _extended_vrb = [
        'accuracy_list', 
        'loss_list',
        'standards',
        'optimizer_state_dict', 
        'model_state_dict'
    ]

    def __init__(
        self,
        variables: dict[str, object],
        model_path: str | None = None
    ) -> None:
        '''
        A base class to manipulate and process eager machine learning models
        and their variables.

        Parameters
        ----------
        variables : dict[str, object]
            Model variables dictionary.
        model_path : str or None, optional
            Model filepath. The default is None.

        Raises
        ------
        ValueError
            Raised if model version is incompatible because the app is outdated.
        KeyError
            Raised if model is missing variables.

        '''
    # Set main attributes
        self.variables = variables
        self.filepath = model_path

    # Check model version compatibility
        version = self.variables.get('version', 0)
        if version < self._current_version:
            self._convert_legacy_model(version, self.filepath)
        elif version > self._current_version:
            raise ValueError(f'This model requires an updated app version.')

    # Check for missing variables
        if len(mv := self.missing_variables()):
            raise KeyError(f'Missing model variables: {mv}')

        
    @classmethod
    def initialize_empty(cls) -> 'EagerModel':
        '''
        Build new model with empty variables.

        Returns
        -------
        EagerModel
            A new instance of EagerModel.

        '''
        keys = cls._base_vrb + cls._extended_vrb
        variables = dict().fromkeys(keys)
        variables['version'] = cls._current_version
        return cls(variables)


    @classmethod
    def load(cls, model_path: str) -> 'EagerModel':
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
    def algorithm(self) -> str:
        '''
        Name of the algorithm (i.e., Neural Network) used to train the model.

        Returns
        -------
        str
            Algorithm name

        '''
        return self.variables.get('algorithm')
    
    @property
    def optimizer(self) -> str:
        '''
        Name of the optimizer used to train the model.

        Returns
        -------
        str
            Optimizer name.

        '''
        return self.variables.get('optimizer')
    
    @property
    def hyperparameters(self) -> tuple[float, float, float, int]:
        '''
        Hyperparameters values used to train the model.

        Returns
        -------
        lr : float
            Learning Rate.
        wd : float
            Weight Decay.
        mtm : float
            Momentum.
        epochs : int
            Number of epochs.

        '''
        lr = self.variables.get('learning_rate')
        wd = self.variables.get('weight_decay')
        mtm = self.variables.get('momentum')
        epochs = self.variables.get('epochs')
        # should also return Batch Size
        return lr, wd, mtm, epochs

    @property
    def features(self) -> list[str]:
        '''
        List of input features required by this model to predict targets.

        Returns
        -------
        list[str]
            List of input features.

        '''
        return self.variables.get('input_features')
    
    @property
    def targets(self) -> list[str]:
        '''
        List of targets that this model can predict.

        Returns
        -------
        list[str]
            List of targets.

        '''
        return list(self.encoder.keys())

    @property
    def encoder(self) -> dict[str, int]:
        '''
        Targets encoder -> {target_label: target_ID}.

        Returns
        -------
        dict[str, int]
            Targets encoder.

        '''
        return self.variables.get('class_encoder')

    @property
    def x_mean(self) -> torch.Tensor:
        '''
        Means used by this model to standardize input features.

        Returns
        -------
        torch Tensor
            Mean Tensor.

        '''
        return self.variables.get('standards')[0]

    @property
    def x_stdev(self) -> torch.Tensor:
        '''
        Standard deviations used by this model to standardize input features.

        Returns
        -------
        torch Tensor
            Standard deviation Tensor.

        '''
        return self.variables.get('standards')[1]

    @property
    def state_dict(self) -> dict[str, torch.Tensor]:
        '''
        Model's state dictionary.

        Returns
        -------
        dict[str, torch.Tensor]
            State dictionary.

        '''
        return self.variables.get('model_state_dict')

    @property
    def poly_degree(self) -> int:
        '''
        Polynomial degree used for input feature mapping.

        Returns
        -------
        int
            Polynomial degree.

        '''
        return self.variables.get('polynomial_degree')
    
    @property
    def seed(self) -> int:
        '''
        Random seed for reproducibility purposes.

        Returns
        -------
        int
            Random seed.
            
        '''
        return self.variables.get('seed')
    

    def _convert_legacy_model(self, version: int, path: str | None = None):
        '''
        Convert old model to latest version. The applied changes depend on the
        version of the old model.

        Parameters
        ----------
        version : int
            Model version.
        path : str or None, optional
            Model filepath. If provided, the model file will be overwritten.
            The default is None.

        '''
    # Apply changes based on model version
        if version < 1:
            self.variables['algorithm'] = self.variables.pop('algm_name')
            self.variables['loss'] = self.variables.pop('loss_name')
            self.variables['optimizer'] = self.variables.pop('optim_name')
            self.variables['input_features'] = self.variables.pop('ordered_Xfeat')
            self.variables['class_encoder'] = self.variables.pop('Y_dict')
            self.variables['parent_model_path'] = self.variables.pop('parentModel_path')
            self.variables['dataset_path'] = self.variables.pop('GT_dataset_path')
            self.variables['tvt_ratios'] = self.variables.pop('TVT_rateos')
            self.variables['polynomial_degree'] = self.variables.pop('regressorDegree')
            self.variables['f1_scores'] = self.variables.pop('F1_scores')

            self.variables['batch_size'] = 0

        if version < 2:
            self.variables['learning_rate'] = self.variables.pop('lr')
            self.variables['weight_decay'] = self.variables.pop('wd')
            self.variables['momentum'] = self.variables.pop('mtm')
            self.variables['optimizer_state_dict'] = self.variables.pop('optim_state_dict')

    # Set updated model version
        self.variables['version'] = self._current_version

    # Reorder variables
        var_order = self._base_vrb + self._extended_vrb
        self.variables = cf.sort_dict_by_list(self.variables, var_order)

    # Save the converted model and its log file if a path is provided
        if path is not None:
            log_path = self.generate_log_path(path)
            extended = pref.get_setting('data/extended_model_log')
            self.save(path, log_path=log_path, extended_log=extended)



    def missing_variables(self) -> set[str]:
        '''
        Return missing model's variables.

        Returns
        -------
        missing : set[str]
            Missing variables.

        '''
        required_vrb = self._base_vrb + self._extended_vrb
        missing = set(required_vrb) - set(self.variables.keys())
        return missing
    

    def get_network_architecture(self) -> NeuralNetwork:
        '''
        Return the neural network architecture used by this model.

        Returns
        -------
        NeuralNetwork
            The neural network architecture.

        Raises
        ------
        ValueError
            Raised if network cannot be identified from model's variables. 

        '''
        network = self.algorithm
        infeat = self.true_features_number()
        outcls = len(self.encoder)
        seed = self.seed

        match network:
            case 'Softmax Regression':
                return SoftMaxRegressor(infeat, outcls, seed=seed)
            case _:
                raise ValueError(f'Invalid "algorithm" variable: {network}')


    def get_trained_network(self) -> NeuralNetwork:
        '''
        Return the neural network architecture trained with model's parameters.
        This method should only be used for model's exploitation (e.g., by 
        model-based classifiers).

        Returns
        -------
        network : NeuralNetwork
            Trained neural network.

        '''
        network = self.get_network_architecture()
    # Use map_location arg if a machine tries to use a model trained on gpu but
    # has no available gpu. However, maybe this is not even a problem.
        # network.load_state_dict(self.state_dict, map_location=torch.device('cpu'))
        network.load_state_dict(self.state_dict)
        return network


    def get_optimizer(self, network: NeuralNetwork) -> torch.optim.Optimizer:
        '''
        Return the optimization function used by this model to train the neural
        network 'network'.

        Parameters
        ----------
        network : NeuralNetwork
            Trained neural network.

        Returns
        -------
        torch Optimizer
            The adopted optimizer.

        Raises
        ------
        ValueError
            Raised if optimizer cannot be identified from model's variables.

        Example
        -------
        variables = {'algorithm': 'Softmax Regression', 'optimizer': 'SGD', ...}
        model = EagerModel(variables)
        network = model.get_network_architecture()
        network.to(model.variables['device']) # optional
        optimizer = model.get_optimizer(network)

        # To update previous model, just load its state dicts:
        network.load_state_dict(model.variables['model_state_dict'])
        optimizer.load_state_dict(model.variables['optimizer_state_dict'])

        '''
        optimizer = self.optimizer
        lr, wd, mtm, _ = self.hyperparameters
        params = network.parameters()

        match optimizer:
            case 'SGD':
                return torch.optim.SGD(params, lr, momentum=mtm, weight_decay=wd)
            case _: 
                raise ValueError(f'Invalid "optimizer" variable: {optimizer}')
        

    def true_features_number(self) -> int:
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


    def save(
        self,
        outpath: str,
        log_path: str | None = None,
        extended_log: bool = False
    ) -> None:
        '''
        Save model variables to file.

        Parameters
        ----------
        outpath : str
            Output filepath.
        log_path : str or None, optional
            Log file output. If None, no log file will be compiled. The default
            is None.
        extended_log : bool, optional
            Whether the log file should include extended information. This is
            ignored if 'log_path' is None. The default is False.

        '''
        torch.save(self.variables, outpath)
        self.filepath = outpath
        if log_path is not None:
            self.save_log(log_path, extended_log)


    def save_log(self, outpath: str, extended: bool = False) -> None:
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
                log.write(f'{k.upper().replace('_', ' ')}\n{repr(v)}\n\n\n')


    def generate_log_path(self, path: str) -> str:
        '''
        Automatically generate a log filepath from the given path.

        Parameters
        ----------
        path : str or None
            Reference path. Usually it is the model's filepath.

        Returns
        -------
        logpath : str
            Generated log filepath.

        '''
        logpath = cf.extend_filename(path, '_log', ext='.txt')
        return logpath



class _ClassifierBase():

    def __init__(
        self,
        kind: str,
        name: str,
        classification_steps: int,
        thread: threads.FixedStepsThread,
        input_stack: InputMapStack
    ) -> None:
        '''
        Base class for all types of mineral classifiers.

        Parameters
        ----------
        kind : str
            Description of the type of classifier.
        name : str
            Descriptive name of the classifier.
        classification_steps : int
            Number of classification steps required.
        thread : FixedStepsThread
            Employed mineral classification worker.
        input_stack : InputMapStack
            Stack of input maps.

        '''        
    # Set main attributes
        self.kind = kind
        self.name = name
        self.classification_steps = classification_steps
        self.thread = thread
        self.algorithm = None # to be reimplemented in each subclass
        self.input_stack = input_stack
        self.map_shape = input_stack.maps_shape


    @property
    def classification_pipeline(self) -> tuple:
        '''
        Defines the classification pipeline of this classifier. To reimplement
        in each child.

        Returns
        -------
        tuple
            Classification pipeline.

        '''
        return ()
        
        
    def startThreadedClassification(self) -> None:
        '''
        Launch the classification external thread (worker).

        '''
        self.thread.set_pipeline(self.classification_pipeline)
        self.thread.start()



class ModelBasedClassifier(_ClassifierBase):

    def __init__(self, input_stack: InputMapStack, model: EagerModel) -> None:
        '''
        Base class for all model-based classifiers.

        Parameters
        ----------
        input_stack : InputMapStack
            Stack of input maps.
        model : EagerModel
            ML supervised model.

        '''
    # Set main attributes
        kwargs = {
            'kind': 'Model-based',
            'name': cf.path2filename(model.filepath),
            'classification_steps': 4,
            'thread': threads.ModelBasedClassificationThread(),
            'input_stack': input_stack
        }
        super().__init__(**kwargs)
        self.model = model
        self.algorithm = model.get_trained_network()


    @property
    def classification_pipeline(self) -> tuple[Callable, Callable, Callable]:
        '''
        Classification pipeline for all model-based classifiers.

        Returns
        -------
        f1 : Callable
            Pre-process feature data.
        f2 : Callable
            Predict targets.
        f3 : Callable
            Post-process target data.

        '''
        f1 = self.preProcessFeatureData
        f2 = self.predict
        f3 = self.postProcessOutputData
        return f1, f2, f3
    

    def classify(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Run entire not-threaded classification process.

        Returns
        -------
        pred: numpy ndarray
            Predictions.
        prob: numpy ndarray
            Probability scores.

        '''
        feat_data = self.preProcessFeatureData()
        prob, pred = self.predict(feat_data)
        prob, pred = self.postProcessOutputData(prob, pred)
        return pred, prob


    def preProcessFeatureData(self) -> torch.Tensor:
        '''
        Perform several pre-processing operations on input feature data.

        Returns
        -------
        feat_data : torch Tensor
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
    

    def predict(self, feat_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Classify and compute probability scores.

        Parameters
        ----------
        feat_data : torch Tensor
            Input feature data.

        Returns
        -------
        prob : torch Tensor
            Probability scores.
        pred : torch Tensor
            Predictions.

        '''
        prob, pred = self.algorithm.predict(feat_data.float())
        return prob, pred
    

    def postProcessOutputData(
        self,
        prob: torch.Tensor,
        pred: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
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
        prob : numpy ndarray
            Rounded probability scores.
        pred : numpy ndarray
            Decoded predictions.

        '''
        prob = prob.detach().numpy().round(2)
        pred = self.decodeLabels(pred)
        return prob, pred
    

    def encodeLabels(
        self,
        array: np.ndarray | torch.Tensor,
        dtype: DTypeLike = 'int16'
    ) -> np.ndarray:
        '''
        Encode labels from text names to class IDs.

        Parameters
        ----------
        array : numpy ndarray or torch Tensor
            Labels array.
        dtype : numpy DTypeLike, optional
            Encoded array dtype. The default is 'int16'.

        Returns
        -------
        res : numpy ndarray
            Encoded labels array.

        '''
        res = np.copy(array)
        for k, v in self.model.encoder.items(): res[array==k] = v
        return res.astype(dtype)


    def decodeLabels(
        self,
        array: np.ndarray | torch.Tensor,
        dtype: DTypeLike = 'U8'
    ) -> np.ndarray:
        '''
        Decode labels from class IDs to text names.

        Parameters
        ----------
        array : numpy ndarray or torch Tensor
            Labels array.
        dtype : numpy DTypeLike, optional
            Decoded array dtype. Only use str-like dtypes. The default is 'U8'.

        Returns
        -------
        res : numpy ndarray
            Decoded labels array.
            
        '''
        res = np.copy(array).astype(dtype)
        for k, v in self.model.encoder.items(): res[array==v] = k
        return res



class RoiBasedClassifier(_ClassifierBase):

    def __init__(
        self,
        input_stack: InputMapStack,
        roimap: RoiMap, 
        algorithm_name: str,
        n_jobs: int = 1,
        pixel_proximity: bool = False
    ) -> None:
        '''
        Base class for all ROI based classifiers.

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
        kwargs = {
            'kind': 'ROI-based',
            'name': algorithm_name,
            'classification_steps': 5,
            'thread': threads.RoiBasedClassificationThread(),
            'input_stack': input_stack
        }
        super().__init__(**kwargs)
        self.roimap = roimap
        self.algorithm = None # to be reimplemented in each subclass
        self.n_jobs = n_jobs
        self.proximity = pixel_proximity

    
    @property
    def classification_pipeline(self) -> tuple[Callable, Callable, Callable, Callable]:
        '''
        Classification pipeline for all ROI-based classifiers.

        Returns
        -------
        f1 : Callable
            Get training data from ROIs.
        f2 : Callable
            Fit ROI-based classifier to training data.
        f3 : Callable
            Predict unlabeled data.
        f4 : Callable
            Compute probability scores.

        '''
        f1 = self.getTrainingData
        f2 = self.fit
        f3 = self.predict
        f4 = self.computeProbabilityScores
        return f1, f2, f3, f4
    

    def classify(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Run entire not-threaded classification process.

        Returns
        -------
        pred: numpy ndarray
            Predictions.
        prob: numpy ndarray
            Probability scores.

        '''
        x_train, y_train, in_data = self.getTrainingData()
        self.fit(x_train, y_train)
        pred = self.predict(in_data)
        prob = self.computeProbabilityScores(in_data)
        return pred, prob


    def getCoordMaps(self) -> list[np.ndarray]:
        '''
        Return X, Y pixel indices (coordinates) maps.

        Returns
        -------
        coord_maps : list[numpy ndarrays]
            X, Y coordinates maps.

        '''
        shape = self.map_shape
        xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        coord_maps = [InputMap(xx), InputMap(yy)]
        return coord_maps


    def preProcessFeatureData(self) -> np.ndarray:
        '''
        Perform several pre-processing operations on input feature data.

        Returns
        -------
        feat_data : numpy ndarray
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


    def preProcessRoiData(self) -> np.ndarray:
        '''
        Perform several pre-processing operations on ROI training data.

        Returns
        -------
        roidata : numpy ndarray
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


    def getTrainingData(self, return_full_input: bool = True) -> list[np.ndarray]:
        '''
        Return training data, splitted into features (X) and labels (Y).

        Parameters
        ----------
        return_full_input : bool, optional
            Whether to also return the full input data. The default is True.

        Returns
        -------
        tr_data : list[ndarray]
            Training data, splitted in feature and label data. The list also
            includes the full input feature data if 'return_full_input' is 
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
    

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        '''
        Fit classifier to training data.

        Parameters
        ----------
        x_train : numpy ndarray
            Training feature data. 
        y_train : numpy ndarray
            Training target data.

        '''
        self.algorithm.fit(x_train, y_train)
    

    def predict(self, in_data: np.ndarray) -> np.ndarray:
        '''
        Predict unknown data. This method must always be called after 'fit'.

        Parameters
        ----------
        in_data : numpy ndarray
            Input unlabeled data.

        Returns
        -------
        pred : numpy ndarray
            Predicted data.

        '''
        pred = self.algorithm.predict(in_data)
        return pred
    

    def computeProbabilityScores(self, in_data: np.ndarray) -> np.ndarray:
        '''
        Compute confidence scores.

        Parameters
        ----------
        in_data : numpy ndarray
            Input unlabeled data.

        Returns
        -------
        prob : numpy ndarray
            Probability scores.

        '''
        prob = self.algorithm.predict_proba(in_data).max(axis=1)
        return prob



class KNearestNeighbors(RoiBasedClassifier):

    def __init__(
        self,
        input_stack: InputMapStack,
        roimap: RoiMap,
        neigh: int,
        weights: str,
        **kwargs
    ) -> None:
        '''
        K-Nearest Neighbors classifier.

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
        super().__init__(input_stack, roimap, 'KNN', **kwargs)
        self.n_neigh = neigh
        self.weights = weights
        kw = {'weights': weights, 'n_jobs': self.n_jobs}
        self.algorithm = sklearn.neighbors.KNeighborsClassifier(neigh, **kw)



class UnsupervisedClassifier(_ClassifierBase):

    def __init__(
        self,
        input_stack: InputMapStack,
        seed: int, 
        algorithm_name: str,
        n_jobs: int = 1,
        pixel_proximity: bool = False,
        sil_score: bool = False,
        sil_ratio: float = 0.25,
        chi_score: bool = False, 
        dbi_score: bool = False
    ) -> None:
        '''
        Base class for all unsupervised classifiers.

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
        kwargs = {
            'kind': 'Unsupervised',
            'name': algorithm_name,
            'classification_steps': 8,
            'thread': threads.UnsupervisedClassificationThread(),
            'input_stack': input_stack
        }
        super().__init__(**kwargs)
        self.seed = seed
        self.algorithm = None  # to be reimplemented in each child class
        self.n_jobs = n_jobs
        self.proximity = pixel_proximity
    
    # Set clustering scores related attributes
        self.do_silhouette_score = sil_score
        self.silhouette_ratio = sil_ratio
        self.do_chi_score = chi_score
        self.do_dbi_score = dbi_score


    @property
    def classification_pipeline(self) -> tuple[
        Callable, Callable, Callable, Callable, Callable, Callable, Callable
    ]:
        '''
        Classification pipeline for all unsupervised classifiers.

        Returns
        -------
        f1 : Callable
            Pre-process input feature data.
        f2 : Callable
            Fit classifier to feature data.
        f3 : Callable
            Cluster data.
        f4 : Callable
            Compute probability scores.
        f5 : Callable
            Compute Silhouette score.
        f6 : Callable
            Compute Calinski-Harabasz Index.
        f7 : Callable
            Compute Davies-Bouldin Index.

        '''
        f1 = self.preProcessFeatureData
        f2 = self.fit
        f3 = self.predict
        f4 = self.computeProbabilityScores
        f5 = self.computeSilhouetteScore
        f6 = self.computeChiScore
        f7 = self.computeDbiScore
        return f1, f2, f3, f4, f5, f6, f7
    

    def classify(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Run entire not-threaded classification process. Warning: this method
        returns prediction labels (clusters) as integer values.

        Returns
        -------
        pred: numpy ndarray
            Predictions, with clusters expressed as integer IDs.
        prob: numpy ndarray
            Probability scores.

        '''
        in_data = self.preProcessFeatureData()
        self.fit(in_data)
        pred = self.predict(in_data)
        prob = self.computeProbabilityScores(in_data)
        return pred, prob


    def getCoordMaps(self) -> list[np.ndarray]:
        '''
        Return X, Y pixel indices (coordinates) maps.

        Returns
        -------
        coord_maps : list[numpy ndarray]
            X, Y coordinates maps.

        '''
        shape = self.map_shape
        xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        coord_maps = [InputMap(xx), InputMap(yy)]
        return coord_maps
    

    def preProcessFeatureData(self) -> np.ndarray:
        '''
        Perform several pre-processing operations on input feature data.

        Returns
        -------
        feat_data : numpy ndarray
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
    

    def fit(self, in_data: np.ndarray) -> None:
        '''
        Fit classifier to input data.

        Parameters
        ----------
        in_data : ndarray
            Input data.

        '''
        self.algorithm.fit(in_data)

    
    def predict(self, in_data: np.ndarray) -> np.ndarray:
        '''
        Cluster input data.

        Parameters
        ----------
        in_data : numpy ndarray
            Input data.

        Returns
        -------
        pred : numpy ndarray
            Clustered data.

        '''
        pred = self.algorithm.predict(in_data)
        return pred
    

    def computeProbabilityScores(self, in_data: np.ndarray) -> np.ndarray:
        '''
        Compute confidence scores.

        Parameters
        ----------
        in_data : ndarray
            Input data.

        Returns
        -------
        prob : numpy ndarray
            Probability scores.

        '''
        dist = self.algorithm.transform(in_data).min(axis=1)
        prob = 1 - dist/dist.max()
        return prob
    

    def computeSilhouetteScore(self, in_data: np.ndarray, pred: np.ndarray) -> (
        tuple[
            dict[int, np.ndarray] | None,
            float | None
        ]
    ):
        '''
        Compute silhouette score. The computation is ignored if the class 
        attribute 'do_silhouette_score' is set to False.

        Parameters
        ----------
        in_data : numpy ndarray
            Input data.
        pred : numpy ndarray
            Clustered data.

        Returns
        -------
        sil_clust : dict[int, numpy ndarray] or None
            Silhouette scores by cluster. Returns None if 'do_silhouette_score'
            is False.
        sil_avg : float or None
            Average silhouette score. Returns None if 'do_silhouette_score' is
            False.

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
    

    def computeChiScore(self, in_data: np.ndarray, pred: np.ndarray) -> float | None:
        '''
        Compute Calinski-Harabasz Index. The computation is ignored if the 
        class attribute 'do_chi_score' is set to False.

        Parameters
        ----------
        in_data : numpy ndarray
            Input data.
        pred : numpy ndarray
            Clustered data.

        Returns
        -------
        chi : float or None
            Calinski-Harabasz Index. Returns None if 'do_chi_score' is False.

        '''
        if self.do_chi_score:
            chi = sklearn.metrics.calinski_harabasz_score(in_data, pred)
        else:
            chi = None
        return  chi
    

    def computeDbiScore(self, in_data: np.ndarray, pred: np.ndarray) -> float | None:
        '''
        Compute Davies-Bouldin Index. The computation is ignored if the class
        attribute 'do_dbi_score' is set to False.

        Parameters
        ----------
        in_data : numpy ndarray
            Input data.
        pred : numpy ndarray
            Clustered data.

        Returns
        -------
        dbi : float or None
            Davies-Bouldin Index. Returns None if 'do_dbi_score' is False.

        '''
        if self.do_dbi_score:
            dbi = sklearn.metrics.davies_bouldin_score(in_data, pred)
        else:
            dbi = None
        return dbi
        

       
class KMeans(UnsupervisedClassifier):

    def __init__(
        self,
        input_stack: InputMapStack,
        seed: int,
        nclust: int,
        **kwargs
    ) -> None:
        '''
        K-Means classifier.

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
        super().__init__(input_stack, seed, 'K-Means', **kwargs)
        self.n_clust = nclust
        self.algorithm = sklearn.cluster.KMeans(nclust, random_state=self.seed)



def array2tensor(array: np.ndarray, dtype: DTypeLike | None = None) -> torch.Tensor:
    '''
    Convert numpy array to torch Tensor.

    Parameters
    ----------
    array : numpy ndarray
        Input numpy array
    dtype : numpy DTypeLike, optional
        Output dtype. If None, it is inferred from array. The default is None.

    Returns
    -------
    torch Tensor
        Output torch Tensor.

    '''
    if dtype: array = array.astype(dtype)
    return torch.tensor(array)


def norm_data(
    data: np.ndarray | torch.Tensor, 
    mean: np.ndarray | torch.Tensor|None = None, 
    stdev: np.ndarray | torch.Tensor|None = None, 
    return_standards: bool = True
) -> (
    np.ndarray | torch.Tensor
    | tuple[
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor
    ]
):
    '''
    Apply standard score data normalization to input array.

    Parameters
    ----------
    data : numpy ndarray or torch Tensor
        Input data.
    mean : numpy ndarray or torch Tensor or None, optional
        Mean scores per input feature. If None, it is computed. The default is
        None.
    stdev : numpy ndarray or torch Tensor or None, optional
        Standard deviation scores per input feature. If None, it is computed. 
        The default is None.
    return_standards : bool, optional
        Whether to return means and standard deviations. The default is True.

    Returns
    -------
    data_norm : numpy ndarray or torch Tensor
        Normalized input data
    mean : numpy ndarray or torch Tensor, optional
        Mean scores per input features. Returned only if 'return_standards' is
        True.
    stdev : numpy ndarray or torch Tensor, optional
        Standard deviation scores per input features. Returned only if 
        'return_standards' is True.

    '''
    mean = data.mean(0) if mean is None else mean
    stdev = data.std(0) if stdev is None else stdev
    data_norm = (data - mean) / stdev
    if return_standards:
        return (data_norm, mean, stdev)
    else:
        return data_norm


def map_polinomial_features(array: np.ndarray, degree: int) -> np.ndarray:
    '''
    Apply polynomial kernel to input features of array.  

    Parameters
    ----------
    array : numpy ndarray
        Input array.
    degree : int
        Polynomial degree.

    Returns
    -------
    poly_features : numpy ndarray
        Output polynomial features array.
    
    '''
    poly = sklearn.preprocessing.PolynomialFeatures(degree, include_bias=False)
    poly_features = poly.fit_transform(array)
    return poly_features


def cuda_available() -> bool:
    '''
    Check if a cuda-compatible GPU is available on the local machine.

    Returns
    -------
    bool
        Whether cuda-GPU is available.

    '''
    return torch.cuda.is_available()


def confusion_matrix(
    true: np.ndarray | torch.Tensor, 
    pred: np.ndarray | torch.Tensor, 
    ids: list[int] | tuple[int, ...]
) -> np.ndarray:
    '''
    Compute confusion matrix.

    Parameters
    ----------
    true : numpy ndarray or torch Tensor
        True classes.
    pred : numpy ndarray or torch Tensor
        Predicted classes.
    ids : list[int] or tuple[int, ...]
        List of classes IDs.

    Returns
    -------
    cm : numpy ndarray
        Confusion matrix of shape (n_classes, n_classes).

    '''
    if isinstance(true, torch.Tensor):
        true = true.cpu().detach().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy()

    cm = sklearn.metrics.confusion_matrix(true, pred, labels=ids)
    return cm


def f1_score(
    true: np.ndarray | torch.Tensor,
    pred: np.ndarray | torch.Tensor, 
    avg: str
) -> float:
    '''
    Compute average F1 score of type 'avg'.

    Parameters
    ----------
    true : numpy ndarray or torch Tensor
        True classes.
    pred : numpy ndarray or torch Tensor
        Predicted classes.
    avg : str
        Average type. Must be one of ('micro', 'macro', 'weighted').

    Returns
    -------
    f1 : float
        Average F1 score.

    '''
    if isinstance(true, torch.Tensor):
        true = true.cpu().detach().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy()

    f1 = sklearn.metrics.f1_score(true, pred, average=avg)
    return f1


def accuracy_score(
    true: np.ndarray | torch.Tensor, 
    pred: np.ndarray | torch.Tensor
) -> float:
    '''
    Compute accuracy score.

    Parameters
    ----------
    true : numpy ndarray or torch Tensor
        True classes.
    pred : numpy ndarray or torch Tensor
        Predicted classes.

    Returns
    -------
    accuracy : float
        Accuracy score.

    '''
    if isinstance(true, torch.Tensor):
        true = true.cpu().detach().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy()

    accuracy = sklearn.metrics.accuracy_score(true, pred)
    return accuracy


def num_cores() -> int:
    '''
    Get the number of CPU cores of the machine.

    Returns
    -------
    int
        Number of cores.

    '''
    return multiprocessing.cpu_count()