# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 13:06:51 2021

@author: albdag
"""

from collections.abc import Callable, Sequence
import json
import math
import multiprocessing
from typing import Any, Literal, Protocol, Self, TypedDict, cast, overload

import numpy as np
import numpy.typing as npt
import sklearn.cluster
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing
import torch
import torch.utils.data as torch_data

from base import *
import dataset_tools as dtools
import convenient_functions as cf
import settings
import threads

# Define a generic TensorLike type, that includes both tensors and numpy arrays
type TensorLike = torch.Tensor | npt.NDArray[Any]

# Set allowed global classes when loading pytorch models 
torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct, # pyright: ignore[reportAttributeAccessIssue]
    np.core.multiarray.scalar,       # pyright: ignore[reportAttributeAccessIssue]
    np.dtype,
    np.ndarray,
    np.dtypes.Int8DType,
    np.dtypes.Int16DType,
    np.dtypes.Int32DType,
    np.dtypes.UInt8DType,
    np.dtypes.UInt16DType,
    np.dtypes.UInt32DType,
    np.dtypes.Float64DType,
    np.dtypes.StrDType
])


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



type Loss = torch.nn.modules.loss._Loss
type Optimizer = torch.optim.Optimizer

class NeuralNetwork(torch.nn.Module):

    def __init__(
        self,
        name: str = '_name',
        loss: Loss = torch.nn.modules.loss._Loss(),
        loss_name: str ='_loss',
        seed: int | None = None
    ) -> None:
        '''
        Base class for neural network architectures developed in X-Min Learn.

        Parameters
        ----------
        name : str, optional
            Network name. To be defined by each child. The default is '_name'.
        loss : Loss, optional
            Network loss function. To be defined by each child. The default is 
            a generic placeholder _Loss().
        loss_name : str, optional
            Network loss name. To be defined by each child. The default is '_loss'.
        seed : int or None, optional
            Random seed. If None it will be automatically generated. The 
            default is None.

        '''
        super().__init__()

    # Set main attributes
        self._name = name
        self._loss_name = loss_name
        self.loss = loss

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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Defines how to process the input 'x'. To be reimplemented in each child 
        class.

        Parameters
        ----------
        x : torch Tensor
            Input data.

        Returns
        -------
        torch Tensor
            Output data.

        '''
        return x


    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Defines how to predict output classes and probabilities. To be 
        reimplemented in each child class.

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
        return x, x


    def learn(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor, 
        x_test: torch.Tensor,
        y_test: torch.Tensor, 
        optimizer: Optimizer,
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
        optimizer : Optimizer
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


    def batch_learn(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader, 
        optimizer: Optimizer,
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
        optimizer : Optimizer
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
        loss = torch.nn.CrossEntropyLoss()
        loss_name = 'Cross-Entropy loss'
        super().__init__(name=name, loss=loss, loss_name=loss_name, **kwargs)

    # Set main attributes
        self.linear = torch.nn.Linear(in_features, out_classes)
        self.softmax = torch.nn.Softmax(dim=1)


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



class _BaseModelVariables(TypedDict):
    algorithm: str
    loss_function: str
    optimizer: str
    input_features: list[str]
    class_encoder: dict[str, int]
    device: str
    seed: int
    parent_model_path: str
    dataset_path: str
    tvt_ratios: tuple[float, float, float]
    balancing_info: list[dtools.BalancingInfo]
    polynomial_degree: int
    epochs: int
    learning_rate: float
    weight_decay: float
    momentum: float
    batch_size: int
    accuracy: tuple[float, float, float]
    loss: tuple[float, float, None]
    f1_scores: tuple[tuple[float, float, float], tuple[float, float, float]]

class _ExtendedModelVariables(TypedDict):
    accuracy_list: tuple[list[float], list[float]]
    loss_list: tuple[list[float], list[float]]
    standards: tuple[torch.Tensor, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
    model_state_dict: dict[str, Any]
    version: int

class ModelVariables(_BaseModelVariables, _ExtendedModelVariables): pass

class EagerModel():

# Model versioning is used to keep compatibility with old models
    _current_version = 3 

    _base_variables = list(_BaseModelVariables.__annotations__.keys())
    _extended_variables = list(_ExtendedModelVariables.__annotations__.keys())
    _all_variables = list(ModelVariables.__annotations__.keys())

    def __init__(
        self,
        variables: ModelVariables,
        model_path: str | None = None
    ) -> None:
        '''
        A base class to manipulate and process eager machine learning models
        and their variables.

        Parameters
        ----------
        variables : ModelVariables
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
        self._filepath = model_path

    # Check model version compatibility
        version = self.variables.get('version', 0)
        if version < self._current_version:
            self._convert_legacy_model(version)
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
            A new instance of EagerModel with empty variables.

        '''
        variables: ModelVariables = {
            'algorithm': '',
            'loss_function': '',
            'optimizer': '',
            'input_features': [],
            'class_encoder': {},
            'device': '',
            'seed': 0,
            'parent_model_path': '',
            'dataset_path': '',
            'tvt_ratios': (0., 0., 0.),
            'balancing_info': [],
            'polynomial_degree': 0,
            'epochs': 0,
            'learning_rate': 0.,
            'weight_decay': 0.,
            'momentum': 0.,
            'batch_size': 0,
            'accuracy': (0., 0., 0.),
            'loss': (0., 0., None),
            'f1_scores': ((0., 0., 0.), (0., 0., 0.)),
            'accuracy_list': ([], []),
            'loss_list': ([], []),
            'standards': (torch.Tensor([]), torch.Tensor([])),
            'optimizer_state_dict': {},
            'model_state_dict': {},
            'version': cls._current_version
        }
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
    def filepath(self) -> str | None:
        '''
        Current model's filepath, if any.

        Returns
        -------
        str | None
            Model filepath.

        '''
        return self._filepath


    @property
    def algorithm(self) -> str:
        '''
        Name of the algorithm (i.e., Neural Network) used to train the model.

        Returns
        -------
        str
            Algorithm name

        '''
        return self.variables['algorithm']


    @property
    def optimizer(self) -> str:
        '''
        Name of the optimizer used to train the model.

        Returns
        -------
        str
            Optimizer name.

        '''
        return self.variables['optimizer']


    @property
    def hyperparameters(self) -> tuple[float, float, float, int, int]:
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
        batch_size : int
            Data batch size.

        '''
        lr = self.variables['learning_rate']
        wd = self.variables['weight_decay']
        mtm = self.variables['momentum']
        epochs = self.variables['epochs']
        batch_size = self.variables['batch_size']
        return lr, wd, mtm, epochs, batch_size


    @property
    def features(self) -> list[str]:
        '''
        List of input features required by this model to predict targets.

        Returns
        -------
        list[str]
            List of input features.

        '''
        return self.variables['input_features']


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
        return self.variables['class_encoder']


    @property
    def x_mean(self) -> torch.Tensor:
        '''
        Means used by this model to standardize input features.

        Returns
        -------
        torch Tensor
            Mean Tensor.

        '''
        return self.variables['standards'][0]


    @property
    def x_stdev(self) -> torch.Tensor:
        '''
        Standard deviations used by this model to standardize input features.

        Returns
        -------
        torch Tensor
            Standard deviation Tensor.

        '''
        return self.variables['standards'][1]


    @property
    def state_dict(self) -> dict[str, torch.Tensor]:
        '''
        Model's state dictionary.

        Returns
        -------
        dict[str, torch.Tensor]
            State dictionary.

        '''
        return self.variables['model_state_dict']


    @property
    def poly_degree(self) -> int:
        '''
        Polynomial degree used for input feature mapping.

        Returns
        -------
        int
            Polynomial degree.

        '''
        return self.variables['polynomial_degree']


    @property
    def seed(self) -> int:
        '''
        Random seed for reproducibility purposes.

        Returns
        -------
        int
            Random seed.
            
        '''
        return self.variables['seed']


    def _replace_variable_name(self, old_name: str, new_name: str):
        '''
        Convenient method to change variables names. Only to be used for legacy
        models conversion purposes.

        Parameters
        ----------
        old_name : str
            Old variable name.
        new_name : str
            New variable name.

        '''
        if old_name in self.variables.keys():
            self.variables[new_name] = self.variables.pop(old_name)


    def _convert_legacy_model(self, version: int) -> None:
        '''
        Convert old model to latest version. The applied changes depend on the
        version of the old model.

        Parameters
        ----------
        version : int
            Old model version.

        '''
    # Apply changes based on model version
        if version < 1:
            self._replace_variable_name('algm_name', 'algorithm')
            self._replace_variable_name('loss_name', 'loss_function')
            self._replace_variable_name('optim_name', 'optimizer')
            self._replace_variable_name('ordered_Xfeat', 'input_features')
            self._replace_variable_name('Y_dict', 'class_encoder')
            self._replace_variable_name('parentModel_path', 'parent_model_path')
            self._replace_variable_name('GT_dataset_path', 'dataset_path')
            self._replace_variable_name('TVT_rateos', 'tvt_ratios')
            self._replace_variable_name('regressorDegree', 'polynomial_degree')
            self._replace_variable_name('F1_scores', 'f1_scores')
            self.variables['batch_size'] = 0

        if version < 2:
            self._replace_variable_name('lr', 'learning_rate')
            self._replace_variable_name('wd', 'weight_decay')
            self._replace_variable_name('mtm', 'momentum')
            self._replace_variable_name('optim_state_dict', 'optimizer_state_dict')

        if version < 3:
            # 'balancing_info' list[dict] to list[BalancingInfo]
            old_bal_info = cast(list[dict], self.variables['balancing_info'])
            bal_info: list[dtools.BalancingInfo] = [
                {
                    'Strategy': {
                        cf.get_dictkey(self.encoder, k): v 
                        for k, v in i['Strategy'].items()
                    },
                    'Over-sampling algorithm': i['OS'],
                    'Under-sampling algorithm': i['US'],
                    'K-neighbours': i['k-neigh_OS'],
                    'M-neighbours': i['m-neigh_OS'],
                    'N-neighbours': i['n-neigh_US'],
                    'Seed': i['Seed'],
                    'New TVT ratios': i['New TVT rateos']
                } for i in old_bal_info
            ]
            self.variables['balancing_info'] = bal_info
            # 'accuracy' list to tuple
            train_acc, valid_acc, test_acc = self.variables['accuracy']
            self.variables['accuracy'] = (train_acc, valid_acc, test_acc)
            # 'loss' list to tuple
            train_loss, valid_loss, test_loss = self.variables['loss']
            self.variables['loss'] = (train_loss, valid_loss, test_loss)
            # 'f1_scores' array (3x2 or 3x3) to tuple[tuple, tuple] (2x3)
            f1_old = cast(npt.NDArray[np.float_], self.variables['f1_scores'])
            [[*_, mtr, wtr], [*_, mvd, wvd], [*_, mts, wts]] = f1_old.tolist()
            self.variables['f1_scores'] = ((mtr, mvd, mts), (wtr, wvd, wts))

    # Set updated model version
        self.variables['version'] = self._current_version

    # Save as a new model (do not overwrite old one)
        if (path := self.filepath) is not None:
            new_path = cf.extend_filename(path, f'_v{self._current_version}')
            log_path = self.generate_log_path(new_path)
            extended = settings.manager.get('data/extended_model_log')
            self.save(new_path, log_path=log_path, extended_log=extended)
            self._filepath = new_path


    def missing_variables(self) -> set[str]:
        '''
        Return missing model's variables.

        Returns
        -------
        missing : set[str]
            Missing variables.

        '''
        missing = set(self._all_variables) - set(self.variables.keys())
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


    def get_optimizer(self, network: NeuralNetwork) -> Optimizer:
        '''
        Return the optimization function used by this model to train the neural
        network 'network'.

        Parameters
        ----------
        network : NeuralNetwork
            Trained neural network.

        Returns
        -------
        Optimizer
            The adopted optimizer.

        Raises
        ------
        ValueError
            Raised if optimizer cannot be identified from model's variables.

        Example
        -------
        >>> variables = {'algorithm': 'Softmax Regression', 'optimizer': 'SGD', ...}
        >>> model = EagerModel(variables)
        >>> network = model.get_network_architecture()
        >>> network.to(model.variables['device']) # optional
        >>> optimizer = model.get_optimizer(network)

        To update previous model, just load its state dicts:
        >>> network.load_state_dict(model.variables['model_state_dict'])
        >>> optimizer.load_state_dict(model.variables['optimizer_state_dict'])

        '''
        optimizer = self.optimizer
        lr, wd, mtm, _, _ = self.hyperparameters
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
        self._filepath = outpath
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
        var_keys = self._all_variables if extended else self._base_variables
        with open(outpath, 'w', encoding="utf-8") as log:
            for k in var_keys:
                match k:
                    case 'algorithm':
                        title = 'ALGORITHM'
                        value = self.variables[k]
                    case 'loss_function':
                        title = 'LOSS FUNCTION'
                        value = self.variables[k]
                    case 'optimizer':
                        title = 'OPTIMIZER FUNCTION'
                        value = self.variables[k]
                    case 'input_features':
                        title = 'INPUT FEATURES'
                        value = ', '.join(self.variables[k])
                    case 'class_encoder':
                        title = 'OUTPUT CLASSES'
                        value = str(self.variables[k])[1:-1].replace(', ', '\n')
                    case 'device':
                        title = 'DEVICE'
                        value = self.variables[k]
                    case 'seed':
                        title = 'RANDOM SEED'
                        value = self.variables[k]
                    case 'parent_model_path':
                        title = 'PARENT MODEL PATH'
                        value = repr(self.variables[k])
                    case 'dataset_path':
                        title = 'DATASET PATH'
                        value = repr(self.variables[k])
                    case 'tvt_ratios':
                        title = 'TRAIN-VALIDATION-TEST RATIOS'
                        value = str(self.variables[k])[1:-1]
                    case 'balancing_info':
                        title = 'TRAINING DATASET BALANCING INFO'
                        v = [json.dumps(b, indent=2) for b in self.variables[k]]
                        value = '\n'.join(v)
                    case 'polynomial_degree':
                        title = 'POLYNOMIAL DEGREE'
                        value = self.variables[k]
                    case 'epochs':
                        title = 'EPOCHS'
                        value = self.variables[k]
                    case 'learning_rate':
                        title = 'LEARNING RATE'
                        value = self.variables[k]
                    case 'weight_decay':
                        title = 'WEIGHT DECAY'
                        value = self.variables[k]
                    case 'momentum':
                        title = 'MOMENTUM'
                        value = self.variables[k]
                    case 'batch_size':
                        title = 'BATCH SIZE'
                        value = self.variables[k]
                    case 'accuracy':
                        title = 'ACCURACY (TRAIN-VALIDATION-TEST)'
                        value = str(self.variables[k])[1:-1]
                    case 'loss':
                        title = 'LOSS (TRAIN-VALIDATION-TEST)'
                        value = str(self.variables[k])[1:-1]
                    case 'f1_scores':
                        title = 'F1 SCORES (TRAIN-VALIDATION-TEST)'
                        t1, t2 = 'Macro', 'Weighted'
                        v1, v2 = self.variables[k]
                        value = '\n'.join([t1, str(v1)[1:-1], t2, str(v2)[1:-1]])
                    case 'accuracy_list':
                        title = 'ACCURACY LISTS'
                        t1, t2 = 'Train', 'Validation'
                        v1, v2 = self.variables[k]
                        value = '\n'.join([t1, str(v1), t2, str(v2)])
                    case 'loss_list':
                        title = 'LOSS LISTS'
                        t1, t2 = 'Train', 'Validation'
                        v1, v2 = self.variables[k]
                        value = '\n'.join([t1, str(v1), t2, str(v2)])
                    case 'standards':
                        title = 'FEATURES STANDARDS'
                        t1, t2 = 'Mean', 'Standard Deviation'
                        v1, v2 = self.variables[k]
                        value = '\n'.join([t1, str(v1.tolist()), t2, str(v2.tolist())])
                    case 'version':
                        title = 'VERSION'
                        value = self.variables[k]
                    case _: # ignore state dicts (can be huge in large models)
                        continue

                log.write(f'{title}\n{value}\n\n\n')


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

        Raises
        ------
        ValueError
            Raised if maps from "input_stack" have different shapes.

        '''        
    # Set main attributes
        self.kind = kind
        self.name = name
        self.classification_steps = classification_steps
        self.thread = thread
        self.algorithm = None # to be reimplemented in each subclass
        self.input_stack = input_stack
        if (shape := self.input_stack.maps_shape) is None:
            raise ValueError('Maps of "input_stack" do not fit.')
        self.map_shape = shape

    @property
    def classification_pipeline(self) -> tuple[Callable, ...]:
        '''
        Defines the classification pipeline of this classifier. To reimplement
        in each child.

        Returns
        -------
        tuple
            Classification pipeline.

        '''
        return ()


    def getCoordMaps(self) -> tuple[InputMap, InputMap]:
        '''
        Return X, Y pixel indices (coordinates) maps.

        Returns
        -------
        tuple[InputMap, InputMap]
            X, Y coordinates maps.

        '''
        shape = self.map_shape
        xx, yy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        return (InputMap(xx), InputMap(yy))
        
        
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
        path = model.filepath
        kwargs = {
            'kind': 'Model-based',
            'name': 'Unknown model' if path is None else cf.path2filename(path),
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
        return (f1, f2, f3)
    

    def classify(self) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.float_]]:
        '''
        Run entire not-threaded classification process.

        Returns
        -------
        pred: NDArray[np.str_]
            Predictions.
        prob: NDArray[np.float_]
            Probability scores.

        '''
        feat_data = self.preProcessFeatureData()
        prob, pred = self.predict(feat_data)
        prob, pred = self.postProcessOutputData(prob, pred)
        return (pred, prob)


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
        feat_data, _ = norm_data(feat_data, self.model.x_mean, self.model.x_stdev)

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
        return (prob, pred)
    

    def postProcessOutputData(
        self,
        prob: torch.Tensor,
        pred: torch.Tensor
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.str_]]:
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
        prob : NDArray[np.float_]
            Rounded probability scores.
        pred : NDArray[np.str_]
            Decoded predictions.

        '''
        prob_array = prob.detach().numpy().round(2)
        pred_array = self.decodeLabels(pred)
        return (prob_array, pred_array)
    

    def encodeLabels(
        self, array: TensorLike, dtype: npt.DTypeLike = 'int16') -> npt.NDArray[Any]:
        '''
        Encode labels from text names to class IDs.

        Parameters
        ----------
        array : TensorLike
            Labels array as class names.
        dtype : DTypeLike, optional
            Encoded array dtype. The default is 'int16'.

        Returns
        -------
        encoded_array : npt.NDArray[Any]
            Encoded labels array as class IDs.

        '''
        # Convert tensor to array
        if isinstance(array, torch.Tensor):
            array = tensor2array(array)

        # Encode array
        encoded_array = np.copy(array)
        for k, v in self.model.encoder.items(): 
            encoded_array[array == k] = v
        return encoded_array.astype(dtype, copy=False)


    def decodeLabels(
        self, array: TensorLike, dtype: npt.DTypeLike = 'U8') -> npt.NDArray[Any]:
        '''
        Decode labels from class IDs to text names.

        Parameters
        ----------
        array : TensorLike
            Labels array as class IDs.
        dtype : DTypeLike, optional
            Decoded array dtype. Only use str-like dtypes. The default is 'U8'.

        Returns
        -------
        decoded_array : NDArray[Any]
            Decoded labels array as class names.
            
        '''
        # Convert tensor to array
        if isinstance(array, torch.Tensor):
            array = tensor2array(array)

        # Decode array
        decoded_array = np.copy(array).astype(dtype, copy=False)
        for k, v in self.model.encoder.items():
            decoded_array[array == v] = k
        return decoded_array



class ScikitSupervisedClassifier(Protocol):
    def fit(self, X: Any, y: Any, /) -> Self: ...
    def predict(self, X: Any, /) -> npt.NDArray[Any]: ...
    def predict_proba(self, X: Any, /) -> Any: ...

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
        self.algorithm: ScikitSupervisedClassifier # to be reimplemented in each subclass
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
        return (f1, f2, f3, f4)
    

    def classify(self) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.float_]]:
        '''
        Run entire not-threaded classification process.

        Returns
        -------
        pred: NDArray[np.str_]
            Predictions.
        prob: NDArray[np.float_]
            Probability scores.

        '''
        x_train, y_train, in_data = self.getTrainingData()
        self.fit(x_train, y_train)
        pred = self.predict(in_data)
        prob = self.computeProbabilityScores(in_data)
        return (pred, prob)


    def preProcessFeatureData(self) -> npt.NDArray[Any]:
        '''
        Perform several pre-processing operations on input feature data.

        Returns
        -------
        feat_data : NDArray[Any]
            Pre-processed input data.

        '''
    # Include pixel coordinate maps if required
        if self.proximity:
            self.input_stack.add_maps(self.getCoordMaps())

    # Get a 2D features array suited for classification (n_pix x n_maps)
        feat_data = self.input_stack.get_feature_array()

    # Normalize the data
        feat_data, _ = norm_data(feat_data)

        return feat_data


    def preProcessRoiData(self) -> npt.NDArray[np.str_]:
        '''
        Perform several pre-processing operations on ROI training data.

        Returns
        -------
        roidata : NDArray[np.str_]
            Pre-processed ROI training data array.

        '''
        roidata = self.roimap.map

    # Apply mask if required and flatten/compress the map array
        if self.input_stack.mask is None:
            roidata = roidata.flatten()
        else:
            mask_arr = self.input_stack.mask.mask
            roidata = np.ma.masked_where(mask_arr, roidata).compressed()

        return roidata


    def getTrainingData(self) -> tuple[
        npt.NDArray[Any], npt.NDArray[np.str_], npt.NDArray[Any]
    ]:
        '''
        Return training data, splitted into features (X) and labels (Y). This
        method returns the full input feature data as well.

        Returns
        -------
        x_train : NDArray[Any]
            Training feature data.
        y_train : NDArray[np.str_]
            Training target data.
        x : NDArray[Any]
            Full input feature data.

        '''
    # Get pre-processed X (feature) and Y (label) data
        x = self.preProcessFeatureData()
        y = self.preProcessRoiData()

    # Extract indices where Y (ROI map data) is actually populated with labels
        labeled_indices = (y != self.roimap._ND).nonzero()[0]

    # Use the indices to extract training data from both features and labels
        x_train = x[labeled_indices, :]
        y_train = y[labeled_indices]
        return (x_train, y_train, x)


    def fit(self, x_train: npt.NDArray[Any], y_train: npt.NDArray[np.str_]) -> None:
        '''
        Fit classifier to training data.

        Parameters
        ----------
        x_train : NDArray[Any]
            Training feature data. 
        y_train : NDArray[np.str_]
            Training target data.

        '''
        self.algorithm.fit(x_train, y_train)
    

    def predict(self, in_data: npt.NDArray[Any]) -> npt.NDArray[np.str_]:
        '''
        Predict unknown data. This method must always be called after 'fit'.

        Parameters
        ----------
        in_data : NDArray[np._str]
            Input unlabeled data.

        Returns
        -------
        pred : NDArray[Any]
            Predicted data.

        '''
        pred = self.algorithm.predict(in_data)
        return pred
    

    def computeProbabilityScores(
        self, in_data: npt.NDArray[Any]) -> npt.NDArray[np.float_]:
        '''
        Compute confidence scores.

        Parameters
        ----------
        in_data : NDArray[Any]
            Input unlabeled data.

        Returns
        -------
        prob : NDArray[np.float_]
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



class ScikitUnsupervisedClassifier(Protocol):
    def fit(self, X: Any, /) -> Self: ...
    def predict(self, X: Any, /) -> npt.NDArray[Any]: ...

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
        self.algorithm: ScikitUnsupervisedClassifier # to be reimplemented in each child class
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
        return (f1, f2, f3, f4, f5, f6, f7)
    

    def classify(self) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float_]]:
        '''
        Run entire not-threaded classification process. Warning: this method
        returns prediction labels (clusters) as integer values.

        Returns
        -------
        pred: NDArray[np.int_]
            Predictions, with clusters expressed as integer IDs.
        prob: NDArray[np.float_]
            Probability scores.

        '''
        in_data = self.preProcessFeatureData()
        self.fit(in_data)
        pred = self.predict(in_data)
        prob = self.computeProbabilityScores(in_data)
        return (pred, prob)


    def preProcessFeatureData(self) -> npt.NDArray[Any]:
        '''
        Perform several pre-processing operations on input feature data.

        Returns
        -------
        feat_data : NDArray[Any]
            Pre-processed input data.

        '''
    # Include pixel coordinate maps if required
        if self.proximity:
            self.input_stack.add_maps(self.getCoordMaps())

    # Get a 2D features array suited for classification (n_pix x n_maps)
        feat_data = self.input_stack.get_feature_array()

    # Normalize the data
        feat_data, _ = norm_data(feat_data)

        return feat_data
    

    def fit(self, in_data: npt.NDArray[Any]) -> None:
        '''
        Fit classifier to input data.

        Parameters
        ----------
        in_data : NDArray[Any]
            Input data.

        '''
        self.algorithm.fit(in_data)

    
    def predict(self, in_data: npt.NDArray[Any]) -> npt.NDArray[np.int_]:
        '''
        Cluster input data.

        Parameters
        ----------
        in_data : NDArray[Any]
            Input data.

        Returns
        -------
        pred : NDArray[np.int_]
            Clustered data with clusters expressed as integer IDs.

        '''
        pred = self.algorithm.predict(in_data)
        return pred
    

    def computeProbabilityScores(
        self, in_data: npt.NDArray[Any]) -> npt.NDArray[np.float_]:
        '''
        Compute confidence scores.

        Parameters
        ----------
        in_data : NDArray[Any]
            Input data.

        Returns
        -------
        prob : NDArray[np.float_]
            Probability scores.

        Raises
        ------
        NotImplementedError
            Raised if scores calculation fails with the implemented algorithm.

        '''
        # Calculate scores from cluster centroid-based distance (KMeans)
        transform_method = getattr(self.algorithm, 'transform', None)
        if callable(transform_method): # TODO probably needs a statistically stronger logic
            dist = np.asarray(transform_method(in_data)).min(axis=1)
            prob = 1 - dist/dist.max()
            return prob

        # This error should raise only during development of new algorithms
        e = 'Missing logic for extracting confidence score with this algorithm.'
        raise NotImplementedError(e)
    

    def computeSilhouetteScore(
        self,
        in_data: npt.NDArray[Any],
        pred: npt.NDArray[np.int_]
    ) -> tuple[dict[int, npt.NDArray[np.float_]], float] | tuple[None, None]:
        '''
        Compute silhouette score. The computation is ignored if the class 
        attribute 'do_silhouette_score' is set to False.

        Parameters
        ----------
        in_data : NDArray[Any]
            Input data.
        pred : NDArray[np.int_]
            Clustered data with clusters expressed as integer IDs.

        Returns
        -------
        sil_clust : dict[int, NDArray[np.float_]] | None
            Silhouette scores by cluster. Returns None if 'do_silhouette_score'
            is False.
        sil_avg : float | None
            Average silhouette score. Returns None if 'do_silhouette_score' is
            False.

        '''
        if not self.do_silhouette_score:
            return (None, None)

    # Define a random data sample of required size
        sample_size = int(self.silhouette_ratio * pred.size)
        rng = np.random.default_rng(self.seed)
        subset_idx = rng.permutation(pred.size)[:sample_size]
        data_slice, pred_slice = in_data[subset_idx, :], pred[subset_idx]

    # Compute silhouette score by cluster
        sil_sam = np.asarray(
            sklearn.metrics.silhouette_samples(data_slice, pred_slice))
        unq_val = np.unique(pred_slice)
        sil_clust = {u: np.sort(sil_sam[pred_slice == u]) for u in unq_val}

    # Compute average silhouette score
        sil_avg = float(np.mean(sil_sam))

        return (sil_clust, sil_avg)
    

    def computeChiScore(
        self,
        in_data: npt.NDArray[Any],
        pred: npt.NDArray[np.int_]
    ) -> float | None:
        '''
        Compute Calinski-Harabasz Index. The computation is ignored if the 
        class attribute 'do_chi_score' is set to False.

        Parameters
        ----------
        in_data : NDArray[Any]
            Input data.
        pred : NDArray[np.int_]
            Clustered data with clusters expressed as integer IDs.

        Returns
        -------
        float | None
            Calinski-Harabasz Index. Returns None if 'do_chi_score' is False.

        '''
        if not self.do_chi_score:
            return None
        return sklearn.metrics.calinski_harabasz_score(in_data, pred)
    

    def computeDbiScore(
        self, in_data: npt.NDArray[Any],
        pred: npt.NDArray[np.int_]
    ) -> float | None:
        '''
        Compute Davies-Bouldin Index. The computation is ignored if the class
        attribute 'do_dbi_score' is set to False.

        Parameters
        ----------
        in_data : NDArray[Any]
            Input data.
        pred : NDArray[np.int_]
            Clustered data with clusters expressed as integer IDs.

        Returns
        -------
        float | None
            Davies-Bouldin Index. Returns None if 'do_dbi_score' is False.

        '''
        if not self.do_dbi_score:
            return None
        return sklearn.metrics.davies_bouldin_score(in_data, pred)
        

       
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



def array2tensor(
    array: npt.NDArray[Any], dtype: npt.DTypeLike | None = None) -> torch.Tensor:
    '''
    Convert numpy array to torch Tensor.

    Parameters
    ----------
    array : NDArray[Any]
        Input numpy array
    dtype : numpy DTypeLike, optional
        Output dtype. If None, it is inferred from array. The default is None.

    Returns
    -------
    tensor : torch Tensor
        Output torch Tensor.

    '''
    # Handle type casting on NumPy side if a explicit dtype is provided
    if dtype is not None:
        array = array.astype(dtype, copy=False)

    # Convert to contiguous array if needed to prevent memory layout crashes
    if not array.flags['C_CONTIGUOUS']:
        array = np.ascontiguousarray(array)

    # Share memory layout via from_numpy (Zero Copy)
    tensor = torch.from_numpy(array)
    return tensor


def tensor2array(
    tensor: torch.Tensor, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
    '''
    Convert torch Tensor to numpy array.

    Parameters
    ----------
    tensor : torch Tensor
        Input torch Tensor.
    dtype : DTypeLike, optional
        Output dtype. If None, it is inferred from tensor. The default is None.

    Returns
    -------
    array : NDArray[Any]
        Output numpy array.

    '''
    # Ensure tensor is on CPU, detached from autograd, and memory-contiguous
    array: npt.NDArray[Any] = tensor.detach().cpu().contiguous().numpy()

    # Perform optional dtype conversion
    if dtype is not None:
        array = array.astype(dtype, copy=False)

    return array


@overload
def norm_data(
    data: torch.Tensor,
    mean: TensorLike | None = None,
    std: TensorLike | None = None,
    eps: float = 1e-8
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: ...

@overload
def norm_data(
    data: npt.NDArray[Any],
    mean: TensorLike | None = None,
    std: TensorLike | None = None,
    eps: float = 1e-8
) -> tuple[npt.NDArray[Any], tuple[npt.NDArray[Any], npt.NDArray[Any]]]: ...

def norm_data(
    data: TensorLike,
    mean: TensorLike | None = None,
    std: TensorLike | None = None,
    eps: float = 1e-8
) -> tuple[TensorLike, tuple[TensorLike, TensorLike]]:
    '''
    Apply standard score data normalization to an input array or tensor.

    Parameters
    ----------
    data : TensorLike
        Input data array or tensor.
    mean : TensorLike | None, optional
        Mean scores per feature. If None, they are computed from data. The 
        default is None.
    std : TensorLike | None, optional
        Standard deviation scores per feature. If None, they are computed from 
        data. The default is None.
    eps : float, optional
        Small constant added to standard deviation for numerical stability. The
        default is 1e-8.

    Returns
    -------
    data_norm : TensorLike
        Normalized data matching input 'data' type.
    standards : tuple[TensorLike, TensorLike]
        Mean and standard deviation matching input 'data' type.

    Raises
    ------
    TypeError
        Raised if 'data' is not a NumPy array or a PyTorch tensor.

    '''
    # Ensure 'mean' and 'std' are tensors if 'data' is a tensor
    if isinstance(data, torch.Tensor):
        with torch.no_grad():
            _mean = data.mean(dim=0) if mean is None else mean
            _std = data.std(dim=0, unbiased=False) if std is None else std
            _mean = array2tensor(_mean) if isinstance(_mean, np.ndarray) else _mean
            _std = array2tensor(_std) if isinstance(_std, np.ndarray) else _std

    # Ensure 'mean' and 'std' are numpy arrays if 'data' is a numpy array
    elif isinstance(data, np.ndarray):
        _mean = np.mean(data, axis=0) if mean is None else mean
        _std = np.std(data, axis=0) if std is None else std
        _mean = tensor2array(_mean) if isinstance(_mean, torch.Tensor) else _mean
        _std = tensor2array(_std) if isinstance(_std, torch.Tensor) else _std

    # Raise error if 'data' has an invalid type
    else:
        raise TypeError(f'Invalid data type: {type(data)}.')

    # Return normalized data
    data_norm = (data - _mean) / (_std + eps)
    return data_norm, (_mean, _std)


def map_polinomial_features(arr: npt.NDArray[Any], deg: int) -> npt.NDArray[Any]:
    '''
    Apply polynomial kernel to input features of array.  

    Parameters
    ----------
    arr : NDArray[Any]
        Input array.
    deg : int
        Polynomial degree.

    Returns
    -------
    poly_features : NDArray[Any]
        Output polynomial features array.
    
    '''
    poly = sklearn.preprocessing.PolynomialFeatures(deg, include_bias=False)
    poly_features = poly.fit_transform(arr)
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
    true: TensorLike, pred: TensorLike, ids: Sequence[int]) -> npt.NDArray[Any]:
    '''
    Compute confusion matrix.

    Parameters
    ----------
    true : TensorLike
        True classes.
    pred : TensorLike
        Predicted classes.
    ids : Sequence[int]
        Sequence of classes IDs.

    Returns
    -------
    confmat : NDArray[Any]
        Confusion matrix of shape (n_classes, n_classes).

    '''
    true = tensor2array(true) if isinstance(true, torch.Tensor) else true
    pred = tensor2array(pred) if isinstance(pred, torch.Tensor) else pred
    confmat = sklearn.metrics.confusion_matrix(true, pred, labels=ids)
    return confmat


def f1_score(
    true: TensorLike,
    pred: TensorLike, 
    avg: Literal['micro', 'macro', 'weighted']
) -> float:
    '''
    Compute average F1 score of type 'avg'.

    Parameters
    ----------
    true : TensorLike
        True classes.
    pred : TensorLike
        Predicted classes.
    avg : Literal['micro', 'macro', 'weighted']
        Average score type.

    Returns
    -------
    f1 : float
        Average F1 score.

    '''
    true = tensor2array(true) if isinstance(true, torch.Tensor) else true
    pred = tensor2array(pred) if isinstance(pred, torch.Tensor) else pred
    f1 = float(sklearn.metrics.f1_score(true, pred, average=avg))
    return f1


def accuracy_score(true: TensorLike, pred: TensorLike) -> float:
    '''
    Compute accuracy score.

    Parameters
    ----------
    true : TensorLike
        True classes.
    pred : TensorLike
        Predicted classes.

    Returns
    -------
    accuracy : float
        Accuracy score.

    '''
    true = tensor2array(true) if isinstance(true, torch.Tensor) else true
    pred = tensor2array(pred) if isinstance(pred, torch.Tensor) else pred
    accuracy = float(sklearn.metrics.accuracy_score(true, pred))
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