"""
Provides functionalities for handling and preprocessing ground truth datasets,
including data loading, transformation, and feature extraction to facilitate
machine learning tasks.

Created on Wed Oct 23 09:21:45 2024
@author: albdag
"""

from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Literal, TypedDict, get_args, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from imblearn import over_sampling, under_sampling

import convenient_functions as cf
import threads
from base import InputMapStack, MineralMap

# Balancing algorithm names
OverSamplingAlgorithmName = Literal[
    'SMOTE',
    'BorderlineSMOTE',
    'SVMSMOTE',
    'ADASYN'
]
UnderSamplingAlgorithmName = Literal[
    'RandUS',
    'NearMiss',
    'ClusterCentroids',
    'TomekLinks',
    'ENN-all',
    'ENN-mode',
    'NCR'
]

# Balancing strategy dict = {class_ID : value}
type BalancingStrategy = int | dict[int, int] | Literal['Min', 'Max', 'Mean', 'Median']

# Verbose balancing strategy dict = {class_lbl : "<old_count> -> <new_count>"}
type BalancingStrategyVerbose = dict[str, str]
BalancingInfo = TypedDict('BalancingInfo', {
    'Strategy': BalancingStrategyVerbose,
    'Over-sampling algorithm': str | None,
    'Under-sampling algorithm': str | None,
    'K-neighbours': int,
    'M-neighbours': int,
    'N-neighbours': int,
    'Seed': int,
    'New TVT ratios': tuple[float, float, float] # train, validation, test
})

class GroundTruthDataset:

    def __init__(self, dataframe: pd.DataFrame, filepath: str | None = None) -> None:
        '''
        A base class to process and manipulate ground truth datasets.

        Parameters
        ----------
        dataframe : pandas DataFrame
            Ground truth dataset.
        filepath : str or None, optional
            Filepath to the stored dataset file. The default is None.

        '''
    # Main attributes
        self.filepath = filepath
        self.dataframe = dataframe

    # X features and Y targets (as labels) and their encoder -> dict(lbl: id)
        self.features: npt.NDArray[Any] = np.array([])
        self.targets: npt.NDArray[Any] = np.array([])
        self.encoder: dict[str, int] = {}

    # Original data, untouched by possible balancing operations on train set
        self.orig_subsets_ratios: tuple[float, float, float] = (0., 0., 0.)
        self.orig_x_train: npt.NDArray[Any] = np.array([])
        self.orig_y_train: npt.NDArray[Any] = np.array([])
        self.orig_train_counter: dict[str, int] = {}

    # Current X features and Y targets (as IDs) split into subsets
        self.x_train: npt.NDArray[Any] = np.array([])
        self.x_valid: npt.NDArray[Any] = np.array([])
        self.x_test: npt.NDArray[Any] = np.array([])
        self.y_train: npt.NDArray[Any] = np.array([])
        self.y_valid: npt.NDArray[Any] = np.array([])
        self.y_test: npt.NDArray[Any] = np.array([])

    # Current subsets counters -> dict(target_labels: count)
        self.train_counter: dict[str, int] = {}
        self.valid_counter: dict[str, int] = {}
        self.test_counter: dict[str, int] = {}


    def __len__(self) -> int:
        '''
        Return the length of the dataset's dataframe.

        Returns
        -------
        int
            Dataset's length.

        '''
        return len(self.dataframe.index)


    @classmethod
    def load(
        cls,
        filepath: str,
        dec: str,
        sep: str | None = None,
        chunks: bool = False
    ) -> 'GroundTruthDataset':
        '''
        Load dataset from CSV file.

        Parameters
        ----------
        filepath : str
            Dataset filepath. Must have .csv extension.
        dec : str
            CSV decimal point character.
        sep : str or None, optional
            CSV separator character. If None, it is inferred. The default is 
            None.
        chunks : bool, optional
            Whether to read the CSV file in chunks. The default is False.

        Returns
        -------
        GroundTruthDataset
            A new ground truth dataset.

        Raises
        ------
        TypeError
            Raised if the filepath extension is not .csv.

        '''
        if not filepath.lower().endswith('.csv'):
            raise TypeError('File must have .csv extension.')
        
        if chunks:
            dataframe = CsvChunkReader(dec, sep).read_combine(filepath)
            
        else:
            eng = 'python' if sep is None else 'c'
            dataframe = pd.read_csv(filepath, sep=sep, decimal=dec, engine=eng)

        return cls(dataframe, filepath)
    

    @classmethod
    def from_maps(
        cls,
        input_stack: InputMapStack,
        mineral_map: MineralMap,
        columns: list[str]
    ) -> 'GroundTruthDataset':
        '''
        Create new dataset from InputMapStack and MineralMap data.

        Parameters
        ----------
        input_stack : InputMapStack
            Input maps stack for dataset features.
        mineral_map : MineralMap
            Mineral map for dataset targets.
        columns : list[str]
            Dataset's column names.

        Returns
        -------
        GroundTruthDataset
            A new ground truth dataset.

        '''
    # Construct full dataset by stacking input maps with mineral map
        input_data = input_stack.get_feature_array()
        minmap_data = mineral_map.minmap.flatten()[:, np.newaxis]
        data = np.hstack((input_data, minmap_data))

    # Return a new ground truth dataset
        dataframe = pd.DataFrame(data=data, columns=columns)
        return cls(dataframe)


    @classmethod
    def from_existent(
        cls,
        datasets: Iterable['GroundTruthDataset']
    ) -> 'GroundTruthDataset':
        '''
        Create a new dataset by merging existent datasets. All datasets must 
        share the same column names.

        Parameters
        ----------
        datasets : Iterable[GroundTruthDataset]
            Datasets to be merged together.

        Returns
        -------
        GroundTruthDataset
            A new ground truth dataset.

        Raises
        ------
        TypeError
            Raised if any of 'datasets' is not a GroundTruthDataset.
        ValueError
            Raised if the datasets have different column names.

        '''
        dataframes: list[pd.DataFrame] = []
        column_names: set[str] = set() # comma-separated names (e.g. 'c1,c2,c3')

        for d in datasets:
            # Check for valid object type
            if not isinstance(d, GroundTruthDataset):
                raise TypeError(f'Cannot merge with object of type {type(d)}.')
            
            # Check for same column names
            column_names.add(','.join(sorted(d.columns_names())))
            if len(column_names) > 1:
                raise ValueError('Datasets have different columns.')

            # Append dataframe to list
            dataframes.append(d.dataframe)
            
        # Return a new ground truth dataset
        merged_dataframe = pd.concat(dataframes, ignore_index=True)
        return cls(merged_dataframe)


    def sub_sample(
        self,
        idx: int,
        filters: Sequence[float | int | str]
    ) -> 'GroundTruthDataset':
        '''
        Return a sub-sampled version of this dataset by selecting only the
        instances whose values in the column at index 'idx' are in 'filters'. 

        Parameters
        ----------
        idx : int
            Index of the column where the filtering is applied. 
        filters : Sequence[float | int | str]
            Sequence of filter values.

        Returns
        -------
        GroundTruthDataset
            A sub-sampled version of the ground truth dataset.

        '''
        df = self.dataframe[self.dataframe.iloc[:, idx].isin(filters)]
        df.reset_index(drop=True, inplace=True)
        return GroundTruthDataset(df)
    

    def merge(self, other: 'GroundTruthDataset') -> None:
        '''
        Merge this dataset with another one. The two datasets must share the
        same column names. Warning: once merged, this dataset attributes will
        be reset.

        Parameters
        ----------
        other : GroundTruthDataset
            The dataset to be merged with this one.

        Raises
        ------
        TypeError
            Raised if 'other' is not a GroundTruthDataset.
        ValueError
            Raised if the two datasets have different column names.

        '''
    # Check for object type
        if not isinstance(other, GroundTruthDataset):
            raise TypeError(f'Cannot merge with object of type {type(other)}')
    
    # Check for same column names
        if sorted(self.columns_names()) != sorted(other.columns_names()):
            raise ValueError('Cannot merge datasets with different columns')
    
    # Merge dataframes and update dataset
        dataframes = [self.dataframe, other.dataframe]
        self.dataframe = pd.concat(dataframes, ignore_index=True)
        self.reset()


    def rename_target(self, old_name: str, new_name: str) -> None:
        '''
        Replace target with name 'old_name' to 'new_name'. Warning: after
        renaming, the dataset attributes will reset.

        Parameters
        ----------
        old_name : str
            Current target name.
        new_name : str
            New target name.

        '''
        self.dataframe.replace(old_name, new_name, inplace=True)
        self.reset()


    def merge_targets(self, targets: list[str], name: str) -> None:
        '''
        Rename two or more targets to the same name 'name'. Warning: after
        merging, the dataset attributes will reset.

        Parameters
        ----------
        targets : list[str]
            List of target to be merged. It must contain at least two values.
        name : str
            New name for the merged targets.

        Raises
        ------
        ValueError
            Raised if 'targets' contains less than two values.

        '''
    # Check for minimum targets amount (= 2) 
        if (n_targets := len(targets)) < 2:
            raise ValueError('Targets list contains less than two elements.')
        
        self.dataframe.replace(targets, [name] * n_targets, inplace=True)
        self.reset()


    def remove_where(self, idx: int, filters: Iterable[float | int | str]) -> None:
        '''
        Remove instances whose values in the column at index 'idx' are in 
        'filters'. Warning: after removing rows, the dataset attributes will
        reset.

        Parameters
        ----------
        idx : int
            Column index.
        filters : list[float | int | str] or tuple[float | int | str, ...]
            List of filters.

        '''
        to_remove = self.dataframe.iloc[:, idx].isin(filters)
        self.dataframe = self.dataframe[~ to_remove]
        self.dataframe.reset_index(drop=True, inplace=True)
        self.reset()

    
    def reset(self) -> None:
        '''
        Reset dataset derived attributes.

        '''
        self.features = np.array([])
        self.targets = np.array([])
        self.encoder.clear()
        self.orig_subsets_ratios = (0., 0., 0.)
        self.orig_x_train = np.array([])
        self.orig_y_train = np.array([])
        self.orig_train_counter.clear()
        self.x_train = np.array([])
        self.x_valid = np.array([])
        self.x_test = np.array([])
        self.y_train = np.array([])
        self.y_valid = np.array([])
        self.y_test = np.array([])
        self.train_counter.clear()
        self.valid_counter.clear()
        self.test_counter.clear()


    def save(
        self,
        outpath: str,
        sep: str,
        dec: str,
        split: bool = False, 
        max_rows: int = 2**20 # 2**20 = Excel maximum number of rows
    ) -> None: 
        '''
        Save dataframe to one or multiple CSV file(s).

        Parameters
        ----------
        outpath : str
            Output file. Must have a .csv extension.
        sep : str
            CSV separator character.
        dec : str
            CSV decimal character.
        split : bool, optional
            Whether to split the dataset into multiple files. If True, the 
            number of files is determined by 'max_rows'. The default is False.
        max_rows : int, optional
            If 'split' is True, this is the maximum allowed number of rows per
            file. If 'split' is False, this is ignored. The default is 2**20.

        Raises
        ------
        TypeError
            Raised if output path extension is not .csv.

        '''
        if not outpath.lower().endswith('.csv'):
            raise TypeError('The file extension must be .csv')
        
    # Save multiple CSV files
        if split:
            split_indices = range(0, len(self.dataframe.index), max_rows)
            dfs = (self.dataframe.iloc[i:i+max_rows, :] for i in split_indices)
            for n, df in enumerate(dfs, start=1):
                path = cf.extend_filename(outpath, str(n))
                df.to_csv(path, sep=sep, index=False, decimal=dec)

    # Save one CSV file
        else:
            self.dataframe.to_csv(outpath, sep=sep, index=False, decimal=dec)


    def are_subsets_split(self) -> bool:
        '''
        Check if dataset has been already split into train, validation and test
        subsets.

        Returns
        -------
        bool
            Whether dataset has been split.

        '''
        return sum(self.orig_subsets_ratios) > 0


    def split_features_targets(
        self,
        split_idx: int = -1,
        xtype: npt.DTypeLike = 'int32',
        ytype: npt.DTypeLike = 'U8',
        spliton: Literal['columns', 'rows'] = 'columns'
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Split X features from Y targets.

        Parameters
        ----------
        split_idx : int, optional
            The splitting index. The default is -1.
        xtype : numpy DTypeLike, optional
            X features dtype. The default is 'int32'.
        ytype : numpy DTypeLike, optional
            Y targets dtype. The default is 'U8'.
        spliton : Literal['columns', 'rows'], optional
            Split dataset along 'columns' or 'rows'. The default is 'columns'.

        Returns
        -------
        self.features : numpy ndarray
            X features.
        self.targets : numpy ndarray
            Y targets.

        Raises
        ------
        ValueError
            Raised if 'spliton' is not 'columns' or 'rows'.

        '''
        dataset = self.dataframe.to_numpy()

        match spliton:
            case 'rows':
                dataset = dataset.T
            case 'columns':
                pass
            case _:
                raise ValueError(f'Invalid "spliton" argument: {spliton}')

        self.features = dataset[:, :split_idx].astype(xtype, copy=False)
        self.targets = dataset[:, split_idx].astype(ytype, copy=False)
        return self.features, self.targets
    

    def update_encoder(self, parent_encoder: dict[str, int] | None = None) -> None:
        '''
        Refresh the encoder. Can inherit from a parent encoder. Features and
        targets must be split before calling this method. For more details, see
        'split_features_targets' method.

        Parameters
        ----------
        parent_encoder : dict[str, int] or None, optional
            Existent encoder -> {target_label: target_ID}. The default is None.

        Raises
        ------
        ValueError
            Raised if features and targets have not been split yet.

        '''
        if not self.features.size or not self.targets.size:
            raise ValueError('Features and targets are not split yet.')
        
        self.encoder = parent_encoder if parent_encoder is not None else {}
        for u in np.unique(self.targets):
            if not u in self.encoder:
                self.encoder[u] = len(self.encoder)


    def split_subsets( # TODO check that ratios sum up to 1 (?)
        self,
        train_ratio: float,
        valid_ratio: float, 
        test_ratio: float,
        seed: int | None = None,
        axis: int = 0
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        '''
        Split X features and Y targets into train, validation and test sets. 
        The encoder must be populated before calling this method. For more 
        details, see 'update_encoder' method.

        Parameters
        ----------
        train_ratio : float
            Percentage of data to be included in training set (range [0, 1]).
        validation_ratio : float
            Percentage of data to be included in validation set (range [0, 1]). 
        test_ratio : float
            Percentage of data to be included in test set (range [0, 1]).
        seed : int, optional
            Random seed for reproducibility. The default is None.
        axis : int, optional
            The array axis along which to split. The default is 0.

        Returns
        -------
        feat_split : list[np.ndarray]
            Train, validation and test sets of X features.
        targ_split : list[np.ndarray]
            Train, validation and test sets of Y targets.

        Raises
        ------
        ValueError
            Raised if the encoder has not been populated yet.

        '''
        if not len(self.encoder):
            raise ValueError('Dataset encoder is empty.')
        
    # Store the current ratios as the original subsets ratios
        self.orig_subsets_ratios = (train_ratio, valid_ratio, test_ratio)

    # Encode targets' labels as IDs    
        targ_ids = np.empty(self.targets.shape, dtype='uint16')
        for lbl, id in self.encoder.items():
            targ_ids[self.targets==lbl] = id

    # Apply permutations to dataset
        feat_perm, targ_perm = self.shuffle(self.features, targ_ids, axis, seed)

    # Define split indices
        n_instances = self.features.shape[axis]
        idx = [
            int(n_instances * train_ratio), 
            int(n_instances * (train_ratio + valid_ratio))
        ]

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
    

    def shuffle(
        self,
        x_feat: np.ndarray,
        y_targ: np.ndarray,
        axis: int = 0,
        seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Apply random permutation to provided features and targets arrays.

        Parameters
        ----------
        x_feat : numpy ndarray
            Features array.
        y_targ : numpy ndarray
            Targets array.
        axis : int, optional
            Permutation is applied along this axis. The default is 0.
        seed : int or None, optional
            If provided, sets a permutation seed. If None, the seed is chosen 
            randomly. The default is None.
            
        Returns
        -------
        x_feat : numpy ndarray
            Permuted features array.
        y_targ : numpy ndarray
            Permuteed targets array.

        Raises
        ------
        ValueError
            Raised if 'x_feat' and 'y_targ' have different lengths along 'axis'.

        '''
        
        if (len_x := x_feat.shape[axis]) != (len_y := y_targ.shape[axis]):
            raise ValueError(f'Different length for x: {len_x} and y: {len_y}.')
        
        perm = np.random.default_rng(seed).permutation(len_x)
        return x_feat[perm], y_targ[perm]


    def update_counters(self) -> None:
        '''
        Refresh train, validation and test counters. Dataset must be split into
        subsets before calling this method. See 'split_subsets' method for more
        details.

        Raises
        ------
        ValueError
            Raised if dataset has not been split into subsets yet.

        '''
        if not self.are_subsets_split():
            raise ValueError('Dataset is not split in subsets.')
        
        for lbl, id in self.encoder.items():
            self.train_counter[lbl] = np.count_nonzero(self.y_train == id)
            self.valid_counter[lbl] = np.count_nonzero(self.y_valid == id)
            self.test_counter[lbl] = np.count_nonzero(self.y_test == id)


    @overload # when 'verbose' is explicitly False (default)
    def parse_balancing_strategy(
        self,
        strategy: BalancingStrategy,
        verbose: Literal[False] = False
    ) -> tuple[dict[int, int], dict[int, int]]: ...

    @overload # when 'verbose' is explicitly True
    def parse_balancing_strategy(
        self,
        strategy: BalancingStrategy,
        verbose: Literal[True]
    ) -> tuple[dict[int, int], dict[int, int], BalancingStrategyVerbose]: ...

    @overload # when 'verbose' is a boolean variable
    def parse_balancing_strategy(
        self,
        strategy: BalancingStrategy,
        verbose: bool
    ) -> tuple[
        dict[int, int], dict[int, int]
        ] | tuple[
        dict[int, int], dict[int, int], BalancingStrategyVerbose
        ]: ...

    def parse_balancing_strategy(
        self,
        strategy: BalancingStrategy,
        verbose: bool = False
    ) -> tuple[
        dict[int, int], dict[int, int]
        ] | tuple[
        dict[int, int], dict[int, int], BalancingStrategyVerbose
        ]:
        '''
        Build over-sampling and under-sampling strategies based on the provided
        overall balancing strategy. The outputs of this method are a required
        input parameter for 'oversample' and 'undersample' methods. Dataset 
        must be split into subsets before calling this method. For more details
        see 'split_subsets' method.

        Parameters
        ----------
        strategy : BalancingStrategy
            Overall balancing strategy. It can be:
                - int: all classes will be resampled to this specific value.
                - str: a predefined function: ['Min', 'Max', 'Mean', 'Median'].
                - dict: required value for each class ID -> {class_ID: value}.
        verbose : bool, optional
            If True, include a class by class strategy info dictionary. The 
            default is False.

        Returns
        -------
        os_strat : dict[int, int]
            Over-sampling strategy -> {class_ID: requested amount}.
        us_strat : dict[int, int]
            Under-sampling strategy -> {class_ID: requested amount}.
        strategy_info : BalancingStrategyVerbose, optional
            Strategy info dictionary. Returned if 'verbose' is True.

        Raises
        ------
        ValueError
            Raised if dataset has not been split into subsets yet.
        ValueError
            Raised if 'strategy' is an invalid string (not one of 'Min', 'Max',
            'Mean', 'Median').
        TypeError
            Raised if 'strategy' is not a valid type (not int, str or dict).
        KeyError
            Raised if 'strategy' is a dict, but it does not contain all train
            class IDs.

        '''
        if not self.are_subsets_split():
            raise ValueError('Dataset is not split in subsets.')
    
    # Parse strategy parameter
        unq, cnt = np.unique(self.y_train, return_counts=True)

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
                raise ValueError(f'Invalid strategy: {strategy}')

        elif type(strategy) == dict:
            # Raises KeyError if any of the class IDs in 'unq' is missing from
            # 'strategy' dict keys
            num = [strategy[u] for u in unq]

        else:
            raise TypeError(f'Invalid type for strategy: {type(strategy)}')
        
    # Set over-sampling and under-sampling strategies
        os_strat, us_strat = {}, {}
        for u, c, n in zip(unq, cnt, num):
            if n >= c:
                os_strat[u] = n
            else:
                us_strat[u] = n

    # If verbose is True, also return a class by class strategy info dictionary
        if verbose:
            unq_labels = [str(cf.get_dictkey(self.encoder, u)) for u in unq]
            strategy_verbose = [f'{c} -> {n}' for c, n in zip(cnt, num)]
            strategy_info = dict(zip(unq_labels, strategy_verbose))
            return (os_strat, us_strat, strategy_info)
        else:
            return (os_strat, us_strat)


    @overload # when 'verbose' is explicitly False (default)
    def oversample(
        self,
        os_strat: dict[int, int],
        algorithm: OverSamplingAlgorithmName | None,
        seed: int,
        k_neigh: int, 
        m_neigh: int,
        x: npt.NDArray[Any] | None,
        y: npt.NDArray[Any] | None,
        verbose: Literal[False] = False
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...

    @overload # when 'verbose' is explicitly True
    def oversample(
        self,
        os_strat: dict[int, int],
        algorithm: OverSamplingAlgorithmName | None,
        seed: int,
        k_neigh: int, 
        m_neigh: int,
        x: npt.NDArray[Any] | None,
        y: npt.NDArray[Any] | None,
        verbose: Literal[True]
    ) -> tuple[
        npt.NDArray[Any], npt.NDArray[Any], tuple[str | None, int, int, int]
        ]: ...

    @overload # when 'verbose' is a boolean variable
    def oversample(
        self,
        os_strat: dict[int, int],
        algorithm: OverSamplingAlgorithmName | None,
        seed: int,
        k_neigh: int, 
        m_neigh: int,
        x: npt.NDArray[Any] | None,
        y: npt.NDArray[Any] | None,
        verbose: bool
    ) -> tuple[
        npt.NDArray[Any], npt.NDArray[Any]
        ] | tuple[
        npt.NDArray[Any], npt.NDArray[Any], tuple[str | None, int, int, int]
        ]: ...

    def oversample(
        self,
        os_strat: dict[int, int],
        algorithm: OverSamplingAlgorithmName | None,
        seed: int,
        k_neigh: int = 5, 
        m_neigh: int = 10,
        x: npt.NDArray[Any] | None = None,
        y: npt.NDArray[Any] | None = None,
        verbose: bool = False
    ) -> tuple[
        npt.NDArray[Any], npt.NDArray[Any]
        ] | tuple[
        npt.NDArray[Any], npt.NDArray[Any], tuple[str | None, int, int, int]
        ]:
        '''
        Apply over-sampling balancing operations.

        Parameters
        ----------
        os_strat : dict[int, int]
            Over-sampling strategy. See 'parse_balancing_strategy' method.
        algorithm : OverSamplingAlgorithmName or None
            Over-sampling algorithm. Must be one of 'SMOTE', 'BorderlineSMOTE',
            'SVMSMOTE', 'ADASYN' or None. If None, no over-sampling will be 
            performed.
        seed : int
            Random seed for reproducible results.
        k_neigh : int, optional
            Number of neighbours to be used to generate synthetic samples. The 
            default is 5.
        m_neigh : int, optional
            Number of neighbours to be used to determine if a minority sample 
            is in "danger". Only used for 'BorderlineSMOTE' and 'SVMSMOTE' 
            algorithms. The default is 10.
        x : NDArray[Any] | None, optional
            Features array. If None, the train subset features will be used.
            The default is None.
        y : NDArray[Any] | None, optional
            Targets array. If None, the train subset targets will be used. The
            default is None.
        verbose : bool, optional
            If True, also return a tuple containing info on the parameters used
            for the computation. The default is False.

        Returns
        -------
        x_bal : NDArray[Any]
            Over-sampled features array.
        y_bal : NDArray[Any]
            Over-sampled targets array.
        info : tuple[str or None, int, int, int], optional
            Parameters info tuple: ('algorithm', 'seed', 'k_neigh', 'm-neigh').
            Returned if 'verbose' is True.

        Raises
        ------
        ValueError
            Raised if 'algorithm' is not a valid over-sampling algorithm name.

        '''
    # Initialize over-sampler
        match algorithm:
            case None:
                ovs = None
            case 'SMOTE':
                ovs = over_sampling.SMOTE(
                    sampling_strategy = os_strat, # pyright: ignore[reportArgumentType]
                    random_state = seed,
                    k_neighbors = k_neigh
                )
            case 'BorderlineSMOTE':
                ovs = over_sampling.BorderlineSMOTE(
                    sampling_strategy = os_strat, # pyright: ignore[reportArgumentType]
                    random_state = seed,
                    k_neighbors = k_neigh,
                    m_neighbors = m_neigh
                )
            case 'SVMSMOTE':
                ovs = over_sampling.SVMSMOTE(
                    sampling_strategy = os_strat, # pyright: ignore[reportArgumentType]
                    random_state = seed,
                    k_neighbors = k_neigh,
                    m_neighbors = m_neigh
                )
            case 'ADASYN':
                ovs = over_sampling.ADASYN(
                    sampling_strategy = os_strat, # pyright: ignore[reportArgumentType]
                    random_state = seed,
                    n_neighbors = k_neigh
                )
            case _:
                valid = get_args(OverSamplingAlgorithmName)
                err = f'Invalid algorithm: {algorithm}. Must be one of {valid}'
                raise ValueError(err)
        
    # Compute over-sampling
        x = self.x_train if x is None else x
        y = self.y_train if y is None else y
        if ovs:
            x_bal, y_bal = [np.asarray(a) for a in ovs.fit_resample(x, y)]
        else:
            x_bal, y_bal = x, y

    # If verbose is True, also return an info tuple
        if verbose:
            info = (algorithm, seed, k_neigh, m_neigh)
            return x_bal, y_bal, info
        else:
            return x_bal, y_bal
        

    @overload # when 'verbose' is explicitly False (default)
    def undersample(
        self,
        us_strat: dict[int, int],
        algorithm: UnderSamplingAlgorithmName | None,
        seed: int,
        n_neigh: int,
        njobs: int,
        x: npt.NDArray[Any] | None,
        y: npt.NDArray[Any] | None,
        verbose: Literal[False] = False
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]: ...

    @overload # when 'verbose' is explicitly True
    def undersample(
        self,
        us_strat: dict[int, int],
        algorithm: UnderSamplingAlgorithmName | None,
        seed: int,
        n_neigh: int,
        njobs: int,
        x: npt.NDArray[Any] | None,
        y: npt.NDArray[Any] | None,
        verbose: Literal[True]
    ) -> tuple[
        npt.NDArray[Any], npt.NDArray[Any], tuple[str | None, int, int, int]
        ]: ...

    @overload # when 'verbose' is a boolean variable
    def undersample(
        self,
        us_strat: dict[int, int],
        algorithm: UnderSamplingAlgorithmName | None,
        seed: int,
        n_neigh: int,
        njobs: int,
        x: npt.NDArray[Any] | None,
        y: npt.NDArray[Any] | None,
        verbose: bool
    ) -> tuple[
        npt.NDArray[Any], npt.NDArray[Any]
        ] | tuple[
        npt.NDArray[Any], npt.NDArray[Any], tuple[str | None, int, int, int]
        ]: ...

    def undersample(
        self,
        us_strat: dict[int, int],
        algorithm: UnderSamplingAlgorithmName | None,
        seed: int,
        n_neigh: int = 3, 
        njobs: int = 1,
        x: npt.NDArray[Any] | None = None, 
        y: npt.NDArray[Any] | None = None,
        verbose: bool = False
    ) -> tuple[
        npt.NDArray[Any], npt.NDArray[Any]
        ] | tuple[
        npt.NDArray[Any], npt.NDArray[Any], tuple[str | None, int, int, int]
        ]:
        '''
        Apply under-sampling balancing operations.

        Parameters
        ----------
        us_strat : dict
            Under-sampling strategy. See 'parse_balancing_strategy' method.
        algorithm : UnderSamplingAlgorithmName or None
            Under-sampling algorithm. Must be one of 'RandUS', 'NearMiss', 
            'ClusterCentroids', 'TomekLinks', 'ENN-all', 'ENN-mode', 'NCR' or
            None. If None, no under-sampling will be performed.
        seed : int
            Random seed for reproducible results.
        n_neigh : int, optional
            Number of neighbours to be used to compute the average distance to 
            the minority point samples. The default is 3.
        njobs : int, optional
            Number of CPU cores used during computation. If -1 all available 
            processessors are used. The default is 1.
        x : numpy ndarray or None, optional
            Features array. If None, the train subset features will be used.
            The default is None.
        y : numpy ndarray or None, optional
            Targets array. If None, the train subset targets will be used. The
            default is None.
        verbose : bool, optional
            If True, also return a tuple containing info on the parameters used
            for the computation. The default is False.

        Returns
        -------
        x_bal : numpy ndarray
            Under-sampled features array.
        y_bal : numpy ndarray
            Under-sampled targets array.
        info : tuple[str or None, int, int, int], optional
            Parameters info tuple: ('algorithm', 'seed', 'n_neigh', 'njobs').
            Returned if 'verbose' is True.

        Raises
        ------
        ValueError
            Raised if 'algorithm' is not a valid under-sampling algorithm name.

        '''
    # Initialize under-sampler
        match algorithm:
            case None:
                uds = None
            case 'RandUS':
                uds = under_sampling.RandomUnderSampler(
                    sampling_strategy = us_strat, # pyright: ignore[reportArgumentType]
                    random_state = seed
                )
            case 'NearMiss':
                uds = under_sampling.NearMiss(
                    sampling_strategy = us_strat, # pyright: ignore[reportArgumentType]
                    n_neighbors = n_neigh,
                    n_jobs = njobs
                )
            case 'ClusterCentroids':
                uds = under_sampling.ClusterCentroids(
                    sampling_strategy = us_strat, # pyright: ignore[reportArgumentType]
                    random_state = seed
                )
            case 'TomekLinks':
                uds = under_sampling.TomekLinks(
                    sampling_strategy = list(us_strat.keys()), # pyright: ignore[reportArgumentType]
                    n_jobs = njobs
                )
            case 'ENN-all' | 'ENN-mode':
                uds = under_sampling.EditedNearestNeighbours(
                    sampling_strategy = list(us_strat.keys()), # pyright: ignore[reportArgumentType]
                    n_neighbors = n_neigh,
                    kind_sel = algorithm.split('-')[-1],
                    n_jobs = njobs
                )
            case 'NCR':
                uds = under_sampling.NeighbourhoodCleaningRule(
                    sampling_strategy = list(us_strat.keys()), # pyright: ignore[reportArgumentType]
                    n_neighbors = n_neigh,
                    n_jobs = njobs
                )
            case _:
                valid = get_args(UnderSamplingAlgorithmName)
                err = f'Invalid algorithm: {algorithm}. Must be one of {valid}'
                raise ValueError(err)

    # Compute under-sampling
        x = self.x_train if x is None else x
        y = self.y_train if y is None else y
        if uds:
            x_bal, y_bal = [np.asarray(a) for a in uds.fit_resample(x, y)]
        else:
            x_bal, y_bal = x, y

    # If verbose is True, also return an info tuple
        if verbose:
            info = (algorithm, seed, n_neigh, njobs)
            return x_bal, y_bal, info
        else:
            return x_bal, y_bal
        

    def apply_balancing(self, balanced_x: np.ndarray, balanced_y: np.ndarray) -> None:
        '''
        Update train subset and its counter after having performed balancing
        operations. See 'oversample' and 'undersample' methods. Dataset must be
        split into subsets before calling this method. For more details, see
        'split_subsets' method.

        Parameters
        ----------
        balanced_x : numpy ndarray
            Balanced features.
        balanced_y : numpy ndarray
            Balanced targets ad IDs.

        Raises
        ------
        ValueError
            Raised if dataset has not been split into subsets yet.

        '''
        if not self.are_subsets_split():
            raise ValueError('Dataset is not split in subsets.')

        self.x_train, self.y_train = balanced_x, balanced_y
        for lbl, id in self.encoder.items():
            self.train_counter[lbl] = np.count_nonzero(self.y_train==id)


    def discard_balancing(self) -> None:
        '''
        Discard all balancing operations on train set by restoring original
        train subset and its counter. Dataset must be split into subsets before
        calling this method. For more details, see 'split_subsets' method.

        Raises
        ------
        ValueError
            Raised if dataset has not been split into subsets yet.

        '''
        if not self.are_subsets_split():
            raise ValueError('Dataset is not split in subsets.')
        
        self.x_train = self.orig_x_train.copy()
        self.y_train = self.orig_y_train.copy()
        self.train_counter = self.orig_train_counter.copy()


    def balance_trainset(
        self,
        strategy: BalancingStrategy,
        seed: int, 
        osa: OverSamplingAlgorithmName | None = None,
        usa: UnderSamplingAlgorithmName | None = None,
        kos: int = 5,
        mos: int = 10,
        nus: int = 3,
        njobs: int = 1
    ) -> None:
        '''
        Convenient method to run entire not-threaded balancing session on the 
        train subset.

        Parameters
        ----------
        strategy : BalancingStrategy
            Overall balancing strategy. See 'parse_balancing_strategy' method.
        seed : int
            Random seed for reproducible results.
        osa : OverSamplingAlgorithmName or None, optional
            Over-sampling algorithm. See 'oversample' method for a list of 
            possible choices. If None, no over-sampling will be performed. The
            default is None.
        usa : UnderSamplingAlgorithmNames or None, optional
            Under-sampling algorithm. See 'undersample' method for a list of
            possible choices. If None, no under-sampling will be performed. The
            default is None.
        kos : int, optional
            Number of k-neighbours to consider in over-sampling algorithm. See
            'k_neigh' parameter in 'oversample' method for more details. The
            default is 5.
        mos : int, optional
            Number of m-neighbours to consider in over-sampling algorithm. See
            m_neigh parameter in 'oversample' method for more details. The 
            default is 10.
        nus : int, optional
            Number of n-neighbours to consider in under-sampling algorithm. See
            n_neigh parameter in 'undersample' method for more details. The
            default is 3.
        njobs : int, optional
            Number of CPU cores used during under-sampling computation. If -1 
            all available processessors are used. The default is 1.

        '''
    # Get balancing parameters
        x_train, y_train = self.x_train, self.y_train
        os_strat, us_strat = self.parse_balancing_strategy(strategy)

    # Under-sample and then over-sample train subset
        x_train, y_train = self.undersample(us_strat, usa, seed, nus, njobs,
                                            x=None, y=None)
        x_train, y_train = self.oversample(os_strat, osa, seed, kos, mos,
                                           x=x_train, y=y_train)
        
    # Apply balancing operations to dataset
        if osa or usa:
            x_train, y_train = self.shuffle(x_train, y_train, seed=seed)
            self.apply_balancing(x_train, y_train)


    def counters(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        '''
        Return all counters.

        Returns
        -------
        dict[str, int]
            Train set counter.
        dict[str, int]
            Validation set counter.
        dict[str, int]
            Test set counter.

        '''
        return self.train_counter, self.valid_counter, self.test_counter
    

    def current_subsets_ratios(self) -> tuple[float, float, float]:
        '''
        Return the current train, validation and test ratios.

        Returns
        -------
        float
            Train set ratio.
        float
            Validation set ratio.
        float
            Test set ratio.

        '''
        tr_size, vd_size, ts_size = [sum(c.values()) for c in self.counters()]
        tot_size = tr_size + vd_size + ts_size
        tr_ratio = tr_size / tot_size
        vd_ratio = vd_size / tot_size
        ts_ratio = ts_size / tot_size
        return tr_ratio, vd_ratio, ts_ratio
    

    def columns_names(self) -> list[str]:
        '''
        Return a list of the dataset columns names.

        Returns
        -------
        list[str]
            Columns names.
            
        '''
        return self.dataframe.columns.to_list()
    

    def column_unique(self, idx: int) -> list[float | int | str]:
        '''
        Return a list of unique values of column with index 'idx'.

        Parameters
        ----------
        idx : int
            Column index.

        Returns
        -------
        list[float or int or str]
            List of sorted unique values.

        '''
        return sorted(self.dataframe.iloc[:, idx].unique().tolist())
    

    def column_count(self, idx: int) -> dict[Hashable, int]:
        '''
        Return a dictionary containing counts of unique values of column at 
        index 'idx'. The dictionary is sorted by order of occurrence, so that 
        the first key is the most frequently occurring element.

        Parameters
        ----------
        idx : int
            Column index.

        Returns
        -------
        dict[float or int or str, int]
            Counts of unique values -> {unique: count}.

        '''
        return self.dataframe.iloc[:, idx].value_counts().to_dict()
    


class CsvChunkReader:

    def __init__(
        self,
        dec: str,
        sep: str | None = None,
        chunksize: int = 2 ** 20 // 8
    ) -> None:
        '''
        Ready to use class for reading large CSV files in chunks for better
        performance.

        Parameters
        ----------
        dec : str
            CSV decimal point character.
        sep : str or None, optional
            CSV separator character. If None, it is inferred. The default is 
            None.
        chunksize : int, optional
            Dimension of the reading batch. The default is 2 ** 20 // 8.

        '''
    # Set main attributes
        self.dec = dec
        self.sep = sep
        self.chunksize = chunksize
        self.thread = threads.CsvChunkReadingThread()


    def set_decimal(self, dec: str) -> None:
        '''
        Set CSV decimal point character.

        Parameters
        ----------
        dec : str
            Decimal point character.

        '''
        self.dec = dec


    def set_separator(self, sep: str | None) -> None:
        '''
        Set CSV separator character.

        Parameters
        ----------
        sep : str or None
            Separator character. If None, it is inferred.

        '''
        self.sep = sep


    def set_chunksize(self, chunksize: int) -> None:
        '''
        Set maximum CSV chunks size.

        Parameters
        ----------
        chunksize : int
            Chunk size.

        '''
        self.chunksize = chunksize


    def chunks_number(self, filepath: str) -> int:
        '''
        Get number of chunks in the given CSV file.

        Parameters
        ----------
        filepath : str
            Path to CSV file. Must have the .csv file extension.

        Returns
        -------
        n_chunks : int
            Number of chunks.

        Raises
        ------
        TypeError
            Raised if the filepath extension is not .csv.

        '''
        if not filepath.lower().endswith('.csv'):
            raise TypeError('The filepath must have the .csv extension.')
        with open(filepath) as temp:
            n_chunks = sum(1 for _ in temp) // self.chunksize
        return n_chunks


    def read(self, filepath: str) -> list[pd.DataFrame]:
        '''
        Read CSV file chunk by chunk and return them.

        Parameters
        ----------
        filepath : str
            Path to CSV file. Must have the .csv file extension.

        Returns
        -------
        chunk_list : list[pandas DataFrame]
            List of read chunks.

        Raises
        ------
        ValueError
            Raised if the filepath extension is not .csv.

        '''

        if not filepath.lower().endswith('.csv'):
            raise ValueError('The filepath must have the .csv extension.')

        kwargs = {
            'decimal': self.dec,
            'sep': self.sep,
            'engine': 'python',
            'chunksize': self.chunksize
        }
        with pd.read_csv(filepath, **kwargs) as reader:
            chunk_list = [chunk for chunk in reader]

        return chunk_list
    

    def read_threaded(self, filepath: str) -> None:
        '''
        Run a threaded CSV chunk reading session. 

        Parameters
        ----------
        filepath : str
            Path to CSV file. Must have the .csv file extension.

        Raises
        ------
        ValueError
            Raised if the filepath extension is not .csv.

        '''
        if not filepath.lower().endswith('.csv'):
            raise TypeError('The filepath must have the .csv extension.')

        self.thread.set_task(
            lambda: pd.read_csv(
                filepath,
                decimal=self.dec,
                sep=self.sep,
                engine='python',
                chunksize=self.chunksize
            )
        )
        self.thread.run()


    def combine_chunks(
        self,
        chunks: list[pd.DataFrame] | tuple[pd.DataFrame, ...]
    ) -> pd.DataFrame:
        '''
        Combine given chunks to reconstruct the full CSV dataframe.

        Parameters
        ----------
        chunks : list[pandas DataFrame] or tuple[pandas DataFrame, ...]
            List of CSV file chunks.

        Returns
        -------
        pandas DataFrame
            The reconstructed CSV dataframe.

        '''
        return pd.concat(chunks)
    

    def read_combine(self, filepath: str) -> pd.DataFrame:
        '''
        Conveninent method to read and return a full CSV dataframe chunk by
        chunk.

        Parameters
        ----------
        filepath : str
            Path to CSV file. Must have the .csv file extension.

        Returns
        -------
        pandas DataFrame
            The full CSV dataframe.

        Raises
        ------
        TypeError
            Raised if the filepath extension is not .csv.

        '''
        if not filepath.lower().endswith('.csv'):
            raise TypeError('The filepath must have the .csv extension.')
        
        chunks = self.read(filepath)
        dataframe = self.combine_chunks(chunks)
        return dataframe



def dataframe_preview(
    filepath: str,
    dec: str,
    sep: str | None = None,
    n_rows: int = 10
) -> pd.DataFrame:
    '''
    Return a preview of the first 'n_rows' of the dataframe stored at the given
    'filepath'.  

    Parameters
    ----------
    filepath : str
        Dataframe filepath. Must have the .csv extension. 
    dec : str
        CSV decimal point character.
    sep : str or None, optional
        CSV separator character. If None, it is inferred. The default is None.
    n_rows : int, optional
        Number of rows to show in the preview. The default is 10.

    Returns
    -------
    preview : pandas Dataframe
        First 'n_rows' of the dataframe.

    '''
    eng = 'python' if sep is None else 'c'
    preview = pd.read_csv(filepath, decimal=dec, sep=sep, nrows=n_rows, engine=eng) 
    return preview