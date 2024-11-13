# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 09:21:45 2024

@author: albdag
"""
import imblearn.over_sampling as OS
import imblearn.under_sampling as US
import pandas as pd
import numpy as np

from _base import InputMapStack, MineralMap
import convenient_functions as cf
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


    def __len__(self):
        '''
        Return the length of the dataset's dataframe.

        Returns
        -------
        int
            Dataset's length.

        '''
        return len(self.dataframe.index)


    @classmethod
    def load(cls, filepath: str, dec: str, sep: str|None=None, chunks=False):
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
            The filepath must link to a .csv file.

        '''
        if not filepath.lower().endswith('.csv'):
            raise TypeError('File must have .csv extension.')
        
        if chunks:
            dataframe = CsvChunkReader(dec, sep).read_combine(filepath)
            
        else:
            engine = 'python' if sep is None else 'c'
            dataframe = pd.read_csv(filepath, sep=sep, decimals=dec, 
                                    engine=engine)

        return cls(dataframe, filepath)
    

    @classmethod
    def from_maps(cls, input_stack: InputMapStack, mineral_map: MineralMap,
                  columns: str):
        '''
        Create new dataset from InputMap and MineralMap data.

        Parameters
        ----------
        input_stack : InputMapStack
            Input maps stack for dataset features.
        mineral_map : MineralMap
            Mineral map for dataset targets.
        columns : str
            Name of dataset's columns.

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
        

    def sub_sample(self, targets: list|tuple, idx: int):
        '''
        Return a sub-sample version of this dataset by selecting only the
        instances whose targets are within provided 'targets'. 

        Parameters
        ----------
        classes : list or tuple
            List of filter targets.

        idx : int
            Index of the targets column. 

        Returns
        -------
        GroundTruthDataset
            A sub-sampled version of the ground truth dataset.

        '''
        df = self.dataframe[self.dataframe.iloc[:, idx].isin(targets)]
        df.reset_index(drop=True, inplace=True)
        return GroundTruthDataset(df)
    

    def merge(self, other):
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
            'Other' must be a GroundTruthDataset.
        ValueError
            The two datasets must share the same column names.

        '''
    # Check for object type
        if not isinstance(other, GroundTruthDataset):
            raise TypeError(f'Cannot merge with object of type {type(other)}')
    
    # Check for same column names
        if sorted(self.dataframe.columns) != sorted(other.dataframe.columns):
            raise ValueError(f'Cannot merge datasets with different columns')
    
    # Merge dataframes and update dataset
        df = pd.concat([self.dataframe, other.dataframe], ignore_index=True)
        self.dataframe = df
        self.reset()


    def rename_target(self, old_name: str, new_name: str):
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


    def merge_targets(self, targets: list[str], name: str):
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
            Raised if provided targets are less than two.

        '''
    # Check for minimum targets amount (= 2) 
        if (n_targets := len(targets)) < 2:
            raise ValueError('Targets list contains less than two elements.')
        
        self.dataframe.replace(targets, [name] * n_targets, inplace=True)
        self.reset()


    def remove_where(self, column_idx: int, values: tuple|list):
        '''
        Remove entire row of data where column at index 'column_idx' is equal
        to any of the values listed in 'values'. Warning: after removing rows,
        the dataset attributes will reset.

        Parameters
        ----------
        column_idx : int
            Column index.
        values : tuple | list
            List of values to use to remove rows.

        '''
        to_remove = self.dataframe.iloc[:, column_idx].isin(values)
        self.dataframe = self.dataframe[~ to_remove]
        self.dataframe.reset_index(drop=True, inplace=True)
        self.reset()

    
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


    def save(self, outpath: str, sep: str, dec: str, split=False, 
             max_rows=2**20): # 2**20 = Excel maximum number of rows
        '''
        Save dataset's dataframe to one or multiple CSV file(s).

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
            Output path must have a .csv extension.

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


    def split_features_targets(self, split_idx=-1, xtype='int32', ytype='U8',
                               spliton='columns'):
        '''
        Split X features from Y targets.

        Parameters
        ----------
        split_idx : int, optional
            The splitting index. The default is -1.
        xtype : str, optional
            X features dtype. The default is 'int32'.
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
    

    def update_encoder(self, parent_encoder: dict|None=None):
        '''
        Refresh the encoder. Can inherit from a parent encoder.

        Parameters
        ----------
        parent_encoder : dict or None, optional
            Existent encoder. The default is None.

        Raises
        ------
        ValueError
            Features and targets must be split before calling this function.

        '''
        if self.targets is None:
            raise ValueError('Features and targets are not split yet.')
        
        self.encoder = parent_encoder if parent_encoder is not None else {}
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
        targ_ids = np.empty(self.targets.shape, dtype='uint16')
        for lbl, id in self.encoder.items():
            targ_ids[self.targets==lbl] = id

    # Apply permutations to dataset
        feat_perm, targ_perm = self.shuffle(self.features, targ_ids, axis, seed)

    # Define split indices
        n_instances = self.features.shape[axis]
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
    

    def shuffle(self, x_feat: np.ndarray, y_targ: np.ndarray, axis=0,
                seed: int|None = None):
        '''
        Apply permutation to provided features (x) and targets (y) arrays.

        Parameters
        ----------
        x_feat : ndarray
            Features array.
        y_targ : np.ndarray
            Targets array.
        axis : int, optional
            Permutation is applied along this axis. The default is 0.
        seed : int | None, optional
            If provided, sets a permutation seed. If None, the seed is chosen 
            randomly. The default is None.

        Returns
        -------
        x_feat : ndarray
            Permuted features array.
        y_targ : np.ndarray
            Permuteed targets array.

        Raises
        ------
        ValueError
            x_feat and y_targ must have the same length along axis.

        '''
        
        if (len_x := x_feat.shape[axis]) != (len_y := y_targ.shape[axis]):
            raise ValueError(f'Different length for x={len_x} and y={len_y}.')
        
        perm = np.random.default_rng(seed).permutation(len_x)
        return x_feat[perm], y_targ[perm]


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


    def parse_balancing_strategy(self, strategy: int|str|dict, verbose=False):
        '''
        Build over-sampling and under-sampling strategies based on the provided
        overall balancing strategy. The outputs of this function are a required
        input parameter for oversample() and undersample() functions.

        Parameters
        ----------
        strategy : int | str | dict
            Overall balancing strategy. It can be:
                - int: all classes will be resampled to this specific value.
                - str: a predefined function: ['Min', 'Max', 'Mean', 'Median'].
                - dict: a dictionary with the required value for each class.
        verbose : bool, optional
            If True, include a class by class strategy info dictionary. The 
            default is False.

        Returns
        -------
        os_strat : dict
            Over-sampling strategy.
        us_strat : dict
            Under-sampling strategy.
        info : dict, optional
            Class by class strategy info dictionary, if verbose is True.

        Raises
        ------
        ValueError
            Dataset must be split into subsets before calling this function.
        ValueError
            The provided string strategy is invalid.
        TypeError
            The provided strategy is not of a valid type.

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
            num = list(strategy.values())

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
            info = dict(zip(unq, [f'{c} -> {n}' for c, n in zip(cnt, num)]))
            return os_strat, us_strat, info
        else:
            return os_strat, us_strat
        

    def oversample(self, os_strat: dict, algorithm: str|None, seed: int, k=5, 
                   m=10, x: np.ndarray|None = None, y: np.ndarray|None = None,
                   verbose=False):
        '''
        Apply over-sampling balancing operations.

        Parameters
        ----------
        os_strat : dict
            Over-sampling strategy. (See parse_balancing_strategy() function).
        algorithm : str | None
            Over-sampling algorithm. Must be one of 'SMOTE', 'BorderlineSMOTE',
            'SVMSMOTE', 'ADASYN' or None. If None, no over-sampling will be 
            performed.
        seed : int
            Random seed for reproducible results.
        k : int, optional
            Number of neighbours to be used to generate synthetic samples. The 
            default is 5.
        m : int, optional
            Number of neighbours to be used to determine if a minority sample 
            is in "danger". Only valid for 'BorderlineSMOTE' and 'SVMSMOTE'. 
            The default is 10.
        x : ndarray | None, optional
            Features array. If None, the train subset features will be used.
            The default is None.
        y : ndarray | None, optional
            Targets array. If None, the train subset targets will be used. The
            default is None.
        verbose : bool, optional
            If True, include a tuple containing info on the parameters used for
            the computation. The default is False.

        Returns
        -------
        x_bal : ndarray
            Over-sampled features array.
        y_bal : ndarray
            Over-sampled targets array.
        info : dict, optional
            Parameters info tuple, if verbose is True.

        Raises
        ------
        ValueError
            Algorithm must be one of 'SMOTE', 'BorderlineSMOTE', 'SVMSMOTE', 
            'ADASYN' or None.

        '''
    # Initialize over-sampler
        if algorithm is None:
            ovs = None

        elif algorithm == 'SMOTE':
            ovs = OS.SMOTE(sampling_strategy = os_strat,
                           random_state = seed,
                           k_neighbors = k)
        
        elif algorithm == 'BorderlineSMOTE':
            ovs = OS.BorderlineSMOTE(sampling_strategy = os_strat,
                                     random_state = seed,
                                     k_neighbors = k,
                                     m_neighbors = m)
        elif algorithm == 'SVMSMOTE':
            ovs = OS.SVMSMOTE(sampling_strategy = os_strat,
                              random_state = seed,
                              k_neighbors = k,
                              m_neighbors = m)
        
        elif algorithm == 'ADASYN':
            ovs = OS.ADASYN(sampling_strategy = os_strat,
                            random_state = seed,
                            n_neighbors = k)

        else:
            valid_alg = ['SMOTE', 'BorderlineSMOTE', 'SVMSMOTE', 'ADASYN']
            err = f'Invalid algorithm: {algorithm}. Must be one of {valid_alg}'
            raise ValueError(err)
        
    # Compute over-sampling
        x = self.x_train if x is None else x
        y = self.y_train if y is None else y
        if ovs:
            x_bal, y_bal = ovs.fit_resample(x, y)
        else:
            x_bal, y_bal = x, y

    # If verbose is True, also return an info tuple
        if verbose:
            info = (algorithm, seed, k, m)
            return x_bal, y_bal, info
        else:
            return x_bal, y_bal
        

    def undersample(self, us_strat: dict, algorithm: str|None, seed: int, n=3, 
                    njobs=1, x: np.ndarray|None = None, 
                    y: np.ndarray|None = None, verbose=False):
        '''
        Apply under-sampling balancing operations.

        Parameters
        ----------
        us_strat : dict
            Under-sampling strategy. (See parse_balancing_strategy() function).
        algorithm : str | None
            Under-sampling algorithm. Must be one of 'RandUS', 'NearMiss', 
            'ClusterCentroids', 'TomekLinks', 'ENN-all', 'ENN-mode', 'NCR' or
            None. If None, no under-sampling will be performed.
        seed : int
            Random seed for reproducible results.
        n : int, optional
            Number of neighbours to be used to compute the average distance to 
            the minority point samples. The default is 3.
        njobs : int, optional
            Number of CPU cores used during computation. If -1 all available 
            processessors are used. The default is 1.
        x : ndarray | None, optional
            Features array. If None, the train subset features will be used.
            The default is None.
        y : ndarray | None, optional
            Targets array. If None, the train subset targets will be used. The
            default is None.
        verbose : bool, optional
            If True, include a tuple containing info on the parameters used for
            the computation. The default is False.

        Returns
        -------
        x_bal : ndarray
            Under-sampled features array.
        y_bal : ndarray
            Under-sampled targets array.
        info : dict, optional
            Parameters info tuple, if verbose is True.

        Raises
        ------
        ValueError
            Algorithm must be one of 'RandUS', 'NearMiss', 'ClusterCentroids', 
            'TomekLinks', 'ENN-all', 'ENN-mode', 'NCR' or None.

        '''
    # Initialize under-sampler
        if algorithm is None:
            uds = None

        elif algorithm == 'RandUS':
            uds = US.RandomUnderSampler(sampling_strategy = us_strat,
                                        random_state = seed)
        
        elif algorithm == 'NearMiss':
            uds = US.NearMiss(sampling_strategy = us_strat,
                              n_neighbors = n,
                              n_jobs = njobs)
        
        elif algorithm == 'ClusterCentroids':
            uds = US.ClusterCentroids(sampling_strategy = us_strat,
                                      random_state = seed)
        
        elif algorithm == 'TomekLinks':
            us_strat = list(us_strat.keys())
            uds = US.TomekLinks(sampling_strategy = us_strat,
                                n_jobs = njobs)
        
        elif algorithm in ('ENN-all', 'ENN-mode'):
            us_strat = list(us_strat.keys())
            kind = algorithm.split('-')[-1]
            uds = US.EditedNearestNeighbours(sampling_strategy = us_strat,
                                             n_neighbors = n,
                                             kind_sel = kind,
                                             n_jobs = njobs)

        elif algorithm == 'NCR':
            us_strat = list(us_strat.keys())
            uds = US.NeighbourhoodCleaningRule(sampling_strategy = us_strat,
                                               n_neighbors = n,
                                               n_jobs = njobs)

        else:
            valid_alg = ['RandUS', 'NearMiss', 'ClusterCentroids', 'TomekLinks',
                         'ENN-all', 'ENN-mode', 'NCR']
            err = f'Invalid algorithm: {algorithm}. Must be one of {valid_alg}'
            raise ValueError(err)

    # Compute under-sampling
        x = self.x_train if x is None else x
        y = self.y_train if y is None else y
        if uds:
            x_bal, y_bal = uds.fit_resample(x, y)
        else:
            x_bal, y_bal = x, y

    # If verbose is True, also return an info tuple
        if verbose:
            info = (algorithm, seed, n, njobs)
            return x_bal, y_bal, info
        else:
            return x_bal, y_bal
        

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


    def balance_trainset(self, strategy: int|str|dict, seed: int, 
                         osa: str|None = None, usa: str|None = None, kos=5,
                         mos=10, nus=3, njobs=1):
        '''
        Run entire not-threaded balancing session on the train subset.

        Parameters
        ----------
        strategy : int | str | dict
            Overall balancing strategy. See parse_balancing_strategy() for more
            details.
        seed : int
            Random seed for reproducible results.
        osa : str | None, optional
            Over-sampling algorithm. See oversample() for a list of possible
            choices. If None, no over-sampling will be performed. The default 
            is None.
        usa : str | None, optional
            Under-sampling algorithm. See undersample() for a list of possible
            choices. If None, no under-sampling will be performed. The default 
            is None.
        kos : int, optional
            Number of k-neighbours to consider in over-sampling algorithm. See
            k parameter in oversample() for more details. The default is 5.
        mos : int, optional
            Number of m-neighbours to consider in over-sampling algorithm. See
            m parameter in oversample() for more details. The default is 10.
        nus : int, optional
            Number of n-neighbours to consider in under-sampling algorithm. See
            m parameter in undersample() for more details. The default is 3.
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
    

    def columns_names(self):
        '''
        Return a list of the dataset columns names.

        Returns
        -------
        list
            Columns names.
            
        '''
        return self.dataframe.columns.to_list()
    

    def column_unique(self, idx: int):
        '''
        Return a list of unique values of column with index 'idx'.

        Parameters
        ----------
        idx : int
            Column index.

        Returns
        -------
        list
            List of sorted unique values.

        '''
        return sorted(self.dataframe.iloc[:, idx].unique().tolist())
    

    def column_count(self, idx: int):
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
        dict
            Counts of unique values -> {unique: count}.

        '''
        return self.dataframe.iloc[:, idx].value_counts().to_dict()
    


class CsvChunkReader():
    '''
    Ready to use class for reading large CSV files in chunks for better
    performance.
    '''

    def __init__(self, dec: str, sep: str|None=None, chunksize=2**20//8):
        '''
        Constructor.

        Parameters
        ----------
        dec : str
            CSV decimal point character.
        sep : str or None, optional
            CSV separator character. If None, it is inferred. The default is 
            None.
        chunksize : int, optional
            Dimension of the reading batch. The default is 2**20//8.

        '''
    # Set main attributes
        self.dec = dec
        self.sep = sep
        self.chunksize = chunksize
        self.thread = threads.CsvChunkReadingThread()


    def set_decimal(self, dec: str):
        '''
        Set CSV decimal point character.

        Parameters
        ----------
        dec : str
            Decimal point character.

        '''
        self.dec = dec


    def set_separator(self, sep: str|None):
        '''
        Set CSV separator character.

        Parameters
        ----------
        sep : str | None
            Separator character. If None, it is inferred.

        '''
        self.sep = sep


    def set_chunksize(self, chunksize: int):
        '''
        Set maximum CSV chunks size.

        Parameters
        ----------
        chunksize : int
            Chunk size.

        '''
        self.chunksize = chunksize


    def chunks_number(self, filepath: str):
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
            The filepath must have the .csv extension.

        '''
        if not filepath.lower().endswith('.csv'):
            raise TypeError('The filepath must have the .csv extension.')
        with open(filepath) as temp:
            n_chunks = sum(1 for _ in temp) // self.chunksize
        return n_chunks


    def read(self, filepath: str):
        '''
        Read CSV file chunk by chunk and return them.

        Parameters
        ----------
        filepath : str
            Path to CSV file. Must have the .csv file extension.

        Returns
        -------
        chunk_list : list
            List of read chunks.

        Raises
        ------
        TypeError
            The filepath must have the .csv extension.

        '''

        if not filepath.lower().endswith('.csv'):
            raise TypeError('The filepath must have the .csv extension.')

        chunk_list = []
        with pd.read_csv(filepath, decimal=self.dec, sep=self.sep, 
                         engine='python', chunksize=self.chunksize) as reader:
            for chunk in reader:
                chunk_list.append(chunk)

        return chunk_list
    

    def read_threaded(self, filepath: str):
        '''
        Run a threaded CSV chunk reading session. 

        Parameters
        ----------
        filepath : str
            Path to CSV file. Must have the .csv file extension.

        Raises
        ------
        TypeError
            The filepath must have the .csv extension.

        '''
        if not filepath.lower().endswith('.csv'):
            raise TypeError('The filepath must have the .csv extension.')

        read = lambda: pd.read_csv(filepath, decimal=self.dec, sep=self.sep, 
                                   engine='python', chunksize=self.chunksize)
        self.thread.set_task(read)
        self.thread.run()


    def combine_chunks(self, chunks: list|tuple):
        '''
        Combine given chunks to reconstruct the full CSV dataframe.

        Parameters
        ----------
        chunks : list | tuple
            List of CSV file chunks.

        Returns
        -------
        Pandas DataFrame
            The reconstructed CSV dataframe.

        '''
        return pd.concat(chunks)
    

    def read_combine(self, filepath: str):
        '''
        Conveninent function to read and return a full CSV dataframe chunk by
        chunk.

        Parameters
        ----------
        filepath : str
            Path to CSV file. Must have the .csv file extension.

        Returns
        -------
        Pandas DataFrame
            The CSV dataframe.

        Raises
        ------
        TypeError
            The filepath must have the .csv extension.

        '''
        if not filepath.lower().endswith('.csv'):
            raise TypeError('The filepath must have the .csv extension.')
        
        chunks = self.read(filepath)
        dataframe = self.combine_chunks(chunks)
        return dataframe



def dataframe_preview(filepath: str, dec: str, sep: str|None=None, n_rows=10):
    '''
    Return a preview of the first 'n_rows' of the dataset stored at the given
    'filepath'.  

    Parameters
    ----------
    filepath : str
        Dataset filepath. Must have the .csv extension. 
    dec : str
        CSV decimal point character.
    sep : str | None, optional
        CSV separator character. If None, it is inferred. The default is None.
    n_rows : int, optional
        Number of rows to be read for the preview. The default is 10.

    Returns
    -------
    preview : Pandas Dataframe
        First 'n_rows' of the dataset.

    '''
    engine = 'python' if sep is None else 'c'
    preview = pd.read_csv(filepath, decimal=dec, sep=sep, nrows=n_rows, 
                          engine=engine) 
    return preview