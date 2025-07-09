# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:52:30 2022

@author: albdag
"""
from collections.abc import Iterable
import os

import numpy as np


class InputMap():

    _DTYPE = 'int32'
    _FILEXT = ('.gz', '.txt')

    def __init__(self, map_array: np.ndarray, filepath: str | None = None) -> None:
        '''
        Base class for generating, editing, loading and saving an input 2d map.
        The object consists of an input map array (e.g., X-Ray map).

        Parameters
        ----------
        map_array : numpy ndarray
            A 2D array storing input map data.
        filepath : str or None, optional
            Path to the stored input map file. For more details, see 'load' and
            'save' methods. The default is None.

        '''

    # Set main attributes
        self.map = map_array.astype(self._DTYPE)
        self.shape = self.map.shape
        self.filepath = filepath


    @classmethod
    def load(cls, filepath: str) -> 'InputMap':
        '''
        Instantiate a new input map object by loading it from a file.

        Parameters
        ----------
        filepath : str
            A valid filepath to the mineral map.

        Returns
        -------
        InputMap
            A new instance of Input Map.

        Raises
        ------
        ValueError
            Raised if the file type is not supported (not one of .gz and .txt).

        '''
        _, filext = os.path.splitext(filepath)

    # Proper file extensions (.gz or .txt) instantiate a new Input Map object
        if filext in cls._FILEXT:
            inmap = np.loadtxt(filepath, dtype=cls._DTYPE)
            return cls(inmap, filepath)

    # Any other file extension is discarded.
        raise ValueError('This file is not supported.')
    

    def copy(self) -> 'InputMap':
        '''
        Return a copy of this input map.

        Returns
        -------
        InputMap
            A copy of this input map.

        '''
        return InputMap(self.map.copy())


    def get_masked(self, mask: np.ndarray) -> np.ma.MaskedArray | np.ndarray:
        '''
        Returns a masked version of the map array.

        Parameters
        ----------
        mask : numpy ndarray
            A boolean array representing the mask (condition).

        Returns
        -------
        map_array_ma : numpy MaskedArray or numpy ndarray
            Masked input map or the original one if 'mask' is invalid. 

        '''
    # Safety: return un-masked version of the map if mask shape is wrong
        if mask.shape != self.shape:
            print(f'Wrong mask shape: {mask.shape}! Mask not applied')
            return self.map
        map_array_ma = np.ma.masked_where(mask, self.map)
        return map_array_ma


    def invert(self) -> None:
        '''
        Set the higher pixel values as the lower and viceversa.

        '''
        self.map = self.map.max() + self.map.min() - self.map


    def save(self, outpath: str) -> None:
        '''
        Save the input map object to disk.

        Parameters
        ----------
        outpath : str
            The input map object is saved to this path. The file extension
            must be one of .gz or .txt.

        Raises
        ------
        ValueError
            Raised if the outpath extension is not one of ('.gz', '.txt').

        '''
        if (ext := os.path.splitext(outpath)[1]) not in self._FILEXT:
            raise ValueError(f'Unsupported file extension: {ext}.')

        np.savetxt(outpath, self.map, delimiter=' ', fmt='%d')
        self.filepath = outpath


class MineralMap():

    _DTYPE_STR = 'U8'
    _DTYPE_INT = 'uint8'
    _FILEXT = ('.mmp', '.gz', '.txt')


    def __init__(
        self,
        minmap_array: np.ndarray, 
        probmap_array: np.ndarray | None = None, 
        palette_dict: dict[int, tuple[int, int, int]] | None = None,
        filepath: str | None = None):
        '''
        Base class for generating, editing, loading and saving a Mineral Map.
        The object consists of a mineral map array, a linked probability map
        array and a palette dictionary.

        Parameters
        ----------
        minmap_array : numpy ndarray
            A 2D array storing mineral map data.
        probmap_array : numpy ndarray or None, optional
            A 2D array storing probability map data. If None, it is initialized
            as a 0's array with shape of 'minmap_array'. The default is None.
        palette_dict : dict[int, tuple[int, int, int]] or None, optional
            A dictionary storing the unique mineral IDs (keys) paired with
            their corresponding color (values) for visualization purposes. The
            colors must be RGB triplets. If None, the palette will be generated
            randomly. The default is None.
        filepath : str or None, optional
            Path to the stored mineral map file. For more details, see 'load'
            and 'save' methods. The default is None.

        '''
    # Define main attributes
        self.minmap = minmap_array.astype(self._DTYPE_STR)
        self.shape = self.minmap.shape
        self.probmap = np.zeros(self.shape) if probmap_array is None else probmap_array
        self.palette = palette_dict
        self.filepath = filepath

    # Initialize derived attributes
        self.encoder = None
        self.minmap_encoded = None
        self.mode = None

    # Update derived attributes
        self.update_derived_attributes()

    # Optional attributes accessed only by unsupervised classifiers
        self._silhouette_avg = None
        self._silhouette_by_cluster = None
        self._chi_score = None
        self._dbi_score = None

    # Set a random palette if the current is empty
        if self.palette is None:
            self.set_palette(self.rand_colorlist())


    def __eq__(self, value: object) -> bool:
        '''
        Reimplementation of equality method.

        Parameters
        ----------
        value : object
            The object to be compared with this mineral map.

        Returns
        -------
        bool
            Whether the two objects contain the same mineral map data.

        '''
        if isinstance(value, MineralMap):
            return (
                np.array_equal(value.minmap, self.minmap)
                and np.array_equal(value.probmap, self.probmap)
                and value.palette == self.palette
            )
        else:
            return False
        
  
    @classmethod
    def load(cls, filepath: str) -> 'MineralMap':
        '''
        Instantiate a new mineral map object by loading it from a file.

        Parameters
        ----------
        filepath : str
            A valid filepath to the mineral map.

        Returns
        -------
        MineralMap
            A new instance of Mineral Map.

        Raises
        ------
        ValueError
            Raised if the file type is not one of .mmp, .gz and .txt.

        '''
        root, filext = os.path.splitext(filepath)

    # Proper file extension (.mmp).
    # From this we can retrieve all the attributes of the Mineral Map object.
        match filext:
            case '.mmp':
                file = np.load(filepath, allow_pickle=True)
                minmap = file['minmap']
                probmap = file['probmap']
                palette = file['palette'].item()

    # Legacy file extensions (.gz, .txt).
    # From these we can retrieve only the mineral map data. Useful to load
    # old (legacy) mineral maps and data from different software.
            case '.gz' | '.txt':
                minmap = np.loadtxt(filepath, dtype=cls._DTYPE_STR)
        # If a legacy-style probability map is present in the same folder, it
        # is loaded as well. The palette is left empty
                probmap = None
                if os.path.exists(probmap_path := root + '_probMap' + filext):
                    probmap = np.loadtxt(probmap_path)
                palette = None

    # Any other file extension is discarded and raises an exception
            case _:
                raise ValueError(f'The file {filepath} is not supported.')

    # Instantiate a new Mineral Map object
        return cls(minmap, probmap, palette, filepath)
    

    def copy(self) -> 'MineralMap':
        '''
        Return a copy of this mineral map.

        Returns
        -------
        MineralMap
            A copy of this mineral map.

        '''
        return MineralMap(self.minmap.copy(), self.probmap.copy(), self.palette.copy())


    def _compile_encoder(
        self,
        unique: np.ndarray[np.str_] | list[str]
    ) -> dict[int, str]:
        '''
        Generate an encoder for the mineral map.

        Parameters
        ----------
        unique : numpy ndarray[numpy str] or list[str]
            List of unique phases (as labels).

        Returns
        -------
        encoder : dict[int, str]
            Encoder dictionary.

        '''
        return dict(zip(range(len(unique)), unique))


    def _compute_mode(
        self,
        unique: np.ndarray[np.str_],
        counts: np.ndarray
    ) -> dict[int, float]:
        '''
        Compute the modal amounts of phases.

        Parameters
        ----------
        unique : numpy ndarray[numpy str]
            Unique phases names (as labels).
        counts : numpy ndarray
            Number of pixels by phase.

        Returns
        -------
        dict[int, float]
            Mode dictionary -> {ID : mode}, ordered by modal abundancy.

        '''
        freq_sort = np.argsort(-counts)
        percent = 100*counts/counts.sum()
        unique, percent = unique[freq_sort], percent[freq_sort]
        unique_id = (self.as_id(u) for u in unique)
        return dict(zip(unique_id, percent))


    def _encode(self, encoder: dict[int, str]) -> np.ndarray:
        '''
        Encode the mineral map to an ID map, especially useful for interacting
        with Matplotlib.

        Parameters
        ----------
        encoder : dict[int, str]
            The encoder dictionary -> {ID : label}.

        Returns
        -------
        numpy ndarray
            The encoded mineral map.

        '''
        encoded_map = self.minmap.copy()
        for k, v in encoder.items():
            encoded_map[self.minmap==v] = k
        return encoded_map.astype(self._DTYPE_INT)
    

    def _with_nodata(self, prob_thresh: float) -> np.ndarray:
        '''
        Get a version of the mineral map array with NoData pixels (i.e., with
        default class '_ND_') based on a threshold. This method is meant to be
        internally called before saving the mineral map to file. 

        Parameters
        ----------
        prob_thresh : float
            The probability threshold.

        Returns
        -------
        minmap_nd : numpy ndarray
            Mineral map array with '_ND_' pixels.

        '''
        minmap_nd = self.minmap.copy()
        minmap_nd[self.probmap < prob_thresh] = '_ND_'
        return minmap_nd


    def as_id(self, phase: str) -> int:
        '''
        Convert phase name (label) to phase ID.

        Parameters
        ----------
        phase : str
            A valid phase name.

        Returns
        -------
        int
            Corresponding phase ID.

        Raises
        ------
        ValueError
            Raised if 'phase' is not a valid phase.

        '''
        if not self.has_phase(phase):
            raise ValueError(f'{phase} is not an occuring phase.')
        
        ids = self.get_phases_ids()
        idx = next(filter(lambda id_: self.encoder[id_] == phase, ids))
        return ids[idx]
            

    def as_phase(self, id_: int) -> str:
        '''
        Convert phase ID to phase name (label).

        Parameters
        ----------
        id_ : int
            A valid phase ID.

        Returns
        -------
        str
            Corresponding phase name.
        
        Raises
        ------
        ValueError
            Raised if 'id_' is not a valid ID.

        '''
        if not self.has_id(id_):
            raise ValueError(f'{id_} is not a valid ID.')
        
        return self.encoder[id_]
    

    def get_clustering_scores(self) -> (
        tuple[
            float | None,
            dict[int, np.ndarray] | None,
            float | None,
            float | None
        ]
    ):
        '''
        Return clustering scores that are computed after an unsupervised
        mineral classification routine. This method is currently only accessed
        by the Mineral Classifier tool.

        Returns
        -------
        float or None
            Average Silhouette score.
        dict[int, numpy ndarray] or None
            Silhouette score by cluster.
        float or None
            Calinski-Harabasz Index.
        float or None
            Davies-Bouldin Index.

        '''
        return (
            self._silhouette_avg,
            self._silhouette_by_cluster, 
            self._chi_score,
            self._dbi_score
        )
    

    def set_clustering_scores(
        self,
        sil_avg: float,
        sil_clust: dict[int, np.ndarray],
        chi: float,
        dbi: float
    ) -> None:
        '''
        Set clustering scores that are computed after an unsupervised mineral
        classification routine. This method is currently only accessed by the 
        Mineral Classifier tool.

        Parameters
        ----------
        sil_avg : float
            Average silhouette score.
        sil_clust : dict[int, numpy ndarray]
            Silhouette scores by cluster.
        chi : float
            Calinski-Harabasz Index.
        dbi : float
            Davies-Bouldin Index.

        '''
        self._silhouette_avg = sil_avg
        self._silhouette_by_cluster = sil_clust
        self._chi_score = chi
        self._dbi_score = dbi


    def get_labeled_mode(self) -> dict[str, float]:
        '''
        Convenient method to get the mode dictionary with keys expressed as
        labels rather than IDs.

        Returns
        -------
        dict[str, float]
            The labeled mode dictionary.

        '''
        ids, mode = zip(*self.mode.items())
        labels = map(lambda id_: self.as_phase(id_), ids)
        return dict(zip(labels, mode))
    

    def get_labeled_palette(self) -> dict[str, tuple[int, int, int]]:
        '''
        Convenient method to get the palette dictionary with keys expressed as
        labels rather than IDs.

        Returns
        -------
        dict[str, tuple[int, int, int]]
            The labeled palette dictionary.

        '''
        ids, colors = zip(*self.palette.items())
        labels = map(lambda id_: self.as_phase(id_), ids)
        return dict(zip(labels, colors))


    def edit_minmap(
        self,
        edited: np.ndarray,
        alter_probmap: bool = False,
        prob_score: float = np.nan
    ) -> None:
        '''
        Apply user's edits to the mineral map. The probability map, the
        encoded mineral map and the palette get automatically updated
        accordingly.

        Parameters
        ----------
        edited : numpy ndarray
            The edited version of the mineral map.
        alter_probmap : bool, optional
            If True, the probability score of edited pixels will be set to 
            'prob_score'. The default is False.
        prob_score: float, optional
            Probability score to be assigned to edited pixels. Ignored if 
            'alter_probmap' is set to False. The default is np.nan.

        '''
    # Ensure new minmap has the correct data type
        edited = edited.astype(self._DTYPE_STR)

    # Alter probability score of edited pixels, if requested 
        if alter_probmap:
            self.probmap[self.minmap != edited] = prob_score

    # Get the original map phases, BEFORE the editing operation
        phases_old = self.get_phases()

    # Apply edits to the mineral map and update the derived attributes
        self.minmap = edited
        self.update_derived_attributes()

    # Get the new map phases, AFTER the editing operation, and then update the
    # palette
        phases_new = self.get_phases()
        self.update_palette(phases_old, phases_new)


    def get_phase_amount(self, phase: str) -> float:
        '''
        Get the amount of a specific phase.

        Parameters
        ----------
        phase : str
            A valid phase name.

        Returns
        -------
        float
            The amount of the phase in the mineral map.

        '''
        id_ = self.as_id(phase)
        return self.mode[id_]


    def get_phase_color(self, phase: str) -> tuple[int, int, int]:
        '''
        Get the color of a specific phase.

        Parameters
        ----------
        phase : str
            A valid phase name.

        Returns
        -------
        tuple[int, int, int]
            RGB triplet.

        '''
        id_ = self.as_id(phase)
        return self.palette[id_]


    def get_phases(self) -> list[str]:
        '''
        Get a list of the mineral phases (as labels).

        Returns
        -------
        list[str]
            List of phases.

        '''
        return list(self.encoder.values())


    def get_phases_ids(self) -> list[int]:
        '''
        Get a list of the mineral phases IDs.

        Returns
        -------
        list[str]
            List of phases IDs.

        '''
        return list(self.encoder.keys())


    def get_plot_data(self) -> (
        tuple[
            np.ndarray,
            dict[int, str],
            Iterable[tuple[int, int, int]]
        ]
    ):
        '''
        Convenient method to get the data required by 'ImageCanvas' class to
        plot the map. See 'draw_discretemap' method of 'ImageCanvas' class for
        details.

        Returns
        -------
        minmap_encoded : numpy ndarray
            The encoded mineral map.
        encoder : dict[int, str]
            The encoder dictionary -> {ID : label}.
        Iterable[tuple[int, int, int]]
            Colors from the palette dictionary, as an iterable of RGB triplets.

        '''

        return (self.minmap_encoded, self.encoder, self.palette.values())


    def get_masked(self, mask: np.ndarray) -> (
        tuple[
            np.ma.MaskedArray | np.ndarray,
            np.ma.MaskedArray | np.ndarray,
            np.ma.MaskedArray | np.ndarray
        ]
    ):
        '''
        Returns a masked version of mineral map, encoded mineral map and
        probabilty map arrays.

        Parameters
        ----------
        mask : numpy ndarray
            A boolean array representing the mask (condition).

        Returns
        -------
        minmap_ma : numpy MaskedArray or numpy ndarray
            Masked mineral map or the original one if 'mask' is invalid.
        minmap_encoded_ma : numpy MaskedArray or numpy ndarray
            Masked encoded mineral map or the original one if 'mask' is invalid.
        probmap_ma : numpy MaskedArray or numpy ndarray
            Masked probability map or the original one if 'mask' is invalid.

        Example
        -------
        phases_to_highlight = ['Amp', 'Grt', 'Pl']
        mask = ~np.isin(MineralMap.minmap, phases_to_highlight)
        minmap_ma, minmap_encoded_ma, probmap_ma = MineralMap.get_masked(mask)

        '''
    # Safety: return un-masked outputs if mask has the wrong shape
        if mask.shape != self.shape:
            print(f'Wrong mask shape: {mask.shape}! Mask not applied')
            return (self.minmap, self.minmap_encoded, self.probmap)
        
    # Return a masked version of mineral map, encoded mineral map and probmap
        minmap_ma = np.ma.masked_where(mask, self.minmap)
        minmap_encoded_ma = np.ma.masked_where(mask, self.minmap_encoded)
        probmap_ma = np.ma.masked_where(mask, self.probmap)
        return (minmap_ma, minmap_encoded_ma, probmap_ma)
    

    def has_id(self, id_: int) -> bool:
        '''
        Check if the given ID is present in the mineral map.

        Parameters
        ----------
        id_ : int
            ID to check.

        Returns
        -------
        bool
            Whether the mineral map contains the given ID.

        '''
        return id_ in self.get_phases_ids()
    

    def has_phase(self, phase: str) -> bool:
        '''
        Check if the given phase is present in the mineral map.

        Parameters
        ----------
        phase : str
            Phase to check.

        Returns
        -------
        bool
            Whether the mineral map contains the given phase.

        '''
        return phase in self.get_phases()


    def is_obsolete(self) -> bool | None:
        '''
        Check if the mineral map file was generated with an obsolete version
        of X-Min Learn.

        Returns
        -------
        bool or None
            True if the mineral map file is obsolete, False otherwise. Returns
            None if the map is not saved to disk.

        '''
        if self.is_stored():
            filext = os.path.splitext(self.filepath)[1]
            return filext != '.mmp'


    def is_stored(self) -> bool:
        '''
        Check if the mineral map object is saved on the disk.

        Returns
        -------
        bool
            True if the mineral map object has a filepath, False otherwise.

        '''
        return self.filepath is not None


    def merge_phases(self, in_list: list[str], out_merged: str) -> None:
        '''
        Merge a list of phases to one merged phase.

        Parameters
        ----------
        in_list : list[str]
            List of phases to be merged (as labels).
        out_merged : str
            Merged phase (as label).

        '''
    # Get the phases list BEFORE the merging operation
        phases_old = self.get_phases()

    # Merge the phases and update derived attributes
        self.minmap[np.isin(self.minmap, in_list)] = out_merged
        self.update_derived_attributes()

    # Get the phases list AFTER the merging operation
        phases_new = self.get_phases()

    # Update the palette
        self.update_palette(phases_old, phases_new)


    def phase_count(self) -> int:
        '''
        Get the number of unique mineral classes.

        Returns
        -------
        int
            Number of phases.

        '''
        return len(self.encoder)


    def rand_colorlist(
        self,
        n_colors: int | None = None,
        tol: int | None = None
    ) -> list[tuple[int, int, int]]:
        '''
        Generate random color lists of desired length.

        Parameters
        ----------
        n_colors : int or None
            Number of colors in the list. If None, the current number of unique
            phases is used. The default is None.
        tol : int, optional
            RNG tolerance parameter. Controls how similar the colors can be.
            Bigger tolerance values means more different colors. If None, it
            is automatically computed as 256 / 'n_colors'. The default is None.

        Returns
        -------
        list[tuple[int, int, int]]
            A list of random RGB triplets.

        '''
        if n_colors is None: 
            n_colors = self.phase_count()

        if tol is None:
            tol = 256//n_colors

        _rgb = np.random.default_rng().integers(256, size=3)
        RGB_arr = _rgb.reshape(1,3)

        for _ in range(n_colors - 1):
            while np.any(np.all(abs(_rgb - RGB_arr) <= tol, axis=1)):
                _rgb = np.random.default_rng().integers(256, size=3)
            RGB_arr = np.r_[RGB_arr, _rgb.reshape(1, 3)]

        return list(map(tuple, RGB_arr))


    def rename_phase(self, old: str, new: str) -> None:
        '''
        Rename a mineral phase from 'old' to 'new'.

        Parameters
        ----------
        old : str
            The old phase name.
        new : str
            The new phase name.

        '''
    # Rename the phase
        self.minmap[self.minmap == old] = new

    # Update the encoder
        id_ = self.as_id(old)
        self.encoder[id_] = new


    def save(self, outpath: str) -> None:
        '''
        Save the mineral map object to disk.

        Parameters
        ----------
        outpath : str
            The mineral map object is saved to this path. It must have the .mmp
            extension.

        Raises
        ------
        ValueError
            Raised if the outpath has not the '.mmp' extension.

        '''
        with open(outpath, 'wb') as op:
            if (ext := os.path.splitext(outpath)[1]) != '.mmp':
                raise ValueError(f'Unsupported file extension: {ext}.')
            
            np.savez(
                file = op,
                minmap = self.minmap,
                probmap = self.probmap,
                palette=self.palette
            ) #!!! add metadata as a dictionary. Should also be done in InputMaps

        self.filepath = outpath


    def set_palette(self, colorlist: list[tuple[int, int, int]]) -> None:
        '''
        Set a palette based on a color list. The length of the list must match
        the number of phases.

        Parameters
        ----------
        colorlist : list[tuple[int, int, int]]
            A list of RGB triplets (e.g. [(0,0,0), (255, 0, 0), ...].

        '''
        IDs = self.get_phases_ids()
        self.palette = dict(zip(IDs, colorlist))


    def set_phase_color(self, phase: str, color: tuple[int, int, int]) -> None:
        '''
        Change the color of a specific phase.

        Parameters
        ----------
        phase : str
            A valid mineral class.
        color : tuple[int, int, int]
            A valid RGB triplet.

        '''
        id_ = self.as_id(phase)
        if self.validate_rgb(color):
            self.palette[id_] = color


    def update_derived_attributes(self) -> None:
        '''
        Refresh all the derived attributes of the mineral map object. It is 
        meant to be internally called whenever the pixel values get altered.

        '''
        unq, cnt = np.unique(self.minmap, return_counts=True)
        self.encoder = self._compile_encoder(unq)
        self.minmap_encoded = self._encode(self.encoder)
        self.mode = self._compute_mode(unq, cnt)


    def update_palette(self, old_phases: list[str], new_phases: list[str]) -> None:
        '''
        Update the palette after mineral map editings, preserving the colors of
        the unaltered phases.

        Parameters
        ----------
        old_phases : list of str
            A list of the phases (as labels) before the editings.
        new_phases : list of str
            A list of the phases (as labels) after the editings.

        '''
    # Compile a temp palette dict, with keys as class labels instead of IDs
        by_phase = dict(zip(old_phases, self.palette.values()))

    # Identify the removed phases and the added ones, using sets
        old, new = set(old_phases), set(new_phases)
        unaltered_phases = old & new
        removed_phases = old - unaltered_phases
        added_phases = new - unaltered_phases

    # Delete colors of removed phases from temp palette
        for r in removed_phases:
            del by_phase[r]

    # Add a new color to temp palette for each new added phase, if any
        if n := len(added_phases):
            new_colors = self.rand_colorlist(n)
            for num, a in enumerate(added_phases):
                by_phase[a] = new_colors[num]

    # Update the definitive palette {ID: color}, correctly sorted by ID
        palette = {self.as_id(k): by_phase[k] for k in by_phase}
        self.palette = {id_: palette[id_] for id_ in sorted(palette)}


    def validate_rgb(self, rgb: tuple[int, int, int]) -> bool:
        '''
        Validate an RGB triplet.

        Parameters
        ----------
        RGB : tuple[int, int, int]
            The RGB triplet.

        Returns
        -------
        bool
            True if the RGB triplet is valid, False otherwise.

        '''
    # If 'rgb' is not a triplet, it is invalid
        if len(rgb) != 3: return False

    # If any of 'rgb' values is not in range 0-255, 'rgb' is invalid
        for channel in rgb:
            if not channel in range(0, 256):
                return False
            
    # Otherwise, 'RGB' is valid
        return True



# class PointAnalysis():
#     pass # define this class for EDS/WDS point analysis
# MUST HAVE ATTRIBUTE self.filepath



class RoiMap():

    _DTYPE = 'U8'
    _ND = ''
    _FILEXT = ('.rmp', )

    def __init__(
        self,
        map_array: np.ndarray,
        roilist: list[list[str, tuple[float, float, int, int]]], 
        filepath: str | None = None
    ) -> None:
        '''
        Base class for generating, editing, loading and saving a ROI Map. The
        object consists of a 2D array with labeled pixels at ROIs locations and
        a list that stores labels and bounding boxes of each ROI.

        Parameters
        ----------
        map_array : numpy ndarray
            The ROI map array.
        roilist : list[list[str, tuple[float, float, int, int]]]
            List of ROIs. A ROI is a list -> [ROI_label, ROI_bbox]. ROI_label
            is a text and ROI_bbox is a tuple -> (x0, y0, width, height), where
            x0, y0 are the coordinates of the top-left corner of the ROI.
        filepath : str or None, optional
            Path to the stored ROI map file. For more details see 'load' and
            'save' methods. The default is None.

        '''
    # Define main attributes
        self.map = map_array.astype(self._DTYPE)
        self.shape = map_array.shape
        self.roilist = roilist
        self.filepath = filepath

    # Initialize derived attributes
        self.class_count = dict()

    # Update derived attributes
        self.update_class_counter()


    def __eq__(self, value: object) -> bool:
        '''
        Reimplementation of equality method.

        Parameters
        ----------
        value : object
            The object to be compared with this ROI map.

        Returns
        -------
        bool
            Whether the two objects contain the same ROI map data.

        '''
        if isinstance(value, RoiMap):
            return (
                np.array_equal(value.map, self.map) 
                and value.roilist == self.roilist
            )
        else:
            return False


    @classmethod
    def load(cls, filepath: str) -> 'RoiMap':
        '''
        Instantiate a new ROI map object by loading it from a file.

        Parameters
        ----------
        filepath : str
            A valid filepath to the ROI map.

        Returns
        -------
        RoiMap
            A new instance of ROI map.

        Raises
        ------
        ValueError
            Raised if the file extension is not supported (not '.rmp').

        '''

        _, filext = os.path.splitext(filepath)

    # Proper file extensions (just .rmp)
        if filext in cls._FILEXT: 
            file = np.load(filepath, allow_pickle=True)
            map_array = file['map_array']
            roilist = file['roilist'].tolist()

    # Any other file extension is discarded and exits the function.
        else:
            raise ValueError('This file is not supported.')

    # Instantiate a new Roi Map object
        return cls(map_array, roilist, filepath)


    @classmethod
    def from_shape(cls, shape: tuple[int, int]) -> 'RoiMap':
        '''
        Instantiate a new empty ROI map of a given shape.

        Parameters
        ----------
        shape : tuple[int, int]
            The shape of the ROI map (rows, cols).

        Returns
        -------
        RoiMap
            A new instance of ROI map.

        '''
        map_array = np.empty(shape, dtype=cls._DTYPE)
        return cls(map_array, [])
    

    def copy(self) -> 'RoiMap':
        '''
        Return a copy of this ROI map.

        Returns
        -------
        RoiMap
            A copy of this ROI map.

        '''
        return RoiMap(self.map.copy(), self.roilist.copy())
    

    def overwrite_roimap(self, other: 'RoiMap', safe: bool = True) -> None:
        '''
        Merge this ROI map with a second one.

        Parameters
        ----------
        other : RoiMap
            A second ROI map.
        safe : bool, optional
            Check every new ROI to exclude overlaps. The default is True.

        Raises
        ------
        ValueError
           Raised if 'other' ROI map has a wrong shape.
            
        '''
        if other.shape != self.shape:
            raise ValueError('The ROI maps have incompatible shapes')
    
    # Safe mode is slower, but prevents overlapping ROIs
        if safe:
            for name, bbox in other.roilist:
                if not self.bbox_overlaps(bbox):
                    self.add_roi([name, bbox])
    
    # Unsafe mode is faster, but can produce overlapping ROIs
        else:
            self.roilist.extend(other.roilist)
            self.map = np.where(other.map != self._ND, other.map, self.map)
            self.update_class_counter()


    def add_roi(self, name: str, bbox: tuple[float, float, int, int]) -> None:
        '''
        Add a new ROI to the ROI map.

        Parameters
        ----------
        name : str
            ROI label.
        bbox : tuple[float, float, int, int]
            ROI bounding box (x0, y0, width, height).

        '''
        roi = [name, bbox]
        self.roilist.append(roi)

    # Update the map
        self.update_map(roi)

    # Update the counter
        self.update_class_counter()


    def del_roi(self, idx: int) -> None:
        '''
        Remove a ROI from the ROI map.

        Parameters
        ----------
        idx : int
            Index of the ROI that has to be removed.

        '''
        bbox = self.roilist.pop(idx)[1]

    # Update the map
        self.update_map([self._ND, bbox])

    # Update the counter
        self.update_class_counter()


    def rename_roi(self, idx: int, new_name: str) -> None:
        '''
        Rename ROI at index 'idx'.

        Parameters
        ----------
        idx : int
            Index of ROI.
        new_name : str
            New name for ROI. Must contain only ASCII characters.

        Raises
        ------
        ValueError
            Raised if 'new_name' contains non-ASCII characters.

        '''
        if not new_name.isascii():
            raise ValueError(f'{new_name} contains non-ASCII characters.')
        
    # Rename the roi
        try:
            self.roilist[idx][0] = new_name
        except IndexError:
            print(f'{idx} is out of range for ROIs list')
            return
        
    # Update the map
        self.update_map((self.roilist[idx]))

    # Update the counter
        self.update_class_counter()


    def bbox_overlaps(self, bbox: tuple[float, float, int, int]) -> bool:
        '''
        Check if a given bounding box overlaps with any existent ROI.

        Parameters
        ----------
        bbox : tuple[float, float, int, int]
            Bounding box.

        Returns
        -------
        overlaps : bool
            Whether the bbox overlaps.

        '''
        extents = self.bbox_to_extents(bbox)
        overlaps = self.extents_overlaps(extents)
        return overlaps
    

    def extents_overlaps(self, extents: tuple[int, int, int, int]) -> bool:
        '''
        Check if the given extents (x0, x1, y0, y1) overlap with existent ROIs.

        Parameters
        ----------
        extents : tuple[int, int, int, int]
            Extents.

        Returns
        -------
        overlaps : bool
            Whether the extents overlap.

        '''
        x0, x1, y0, y1 = extents
        overlaps = np.any(self.map[y0:y1, x0:x1] != self._ND)
        return overlaps


    @staticmethod
    def bbox_to_extents(bbox: tuple[float, float, int, int]) -> (
            tuple[int, int, int, int]
        ):
        '''
        Convert bbox (x0, y0, width, height) to extents (x0, x1, y0, y1).

        Parameters
        ----------
        bbox : tuple[float, float, int, int]
            Bounding box.

        Returns
        -------
        extents : tuple[int, int, int, int]
            Converted extents.

        '''
        x0, y0, w, h = bbox
        x0 = int(x0 + 0.5)
        y0 = int(y0 + 0.5)
        x1 = x0 + w
        y1 = y0 + h
        extents = (x0, x1, y0, y1)
        return extents


    @staticmethod
    def extents_to_bbox(extents: tuple[int, int, int, int]) -> (
            tuple[float, float, int, int]
        ):
        '''
        Convert extents (x0, x1, y0, y1) to bbox (x0, y0, width, height).

        Parameters
        ----------
        extents : tuple[int, int, int, int]
            Extents.

        Returns
        -------
        bbox : tuple[float, float, int, int]
            Bounding box.

        '''
        x0, x1, y0, y1 = extents
        w = x1 - x0
        h = y1 - y0
        x0 -= 0.5
        y0 -= 0.5
        bbox = (x0, y0, w, h)
        return bbox


    @staticmethod
    def bbox_area(bbox: tuple[float, float, int, int]) -> int:
        '''
        Calculate area of a bounding box.

        Parameters
        ----------
        bbox : tuple[float, float, int, int]
            Bounding box.

        Returns
        -------
        area : int
            Calculated area (in pixels).

        '''
        w, h = bbox[-2:]
        area = w * h
        return area


    def update_map(self, roi: list[str, tuple[float, float, int, int]]) -> None:
        '''
        Update the ROI map array. This method is internally called after a ROI
        is added or removed.

        Parameters
        ----------
        roi : list[str, tuple[float, float, int, int]]
            The edited ROI (added or removed).

        '''
    # ROI = [name, bbox]
        name, bbox = roi
        x0, x1, y0, y1 = self.bbox_to_extents(bbox)
    # y = rows, x = cols
        self.map[y0:y1, x0:x1] = name


    def update_class_counter(self) -> None:
        '''
        Update the class counter, a dictionary that counts the pixels assigned
        to each unique label. This method is called after the ROI map array has
        been updated (e.g., after 'update_map' method).

        '''
        unq, cnt = np.unique(self.map, return_counts=True)
        self.class_count = dict(zip(unq, cnt))

    # Exclude nodata from being counted
        self.class_count.pop(self._ND, None)


    def save(self, outpath: str) -> None:
        '''
        Save the ROI map object to disk.

        Parameters
        ----------
        outpath : str
            The ROI map object is saved to this path. It must have the .rmp
            extension.

        Raises
        ------
        ValueError
            Raised if the outpath extension is not '.rmp'.

        '''
        with open (outpath, 'wb') as op:
            if (ext := os.path.splitext(outpath)[1]) not in self._FILEXT:
                raise ValueError(f'Unsupported file extension: {ext}.')
            
            roilist = np.array(self.roilist, object)
            np.savez(op, map_array=self.map, roilist=roilist)

        self.filepath = outpath



class Mask():

    _DTYPE = 'bool'
    _FILEXT = ('.msk', '.txt')

    def __init__(self, mask_array: np.ndarray, filepath: str | None = None) -> None:
        '''
        Base class for generating, editing, loading and saving a mask. The 
        object consists of a boolean 2D array where 1's represent masked (or 
        hidden) pixels.

        Parameters
        ----------
        mask_array : numpy ndarray
            The boolean mask array.
        filepath : str or None, optional
            Filepath to the stored mask file. For more details see 'load' and
            'save' methods. The default is None.

        '''
    # Define main attributes
        self.mask = mask_array.astype(self._DTYPE)
        self.shape = mask_array.shape
        self.filepath = filepath


    @classmethod
    def load(cls, filepath: str) -> 'Mask':
        '''
        Instantiate a new Mask object by loading it from a file.

        Parameters
        ----------
        filepath : str
            A valid filepath to the mask.

        Returns
        -------
        Mask
            A new instance of Mask.

        Raises
        ------
        ValueError
            Raised if the file type is not supported (not .msk or .txt).

        '''
        _, filext = os.path.splitext(filepath)

    # Proper file extension (.msk)
        if filext == '.msk':
            file = np.load(filepath)
            mask_array = file['mask_array']

    # Compatible file extension (.txt)
        elif filext == '.txt':
            mask_array = np.loadtxt(filepath) # hic sunt leones

    # Any other file extension is discarded and exits the function.
        else:
            raise ValueError('This file is not supported.')

    # Instantiate a new Mask object
        return cls(mask_array.astype(cls._DTYPE), filepath)


    @classmethod
    def from_shape(cls, shape: tuple[int, int], fillwith: int = 0) -> 'Mask':
        '''
        Instantiate a new Mask of a given shape.

        Parameters
        ----------
        shape : tuple[int, int]
            The shape of the mask (rows, cols).
        fillwith : int
            Boolean values to fill the mask with. If 0, the mask will be empty
            (i.e., filled with 0's); if any other number, the mask will be full
            (i.e., filled with 1's). The default is 0.

        Returns
        -------
        Mask
            A new instance of Mask.

        '''
        if fillwith == 0:
            mask_array = np.zeros(shape, dtype=cls._DTYPE)
        else:
            mask_array = np.ones(shape, dtype=cls._DTYPE)

        return cls(mask_array)


    def invert(self) -> None:
        '''
        Invert the mask.

        '''
        self.mask = ~self.mask


    def invert_region(self, extents: tuple[int, int, int, int]) -> None:
        '''
        Invert a region of the mask.

        Parameters
        ----------
        extents : tuple[int, int, int, int]
            Coordinates of the region to invert -> (x0, x1, y0, y1).

        '''
        x0, x1, y0, y1 = extents
        self.mask[y0:y1, x0:x1] = ~self.mask[y0:y1, x0:x1]


    def save(self, outpath: str) -> None:
        '''
        Save the Mask object to disk.

        Parameters
        ----------
        outpath : str
            The Mask object is saved to this path. It must have the .msk
            extension.

        Raises
        ------
        ValueError
            Raised if the outpath extension is not '.msk'.

        '''
        with open (outpath, 'wb') as op:
            if (ext := os.path.splitext(outpath)[1]) != '.msk':
                raise ValueError(f'Unsupported file extension: {ext}.')
            np.savez(op, mask_array=self.mask)

        self.filepath = outpath



class InputMapStack():

    def __init__(self, input_maps: list[InputMap], mask: Mask | None = None) -> None:
        '''
        Base class for layering 2d Input Maps of same shape together into a
        single 3D stack. A mask, if provided, is applied to the entire stack.
        This class also includes several convenient methods for the stack
        classification and management.

        Parameters
        ----------
        input_maps : list[InputMap]
            List of Input Maps. They must be stackable (same shape).
        mask : Mask or None, optional
            Mask to apply to the entire stack. The default is None.

        Raises
        ------
        ValueError
            Raised if 'input_maps' does not contain InputMap objects. 

        '''
    # Make sure that input maps are in a list (no other iterable is allowed)
        if not isinstance(input_maps, list):
            input_maps = list(input_maps)

    # Make sure that input maps list contains only InputMap objects
        if not all(isinstance(i, InputMap) for i in input_maps):
            raise ValueError('"input_maps" must contain only InputMaps.')

    # Set main attributes
        self._stack = None
        self.input_maps = input_maps
        self.arrays = [m.map for m in input_maps]
        self.mask = mask
        if self.mask is not None:
            self._set_mask()


    def __len__(self) -> int:
        '''
        Return the length of the stack.

        Returns
        -------
        int
            Number of maps included in the stack.

        '''
        return len(self.arrays)
    

    def _set_mask(self) -> None:
        '''
        Internal method that applies a mask to the entire stack.

        '''
        if self.maps_fit() and self.mask_fit():
            msk = self.mask.mask
            self.arrays = [np.ma.masked_where(msk, a) for a in self.arrays]
    

    @property
    def maps_shape(self) -> tuple[int, int] | None:
        '''
        Return the shape of one slice of the stack (2D map shape).

        Returns
        -------
        tuple or None
            2D slice shape or None if maps in the stack have different shapes.

        '''
        if self.maps_fit():
            return self.arrays[0].shape
        else:
            return None
        

    @property
    def stack_shape(self) -> tuple[int, int, int] | None:
        '''
        Return the shape of the entire stack (3D shape).

        Returns
        -------
        tuple[int, int, int] or None
            3D stack shape or None if maps in the stack have different shapes 
            or mask does not fit the stack.

        '''
        if self.maps_fit() and self.mask_fit():
            return self.stack.shape
        else:
            return None

        
    @property
    def stack(self) -> np.ndarray:
        '''
        Return the stack.

        Returns
        -------
        numpy ndarray
            The 3D maps stack.

        Raises
        ------
        ValueError
            Raised if any input map and/or mask have incompatible shapes.

        '''
        if self._stack is None:
            if not self.maps_fit() or not self.mask_fit():
                raise ValueError('Inputs do not share the same size.')
            else:
                self._stack = np.stack(self.arrays)
        
        return self._stack


    def add_maps(self, maps: list[InputMap]):
        '''
        Add new maps to the stack.

        Parameters
        ----------
        maps : list[InputMap]
            List of InputMaps

        Raises
        ------
        ValueError
            Raised if 'maps' does not contain InputMap objects. 

        '''
    # Make sure that the list contains only InputMaps
        if all(isinstance(m, InputMap) for m in maps):
            self.input_maps.extend(maps)
            self.__init__(self.input_maps, self.mask)
        else:
            raise ValueError('"maps" must contain only InputMaps.')


    def maps_fit(self) -> bool:
        '''
        Check if all maps have same shape.

        Returns
        -------
        fit : bool
            Whether the arrays share the same shape.

        '''
        fit = all([a.shape == self.arrays[0].shape for a in self.arrays])
        return fit
    
    
    def mask_fit(self) -> bool:
        '''
        Check if the provided maks has the same shape of the input maps.

        Returns
        -------
        fit : bool
            Whether the mask and input maps share the same shape.

        '''
        if self.mask is None:
            fit = True
        else:
            fit = self.mask.shape == self.maps_shape
        return fit
    

    def get_feature_array(self) -> np.ndarray:
        '''
        Return a flatten/compressed and transposed version of the stack, which
        is required by classification algorithms.

        Returns
        -------
        feat_array : numpy ndarray
            The feature array.

        Raises
        ------
        ValueError
            Raised if any input maps and/or mask have incompatible shapes.

        '''
    # Raise error if maps do not share the same shape
        if not self.maps_fit() or not self.mask_fit():
            raise ValueError('Inputs do not share the same shape.')
        
    # If maps are masked, make them 1D with compressed() otherwise use flatten()
        if self.mask is None:
            feat_array = np.vstack([a.flatten() for a in self.arrays]).T
        else:
            feat_array = np.vstack([a.compressed() for a in self.arrays]).T

        return feat_array
    

    def reorder(self, index_list: list[int]) -> None:
        '''
        Sort the stack using a list of indices.

        Parameters
        ----------
        index_list : list[int]
            List of indices.

        '''
        self.input_maps[:] = [self.input_maps[i] for i in index_list]
        self.arrays[:] = [self.arrays[i] for i in index_list]
        self._stack = np.stack(self.arrays)