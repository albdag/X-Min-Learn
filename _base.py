# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:52:30 2022

@author: albdag
"""
import os

import numpy as np


class InputMap():
    '''
    Base class for generating, editing, loading and saving an input 2d map. The
    object consists of an input map array (e.g., X-Ray map).
    '''

    def __init__(self, map_array: np.ndarray, filepath: str|None=None):
        '''
        Input Map class constructor.

        Parameters
        ----------
        map_array : NumPy ndarray
            A 2D numpy array storing the mineral map data.
        filepath : str or None, optional
            Filepath to the stored input map file. See load and save functions
            for more details. The default is None.

        '''

    # Define the principal attributes
        self.map = map_array
        self.shape = self.map.shape
        self.filepath = filepath


    @classmethod
    def load(cls, filepath: str):
        '''
        Instantiate a new input map object by loading it from a file.

        Parameters
        ----------
        filepath : str
            A valid filepath to the mineral map.

        Raises
        ------
        TypeError
            The file type is not supported (not one of .gz and .txt).

        Returns
        -------
        Input Map object
            A new instance of Input Map.

        '''
        _, filext = os.path.splitext(filepath)

    # Proper file extensions (.gz or .txt) instantiate a new Input Map object
        if filext in ('.gz', '.txt'):
            inmap = np.loadtxt(filepath, dtype='int32')
            return cls(inmap, filepath)

    # Any other file extension is discarded.
        raise TypeError('This file is not supported.')


    def get_masked(self, mask: np.ndarray):
        '''
        Returns a masked version of the input map.

        Parameters
        ----------
        mask : numpy ndarray
            A boolean array representing the mask (condition).

        Returns
        -------
        map_array_ma : numpy masked ndarray
            The masked input map.

        '''
    # Safety: return un-masked version of the map if mask shape is wrong
        if mask.shape != self.shape:
            print(f'Wrong mask shape: {mask.shape}! Mask not applied')
            return self.map
        map_array_ma = np.ma.masked_where(mask, self.map)
        return map_array_ma


    def invert(self):
        '''
        Set the higher pixel values as the lower and viceversa.

        '''
        self.map = self.map.max() + self.map.min() - self.map


    def save(self, outpath: str):
        '''
        Save the input map object to disk.

        Parameters
        ----------
        outpath : str
            The input map object is saved to this path. The file extension
            must be one of .gz or .txt.

        Raises
        ------
        AssertionError
            The outpath extension is not one of ('.gz', '.txt').

        '''
        assert os.path.splitext(outpath)[1] in ('.gz', '.txt')
        np.savetxt(outpath, self.map, delimiter=' ', fmt='%d')
        self.filepath = outpath


class MineralMap():
    '''
    Base class for generating, editing, loading and saving a Mineral Map. The
    object consists of a mineral map array, a linked probability map array and
    a palette dictionary.
    '''

    _DTYPE_STR = 'U8'
    _DTYPE_INT = 'uint8'


    def __init__(self, minmap_array: np.ndarray, 
                 probmap_array: np.ndarray|None=None, 
                 palette_dict: dict|None=None, filepath: str|None=None):
        '''
        Mineral Map class constructor.

        Parameters
        ----------
        minmap_array : NumPy ndarray
            A 2D numpy array storing the mineral map data.
        probmap_array : NumPy ndarray or None, optional
            A 2D numpy array storing the probability map data. If None, it is
            initialized as a zeros array of the same shape as <minmap_array>.
            The default is None.
        palette_dict : dict or None, optional
            A dictionary storing the unique mineral IDs (keys) paired with
            their corresponding color (values) for visualization purposes. The
            colors must be RGB tuples. If None, the palette will be generated
            randomly. The default is None.
        filepath : str or None, optional
            Filepath to the stored mineral map file. See load and save
            functions for more details. The default is None.

        '''
    # Define the principal attributes
        self.minmap = minmap_array
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

    # Optional attributes set only by (unsupervised) classifiers
        self._silhouette_avg = None
        self._silhouette_by_cluster = None
        self._chi_score = None
        self._dbi_score = None

    # Set a random palette if the current is empty
        if self.palette is None:
            self.set_palette(self.rand_colorlist())


    @classmethod
    def load(cls, filepath: str):
        '''
        Instantiate a new mineral map object by loading it from a file.

        Parameters
        ----------
        filepath : str
            A valid filepath to the mineral map.

        Raises
        ------
        TypeError
            The file type is not supported (not one of .mmp, .gz and .txt).

        Returns
        -------
        Mineral Map object
            A new instance of Mineral Map.

        '''
        root, filext = os.path.splitext(filepath)

    # Proper file extension (.mmp).
    # From this we can retrieve all the attributes of the Mineral Map object.
        if filext == '.mmp':
            file = np.load(filepath, allow_pickle=True)
            minmap = file['minmap']
            probmap = file['probmap']
            palette = file['palette'].item()

    # Legacy file extensions (.gz, .txt).
    # From these we can retrieve only the mineral map data. Useful to load
    # old (legacy) mineral maps and data from different software.
        elif filext in ('.gz', '.txt'):
            minmap = np.loadtxt(filepath, dtype='U8')

        # If a legacy-style probability map is present in the same folder, it
        # is loaded as well.
            probmap = None
            if os.path.exists(probmap_path := root + '_probMap' + filext):
                probmap = np.loadtxt(probmap_path)

        # The palette is left empty.
            palette = None

    # Any other file extension is discarded and exits the function.
        else: raise TypeError('This file is not supported.')

    # Instantiate a new Mineral Map object
        return cls(minmap, probmap, palette, filepath)


    def _compile_encoder(self, unique: np.ndarray|list):
        '''
        Generate an encoder for the mineral map.

        Parameters
        ----------
        unique : ndarray or list
            List of unique phases (as labels).

        Returns
        -------
        encoder : dict
            Encoder dictionary.

        '''
        return dict(zip(range(len(unique)), unique))


    def _compute_mode(self, unique: np.ndarray, counts: np.ndarray):
        '''
        A function to compute the modal amounts of phases.

        Parameters
        ----------
        unique : NumPy ndarray
            Unique phases names (as labels).
        counts : NumPy ndarray
            Number of pixels by phase.

        Returns
        -------
        dict
            Mode dictionary -> {ID : mode}, ordered by modal abundancy.

        '''
        freq_sort = np.argsort(-counts)
        percent = 100*counts/counts.sum()
        unique, percent = unique[freq_sort], percent[freq_sort]
        uniqueID = (self.as_id(u) for u in unique)
        return dict(zip(uniqueID, percent))


    def _encode(self, encoder: dict):
        '''
        Encode the mineral map to an ID map, especially useful for the
        interaction with matplotlib.

        Parameters
        ----------
        encoder : dict
            The encoder dictionary -> {ID : label}.

        Returns
        -------
        NumPy ndarray
            The encoded mineral map.

        '''
        encoded_map = self.minmap.copy()
        for k, v in encoder.items():
            encoded_map[self.minmap==v] = k
        return encoded_map.astype(self._DTYPE_INT)
    

    def _with_nodata(self, prob_thresh: float):
        '''
        Get a version of the mineral map with NoData pixels based on a 
        confidence (a.k.a. probability) threshold. This function is meant to be
        internally called before saving the mineral map to file. 

        Parameters
        ----------
        prob_thresh : float
            The probability threshold.

        '''
        minmap_nd = self.minmap.copy()
        minmap_nd[self.probmap < prob_thresh] = '_ND_'
        return minmap_nd


    def as_id(self, phase: str):
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

        '''
        IDs = self.get_phases_ids()
        try:
            idx = next(filter(lambda k: self.encoder[k]==phase, IDs))
            return IDs[idx]
        except StopIteration:
            raise ValueError(f'{phase} is not an occuring phase.')


    def as_phase(self, id_: int):
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

        '''
        if id_ > len(self.encoder) - 1:
            raise ValueError(f'{id_} is not a valid ID.')
        return self.encoder[id_]
    

    def get_clustering_scores(self):
        '''
        Return clustering scores that are computed after an unsupervised
        mineral classification routine. This function is currently only 
        accessed by the Mineral Classifier tool.

        Returns
        -------
        tuple
            Clustering scores (= average silhouette score, silhouette score by
            cluster, Calinski-Harabasz Index, Davies-Bouldin Index).
        '''
        return (self._silhouette_avg, self._silhouette_by_cluster, 
                self._chi_score, self._dbi_score)
    

    def set_clustering_scores(self, sil_avg: float, sil_clust: dict, 
                              chi: float, dbi: float):
        '''
        Set clustering scores that are computed after an unsupervised mineral
        classification routine. This function is currently only accessed by the 
        Mineral Classifier tool.

        Parameters
        ----------
        sil_avg : float
            Average silhouette score.
        sil_clust : dict
            Silhouette score by cluster.
        chi : float
            Calinski-Harabasz Index.
        dbi : float
            Davies-Bouldin Index.
        '''
        self._silhouette_avg = sil_avg
        self._silhouette_by_cluster = sil_clust
        self._chi_score = chi
        self._dbi_score = dbi


    def get_labeled_mode(self):
        '''
        Convenient function to get the mode dictionary with keys expressed as
        labels rather than IDs.

        Returns
        -------
        dict
            The labeled mode dictionary.

        '''
        ids, mode = zip(*self.mode.items())
        labels = map(lambda id_: self.as_phase(id_), ids)
        return dict(zip(labels, mode))
    

    def get_labeled_palette(self):
        '''
        Convenient function to get the palette dictionary with keys expressed 
        as labels rather than IDs.

        Returns
        -------
        dict
            The labeled palette dictionary.

        '''
        ids, colors = zip(*self.palette.items())
        labels = map(lambda id_: self.as_phase(id_), ids)
        return dict(zip(labels, colors))


    def edit_minmap(self, new_minmap: np.ndarray, alter_probmap=False):
        '''
        Apply user's edits to the mineral map. The probability map, the
        encoded mineral map and the palette get automatically updated
        accordingly.

        Parameters
        ----------
        new_minmap : NumPy ndarray
            The new edited mineral map.

        alter_probmap : bool, optional
            If True, the probability score of edited pixels will be set to 1.
            The default is False.

        '''
    # Set probability score of edited pixels to 1, if requested 
        if alter_probmap:
            self.probmap[self.minmap != new_minmap] = 1

    # Get the original map phases, BEFORE the editing operation
        old_phases = self.get_phases()

    # Apply edits to the mineral map and update the derived attributes
        self.minmap = new_minmap
        self.update_derived_attributes()

    # Get the new map phases, AFTER the editing operation, and then update the
    # palette
        new_phases = self.get_phases()
        self.update_palette(old_phases, new_phases)


    def get_phase_amount(self, phase: str):
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


    def get_phase_color(self, phase: str):
        '''
        Get the color of a specific phase.

        Parameters
        ----------
        phase : str
            A valid phase name.

        Returns
        -------
        tuple of int
            RGB triplet.

        '''
        id_ = self.as_id(phase)
        return self.palette[id_]


    def get_phases(self):
        '''
        Get a list of the mineral phases (as labels).

        Returns
        -------
        list of str
            List of phases.

        '''
        return list(self.encoder.values())


    def get_phases_ids(self):
        '''
        Get a list of the mineral phases IDs.

        Returns
        -------
        list of int
            List of phases IDs.

        '''
        return list(self.encoder.keys())


    def get_plot_data(self):
        '''
        Convenient function to get the data required by ImageCanvas object to
        plot the map. See plots.ImageCanvas.draw_discretemap() for further
        details.

        Returns
        -------
        minmap_encoded : NumPy ndarray
            The encoded mineral map.
        encoder : dict
            The encoder dictionary -> {ID : label}.
        palette.values() : dict values
            The colors from the palette, as an iterable of RGB tuples.

        '''

        return (self.minmap_encoded, self.encoder, self.palette.values())


    def get_masked(self, mask: np.ndarray):
        '''
        Returns a masked version of mineral map, encoded mineral map and
        probabilty map.

        Parameters
        ----------
        mask : NumPy ndarray
            A boolean array representing the mask (condition).

        Returns
        -------
        minmap_ma : Masked NumPy ndarray
            The masked mineral map.
        minmap_encoded_ma : Masked NumPy ndarray
            The masked encoded mineral map.
        probmap_ma : Masked NumPy ndarray
            The masked probability map.

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


    def is_obsolete(self):
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


    def is_stored(self):
        '''
        Check if the mineral map object is saved on the disk.

        Returns
        -------
        bool
            True if the mineral map object has a filepath, False otherwise.

        '''
        return self.filepath is not None


    def merge_phases(self, in_list: list[str], out_merged: str):
        '''
        A function to merge a list of phases to one merged phase.

        Parameters
        ----------
        in_list : list of str
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


    def phase_count(self):
        '''
        Get the number of unique mineral classes.

        Returns
        -------
        int
            Number of phases.

        '''
        return len(self.encoder)


    def rand_colorlist(self, n_colors: int|None=None, tol: int|None=None):
        '''
        A function to generate random color lists of desired length.

        Parameters
        ----------
        n_colors : int or None
            Number of random colors in the list (a.k.a. list length). If None
            it is the current number of unique phases. The default is None.
        tol : int, optional
            RNG tolerance parameter. Controls how similar the colors can be.
            Bigger tolerance values means more different colors. If None, it
            is automatically computed as 256 / n_colors. The default is None.

        Returns
        -------
        list of tuples
            A list of random RGB triplets.

        '''
        if n_colors is None: n_colors = self.phase_count()

        if tol is None: tol = 256//n_colors

        _rgb = np.random.randint(256, size=3)
        RGB_arr = _rgb.reshape(1,3)

        for x in range(n_colors-1):
            while np.any(np.all(abs(_rgb - RGB_arr) <= tol, axis=1)):
                _rgb = np.random.randint(256, size=3)
            RGB_arr = np.r_[RGB_arr, _rgb.reshape(1,3)]

        return RGB_arr.tolist()


    def rename_phase(self, old: str, new: str):
        '''
        Rename a mineral phase from <old> to <new>.

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


    def save(self, outpath: str):
        '''
        Save the mineral map object to disk.

        Parameters
        ----------
        outpath : str
            The mineral map object is saved to this path. It must have the .mmp
            extension.

        Raises
        ------
        AssertionError
            The outpath has not the '.mmp' extension.

        '''
        with open (outpath, 'wb') as op:
            assert os.path.splitext(outpath)[1] == '.mmp'
            np.savez(op, minmap=self.minmap,
                         probmap=self.probmap,
                         palette=self.palette) #!!! add other properties (like the name) as attributes? Could be saved as a dictionary. Should also be done in InputMaps
        self.filepath = outpath


    def set_palette(self, colorlist: list[tuple]):
        '''
        Set a palette based on a color list. The length of the list must match
        the number of phases.

        Parameters
        ----------
        colorlist : list of tuples
            A list of RGB triplets (e.g. [(0,0,0), (255, 0, 0), ...].

        '''
        IDs = self.get_phases_ids()
        self.palette = dict(zip(IDs, colorlist))


    def set_phase_color(self, phase: str, color: tuple[int]):
        '''
        Change the color of a specific phase.

        Parameters
        ----------
        phase : str
            A valid mineral class.
        color : tuple of int
            A valid RGB triplet.

        '''
        id_ = self.as_id(phase)
        if self.validate_rgb(color):
            self.palette[id_] = color


    def update_derived_attributes(self):
        '''
        A function to refresh all the derived attributes of the mineral map
        object. It is meant to be internally called whenever the pixel values
        get altered.

        '''
        unq, cnt = np.unique(self.minmap, return_counts=True)
        self.encoder = self._compile_encoder(unq)
        self.minmap_encoded = self._encode(self.encoder)
        self.mode = self._compute_mode(unq, cnt)


    def update_palette(self, old_phases: list[str], new_phases: list[str]):
        '''
        A function to update the palette after mineral map editings. It
        preserves the colors of the unaltered phases.

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

    # Update the definitive palette <{ID:color}>, correctly sorted by ID
        palette = {self.as_id(k): by_phase[k] for k in by_phase}
        self.palette = {id_: palette[id_] for id_ in sorted(palette)}


    def validate_rgb(self, rgb: tuple[int]):
        '''
        A function to validate an RGB triplet.

        Parameters
        ----------
        RGB : tuple of int
            The RGB triplet.

        Returns
        -------
        bool
            True if the RGB triplet is valid, False otherwise.

        '''
    # If <RGB> is not a triplet, it is invalid
        if len(rgb) != 3: return False
    # If any of <RGB> values is not in range 0-255, <RGB> is invalid
        for channel in rgb:
            if not channel in range(0, 256):
                return False
    # Else, <RGB> is valid
        return True


# class PointAnalysis():
#     pass # define this class for EDS/WDS point analysis
# MUST HAVE ATTRIBUTE self.filepath




class RoiMap():
    '''
    Base class for generating, editing, loading and saving a ROI Map. The
    object consists of a 2D array with labeled pixels at ROIs locations and a
    list that stores the label and the bounding box of each ROI.
    '''

    _DTYPE_STR = 'U8'
    _ND = ''

    def __init__(self, map_array: np.ndarray, roilist: list[list[str, tuple]], 
                 filepath:str|None=None):
        '''
        RoiMap class constructor.

        Parameters
        ----------
        map_array : ndarray
            The ROI map array.
        roilist : list of list[str, tuple]
            List of ROIs. A ROI is a list -> [ROI_label, ROI_bbox]. ROI_label
            is a text and ROI_bbox is a tuple -> (x0, y0, width, height), where
            x0, y0 are the coordinates of the top-left corner of the ROI.
        filepath : str or None, optional
            Filepath to the stored ROI map file. See load and save functions
            for more details. The default is None.

        '''
    # Make sure that the map has the correct dtype (string <= 8 characters)
        assert map_array.dtype == self._DTYPE_STR

    # Define the principal attributes
        self.map = map_array
        self.shape = map_array.shape
        self.roilist = roilist
        self.filepath = filepath

    # Initialize derived attributes
        self.class_count = dict()

    # Update derived attributes
        self.update_class_counter()


    @classmethod
    def load(cls, filepath: str):
        '''
        Instantiate a new ROI map object by loading it from a file.

        Parameters
        ----------
        filepath : str
            A valid filepath to the ROI map.

        Raises
        ------
        TypeError
            The file extension is not supported (not '.rmp').

        Returns
        -------
        RoiMap
            A new instance of ROI map.

        '''

        _, filext = os.path.splitext(filepath)

    # Proper file extension (.rmp)
        if filext == '.rmp':
            file = np.load(filepath, allow_pickle=True)
            map_array = file['map_array']
            roilist = file['roilist'].tolist()

    # Any other file extension is discarded and exits the function.
        else: raise TypeError('This file is not supported.')

    # Instantiate a new Roi Map object
        return cls(map_array, roilist, filepath)


    @classmethod
    def from_shape(cls, shape: tuple):
        '''
        Instantiate a new empty ROI map of a given shape.

        Parameters
        ----------
        shape : tuple
            The shape of the ROI map (rows, cols).

        Returns
        -------
        RoiMap
            A new instance of ROI map.

        '''
        map_array = np.empty(shape, dtype='U8')
        return cls(map_array, [])
    

    def copy(self):
        '''
        Return a copy of this ROI map.

        Returns
        -------
        RoiMap
            A copy of this ROI map.

        '''
        return RoiMap(self.map.copy(), self.roilist.copy())
    

    def overwrite_roimap(self, other, safe=True):
        '''
        Merge this ROI map with a second one.

        Parameters
        ----------
        other : RoiMap
            The second ROI map.
        safe : bool, optional
            Check every new ROI to exclude overlaps. The default is True.

        Raises
        ------
        ValueError
            The second ROI map has a wrong shape.
            
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


    def add_roi(self, name: str, bbox: tuple):
        '''
        Add a new ROI to the ROI map.

        Parameters
        ----------
        name : str
            ROI label.
        bbox : tuple
            ROI bounding box (x0, y0, width, height).

        '''
        roi = [name, bbox]
        self.roilist.append(roi)
    # Update the map
        self.update_map(roi)
    # Update the counter
        self.update_class_counter()


    def del_roi(self, idx: int):
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


    def rename_roi(self, idx: int, new_name: str):
        '''
        Rename ROI at index <idx>.

        Parameters
        ----------
        idx : int
            Index of ROI.
        new_name : str
            New name for ROI. Must contain only ASCII characters.

        Raises
        ------
        ValueError
            Name contains non-ASCII characters.

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


    def bbox_overlaps(self, bbox: tuple):
        '''
        Check if a given bounding box overlaps with any existent ROI.

        Parameters
        ----------
        bbox : tuple
            Bounding box.

        Returns
        -------
        overlaps : bool
            Whether the bbox overlaps.

        '''
        extents = self.bbox_to_extents(bbox)
        overlaps = self.extents_overlaps(extents)
        return overlaps
    

    def extents_overlaps(self, extents: tuple):
        '''
        Check if the given extents (x0,x1, y0,y1) overlap with existent ROIs.

        Parameters
        ----------
        extents : tuple
            Extents.

        Returns
        -------
        overlaps : bool
            Whether the extents overlap.

        '''
        x0, x1, y0, y1 = extents
        overlaps = np.any(self.map[y0:y1, x0:x1] != self._ND)
        return overlaps



    def bbox_to_extents(self, bbox: tuple):
        '''
        Convert bounding box (x0, y0, width, height) to extents (x0,x1, y0,y1).

        Parameters
        ----------
        bbox : tuple
            Bounding box.

        Returns
        -------
        extents : tuple
            Converted extents.

        '''
        x0, y0, w, h = bbox
        x0 = int(x0 + 0.5)
        y0 = int(y0 + 0.5)
        x1 = x0 + w
        y1 = y0 + h
        extents = (x0, x1, y0, y1)
        return extents

    def extents_to_bbox(self, extents: tuple):
        '''
        Convert extents (x0,x1, y0,y1) to bounding box (x0, y0, width, height).

        Parameters
        ----------
        extents : tuple
            Extents.

        Returns
        -------
        bbox : tuple
            Bounding box.

        '''
        x0, x1, y0, y1 = extents
        w = x1 - x0
        h = y1 - y0
        x0 -= 0.5
        y0 -= 0.5
        bbox = (x0, y0, w, h)
        return bbox


    def bbox_area(self, bbox: tuple):
        '''
        Calculate area of a bounding box.

        Parameters
        ----------
        bbox : tuple
            Bounding box.

        Returns
        -------
        area : int
            Calculated area (in pixels).

        '''
        w, h = bbox[-2:]
        area = w * h
        return area


    def update_map(self, roi: list[str, tuple]):
        '''
        Update the ROI map array. This function is called after a ROI is added
        or removed.

        Parameters
        ----------
        roi : list[str, tuple]
            The edited ROI (added or removed).

        '''
    # ROI = [name, bbox]
        name, bbox = roi
        x0, x1, y0, y1 = self.bbox_to_extents(bbox)
    # y = rows, x = cols
        self.map[y0:y1, x0:x1] = name


    def update_class_counter(self):
        '''
        Update the class counter, a dictionary that counts the pixels assigned
        to each unique label. This function is called after the ROI map array
        has been updated (e.g. after update_map).

        '''
        unq, cnt = np.unique(self.map, return_counts=True)
        self.class_count = dict(zip(unq, cnt))
    # Exclude nodata from being counted
        self.class_count.pop(self._ND, None)


    def save(self, outpath: str):
        '''
        Save the ROI map object to disk.

        Parameters
        ----------
        outpath : str
            The ROI map object is saved to this path. It must have the .rmp
            extension.

        Raises
        ------
        AssertionError
            The outpath extension is not '.rmp'.

        '''
        with open (outpath, 'wb') as op:
            assert os.path.splitext(outpath)[1] == '.rmp'
            roilist = np.array(self.roilist, object)
            np.savez(op, map_array=self.map, roilist=roilist)
        self.filepath = outpath



class Mask():
    '''
    Base class for generating, editing, loading and saving a mask. The object
    consists of a boolean 2D array where 1's represent masked (hidden) pixels.
    '''

    def __init__(self, mask_array: np.ndarray, filepath: str|None=None):
        '''
        Mask class constructor.

        Parameters
        ----------
        mask_array : ndarray
            The boolean mask array.
        filepath : str or None, optional
            Filepath to the stored mask file. See load and save functions for
            more details. The default is None.

        '''
    # Make sure that the map has the correct dtype (boolean)
        mask_array = mask_array.astype(bool)

    # Define the principal attributes
        self.mask = mask_array
        self.shape = mask_array.shape
        self.filepath = filepath


    @classmethod
    def load(cls, filepath: str):
        '''
        Instantiate a new Mask object by loading it from a file.

        Parameters
        ----------
        filepath : str
            A valid filepath to the mask.

        Raises
        ------
        TypeError
            The file type is not supported (not .msk or .txt).

        Returns
        -------
        Mask
            A new instance of Mask.

        '''

        _, filext = os.path.splitext(filepath)

    # Proper file extension (.msk)
        if filext == '.msk':
            file = np.load(filepath)
            mask_array = file['mask_array']

    # Compatible file extension (.txt)
        elif filext == '.txt':
            mask_array = np.loadtxt(filepath, dtype=bool) # hic sunt leones

    # Any other file extension is discarded and exits the function.
        else: raise TypeError('This file is not supported.')

    # Instantiate a new Mask object
        return cls(mask_array, filepath)


    @classmethod
    def from_shape(cls, shape: tuple, fillwith=0):
        '''
        Instantiate a new Mask of a given shape.

        Parameters
        ----------
        shape : tuple
            The shape of the mask (rows, cols).
        fillwith : bool
            Whether the mask should be empty (0's) or full (1's). The default
            is 0.

        Returns
        -------
        Mask
            A new instance of Mask.

        '''
        if fillwith == 0:
            mask_array = np.zeros(shape, dtype=bool)
        else:
            mask_array = np.ones(shape, dtype=bool)

        return cls(mask_array)


    def invert(self):
        '''
        Invert the mask.

        '''
        self.mask = ~self.mask


    def invert_region(self, extents: tuple):
        '''
        Invert just a portion of the mask.

        Parameters
        ----------
        extents : tuple
            Coordinates of the region to invert -> (x0, x1, y0, y1).

        '''
        x0, x1, y0, y1 = extents
        self.mask[y0:y1, x0:x1] = ~self.mask[y0:y1, x0:x1]


    def save(self, outpath: str):
        '''
        Save the Mask object to disk.

        Parameters
        ----------
        outpath : path-like or str
            The Mask object is saved to this path. It must have the .msk
            extension.

        Raises
        ------
        AssertionError
            The outpath extension is not '.msk'.

        '''
        with open (outpath, 'wb') as op:
            assert os.path.splitext(outpath)[1] == '.msk'
            np.savez(op, mask_array=self.mask)
        self.filepath = outpath


class InputMapStack():
    '''
    Base class for layering 2d Input Maps of same shape together into a single
    3D stack. A mask, if provided, is applied to the entire stack. This class
    also includes several convenient methods for the stack classification and 
    management.
    '''

    def __init__(self, input_maps: list[InputMap], mask: Mask|None=None):
        '''
        InputMapStack class constructor.

        Parameters
        ----------
        input_maps : list of InputMap
            List of Input Maps. They must be stackable (same shape).
        mask : Mask or None, optional
            Mask to apply to the entire stack. The default is None.

        Raises
        ------
        ValueError
            Input maps list must contain only InputMap objects. 

        '''
    # Make sure that input maps are in a list (no other iterable is allowed)
        if not isinstance(input_maps, list):
            input_maps = list(input_maps)

    # Make sure that input maps list contains ONLY InputMap objects
        if not all(isinstance(i, InputMap) for i in input_maps):
            raise ValueError('"input_maps" must contain only InputMaps.')

    # Set main attributes
        self._stack = None
        self.input_maps = input_maps
        self.arrays = [m.map for m in input_maps]
        self.mask = mask
        if self.mask is not None:
            self._set_mask()


    def __len__(self):
        '''
        Return the length of the stack.

        Returns
        -------
        Stack length
            Number of maps included in the stack.

        '''
        return len(self.arrays)
    

    def _set_mask(self):
        '''
        Internal function that applies a mask to the entire stack.

        '''
        if self.maps_fit() and self.mask_fit():
            msk = self.mask.mask
            self.arrays = [np.ma.masked_where(msk, a) for a in self.arrays]
    

    @property
    def maps_shape(self):
        '''
        Return the shape of one slice of the stack (2D map shape).

        Returns
        -------
        tuple
            Maps 2D shape.

        '''
        if self.maps_fit():
            return self.arrays[0].shape
        else:
            return None
        

    @property
    def stack_shape(self):
        '''
        Return the shape of the entire stack (3D shape).

        Returns
        -------
        tuple
            Stack 3D shape.

        '''
        if self.maps_fit() and self.mask_fit():
            return self.stack.shape
        else:
            return None

        
    @property
    def stack(self):
        '''
        Return the stack.

        Returns
        -------
        ndarray
            The 3D maps stack.

        Raises
        ------
        ValueError
            Input maps and/or mask have wrong shapes.

        '''
        if self._stack is None:
            if not self.maps_fit() or not self.mask_fit():
                raise ValueError('Inputs do not share the same size.')
            else:
                self._stack = np.stack(self.arrays)
        
        return self._stack


    def add_maps(self, maps:list[InputMap]):
        '''
        Add new maps to the stack.

        Parameters
        ----------
        maps : list
            List of InputMaps

        Raises
        ------
        ValueError
            Input maps list must contain only InputMap objects. 

        '''
    # Make sure that the list contains only InputMaps
        if all(isinstance(m, InputMap) for m in maps):
            self.input_maps.extend(maps)
            self.__init__(self.input_maps, self.mask)
        else:
            raise ValueError('"maps" must contain only InputMaps.')


    def maps_fit(self):
        '''
        Check if all maps have same shape.

        Returns
        -------
        fit : bool
            Whether the arrays share the same shape.

        '''
        fit = all([a.shape == self.arrays[0].shape for a in self.arrays])
        return fit
    
    
    def mask_fit(self):
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
    

    def get_feature_array(self):
        '''
        Return a flatten/compressed and transposed version of the stack, which
        is required by classification algorithms.

        Returns
        -------
        feat_array : ndarray
            The feature array.

        Raises
        ------
        ValueError
            Input maps and/or mask have differet shapes.

        '''
    # Raise error if maps do not share the same shape
        if not self.maps_fit() or not self.mask_fit():
            raise ValueError('Inputs do not share the same size.')
    # If maps are masked, make them 1D with compressed() otherwise use flatten()
        if self.mask is None:
            feat_array = np.vstack([a.flatten() for a in self.arrays]).T
        else:
            feat_array = np.vstack([a.compressed() for a in self.arrays]).T

        return feat_array
    

    def reorder(self, index_list:list[int]):
        '''
        Sort the stack using a list of indices.

        Parameters
        ----------
        index_list : list
            List of indices.

        '''
        self.input_maps[:] = [self.input_maps[i] for i in index_list]
        self.arrays[:] = [self.arrays[i] for i in index_list]
        self._stack = np.stack(self.arrays)