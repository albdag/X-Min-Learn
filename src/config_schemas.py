"""
Defines schemas for saving tools and panes state and configs to project files.

Created on Fri Oct 02 17:26:30 2026
@author: albdag
"""

from typing import Literal, TypedDict

# DATA MANAGER CONFIG SCHEMA --------------------------------------------------
type MapObjectSchema = tuple[str, str] # name, filepath
type MaskObjectSchema = tuple[str, str, bool] # name, filepath, checkstate
type DataObjectSchema = MapObjectSchema | MaskObjectSchema
SubGroupSchema = TypedDict('SubGroupSchema', {
    'Input Maps': list[MapObjectSchema],
    'Mineral Maps': list[MapObjectSchema],
    'Masks': list[MaskObjectSchema]
})
type DataManagerConfig = dict[str, SubGroupSchema]

DATAMANAGER_DEFAULT_CONFIG: DataManagerConfig = {}

# HISTOGRAM VIEWER CONFIG SCHEMA ----------------------------------------------
class HistogramCanvasSchema(TypedDict):
    Logscale: bool
    Bins: int
class HistogramNavigationSchema(TypedDict):
    Visible: bool
    ActiveMode: Literal['', 'pan/zoom', 'zoom rect']
class HistogramScalerSchema(TypedDict):
    Enabled: bool
    Extents: tuple[int, int] # min value, max value
class HistogramViewerConfig(TypedDict):
    Canvas: HistogramCanvasSchema
    NavTbar: HistogramNavigationSchema
    Scaler: HistogramScalerSchema

HISTOGRAM_DEFAULT_CONFIG: HistogramViewerConfig = {
    'Canvas': {
        'Logscale': True,
        'Bins': 50
    },
    'NavTbar': {
        'Visible': True,
        'ActiveMode': ''
    },
    'Scaler': {
        'Enabled': False,
        'Extents': (0, 0)
    }
}

# MODE VIEWER CONFIG SCHEMA ---------------------------------------------------
class ModeNavigationSchema(TypedDict):
    Visible: bool
    Labelize: bool
class ModeViewerConfig(TypedDict):
    NavTbar: ModeNavigationSchema

MODE_DEFAULT_CONFIG: ModeViewerConfig = {
    'NavTbar': {
        'Visible': True,
        'Labelize': False
    }
}

# ROI EDITOR CONFIG SCHEMA ----------------------------------------------------
class RoiSelectorSchema(TypedDict):
    Extents: tuple[float, float, float, float] # xmin, xmax, ymin, ymax
    FixedExtents: tuple[int, int, int, int] | None # rmin, rmax, cmin, cmax
    Active: bool
class RoiMapSchema(TypedDict):
    Path: str | None
    Visible: bool
class RoiNavigationSchema(TypedDict):
    Visible: bool
    Labelize: bool
class RoiEditorConfig(TypedDict):
    Selector: RoiSelectorSchema
    RoiMap: RoiMapSchema
    NavTbar: RoiNavigationSchema

ROIEDITOR_DEFAULT_CONFIG: RoiEditorConfig = {
    'Selector': {
        'Extents': (0., 0., 0., 0.),
        'FixedExtents': None,
        'Active': False
    },
    'RoiMap': {
        'Path': None,
        'Visible': True
    },
    'NavTbar': {
        'Visible': True,
        'Labelize': False
    }
}

# PROBABILITY MAP VIEWER CONFIG SCHEMA ----------------------------------------
class ProbabilityNavigationSchema(TypedDict):
    Visible: bool
    ActiveMode: Literal['', 'pan/zoom', 'zoom rect']
class ProbabilityViewRangeSchema(TypedDict):
    Enabled: bool
    Range: tuple[float, float]
class ProbabilityViewerConfig(TypedDict):
    NavTbar: ProbabilityNavigationSchema
    ViewRange: ProbabilityViewRangeSchema

PROBABILITY_DEFAULT_CONFIG: ProbabilityViewerConfig = {
    'NavTbar': {
        'Visible': True,
        'ActiveMode': ''
    },
    'ViewRange': {
        'Enabled': False,
        'Range': (0.0, 1.0)
    }
}

# RGBA COMPOSITE MAP VIEWER CONFIG SCHEMA -------------------------------------
class RgbaNavigatorSchema(TypedDict):
    Visible: bool
    ActiveMode: Literal['', 'pan/zoom', 'zoom rect']
class RgbaChannelsSchema(TypedDict):
    R: str
    G: str
    B: str
    A: str
class RgbaViewerConfig(TypedDict):
    NavTbar: RgbaNavigatorSchema
    Channels: RgbaChannelsSchema

RGBA_VIEWER_CONFIG: RgbaViewerConfig = {
    'NavTbar': {
        'Visible': True,
        'ActiveMode': ''
    },
    'Channels': {
        'R': '',
        'G': '',
        'B': '',
        'A': ''
    }
}