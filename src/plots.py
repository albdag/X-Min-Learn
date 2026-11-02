# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 09:14:45 2023

@author: albdag
"""

from collections.abc import Callable, Collection, Iterable, Sequence
from functools import partial
from typing import Any, Literal, overload

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor, QResizeEvent
from PyQt5.QtWidgets import QAction, QMenu, QSizePolicy, QWidget, QWidgetAction

import numpy as np
import numpy.typing as npt

from matplotlib.axes import Axes
from matplotlib.backends import backend_qt, backend_qtagg
from matplotlib.collections import PolyCollection
from matplotlib.colorbar import Colorbar
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation, Text
from matplotlib import backend_bases as mpl_backend_bases
from matplotlib import colormaps as mpl_colormaps
from matplotlib import colors as mpl_colors
from matplotlib import patheffects as mpl_patheffects
from matplotlib import style as mpl_style
from matplotlib import widgets as mpl_widgets
from mpl_interactions import panhandler
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import settings
import style

# mpl.use('Qt5Agg') 'pdf', 'pgf', 'ps', 'svg' add to pyinstaller hooks (https://pyinstaller.org/en/v5.0/hooks-config.html#matplotlib-hook-options)
type MplColor = tuple[float, float, float] | float | str

class PanHandler(panhandler):

    _pressCursor = QCursor(Qt.CursorShape.SizeAllCursor)
    _releaseCursor = QCursor(Qt.CursorShape.ArrowCursor)

    def __init__(self, fig: Figure, button: int = 2) -> None:
        '''
        Reimplemantation of the 'panhandler' class from the 'mpl_interactions'
        library. It simply adds methods to control the type of cursor.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure.
        button : int, optional
            Mouse button that triggers the pan. 1: left-click, 2: middle-click,
            3: right-click. The default is 2.

        '''
        super().__init__(fig, button)
        self.wheelButton = button


    def _press(self, event: mpl_backend_bases.MouseEvent) -> None:
        '''
        Reimplementation of the '_press' method. It changes the cursor before
        calling parent method. Moreover, the current cursor is stored as the 
        new '_releaseCursor'.

        Parameters
        ----------
        event : Matplotlib MouseEvent
            The mouse press event.

        '''
        if event.button == self.wheelButton:
            self._releaseCursor = self.fig.canvas.cursor()
            self.fig.canvas.setCursor(self._pressCursor)
        super()._press(event)


    def _mouse_move(self, event: mpl_backend_bases.MouseEvent) -> None:
        '''
        Reimplementation of the '_mouse_move' method. It forces the cursor to
        be the '_pressCursor' during mouse wheel dragging.

        Parameters
        ----------
        event : mpl_backend_bases.MouseEvent
            The mouse move event.

        '''
        if self._id_drag is not None:
            self.fig.canvas.setCursor(self._pressCursor)
        super()._mouse_move(event)


    def _release(self, event: mpl_backend_bases.MouseEvent) -> None:
        '''
        Reimplementation of the '_release' method. It changes the cursor back 
        after calling the parent method.

        Parameters
        ----------
        event : Matplotlib MouseEvent
            The mouse release event.

        '''
        super()._release(event)
        if event.button == self.wheelButton:
            self.fig.canvas.setCursor(self._releaseCursor)



class _CanvasBase(backend_qtagg.FigureCanvasQTAgg):

    _figure_kwargs = {
        'facecolor': style.IVORY,
        'edgecolor': style.BLACK_PEARL,
        'linewidth': 1
    }
    _ax_facecolor = style.BLACK_PEARL
    _title_pad = 5

    def __init__(
        self,
        size: tuple[float, float] = (6.4, 4.8),
        layout: Literal['constrained', 'compressed', 'tight'] | None = None,
        wheel_zoom: bool = True,
        wheel_pan: bool = True,
        init_blank: bool = True
    ) -> None:
        '''
        A base canvas for any type of plot.

        Parameters
        ----------
        size : tuple[float, float], optional
            Width and height (in inches) of canvas. The default is (6.4, 4.8).
        layout : Literal['constrained', 'compressed', 'tight'] | None, optional
            Layout engine of the figure. The default is 'none'.
        wheel_zoom : bool, optional
            Whether or not allow zooming with mouse wheel. The default is True.
        wheel_pan : bool, optional
            Whether or not allow drag-pan with pressed mouse wheel. The default
            is True.
        init_blank : bool, optional
            Whether ax should be initialized as blank. The default is True.

        '''
    # Define the figure and the ax of the matplotlib canvas
        self.fig = Figure(size, layout=layout, **self._figure_kwargs)
        self.ax = self.fig.add_subplot(111, facecolor=self._ax_facecolor)
        
    # Initialize blank ax if required
        if init_blank:
            self.ax.axis('off')

    # Call the constructor of the parent class
        super().__init__(self.fig)

    # Set the default style
    # Should this change based on current app style (default or darkmode)?
        mpl_style.use('seaborn-v0_8-dark') 

    # Set events connections for mouse-wheel zoom and mouse-wheel pan
        if wheel_zoom:
            self._id_zoom = self.wheelZoomHandler()
            self.setMouseTracking(True)
        if wheel_pan:
            self._id_pan = PanHandler(self.fig)

    # Enable custom context menu request (when right-clicking on canvas)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)


    def set_padded_title(self, title: str) -> None:
        '''
        Convenient function to set ax title with default title pad. Child
        classes can overwrite default title pad, by changing '_title_pad'
        class attribute.

        Parameters
        ----------
        title : str
            New ax title.

        '''
        self.ax.set_title(title, pad=self._title_pad)


    def update_toolbar(self) -> None:
        '''
        Update the navigation toolbar stack associated with the canvas, if
        existent.

        '''
        if hasattr(self, "toolbar") and self.toolbar is not None:
            self.toolbar.update()


    def _clear_canvas(self, deep_clear: bool = False) -> None:
        '''
        Generic canvas clearing actions.

        Parameters
        ----------
        deep_clear : bool, optional
            Fully reset the ax. The default is False.

        '''
    # Fully reset the ax if requested:
        if deep_clear: 
            self.ax.clear()

    # Hide the axis (borders and ticks)
        self.ax.axis('off')

    # Reset the navigation toolbar stack to fix unwanted glitches with clims 
        self.update_toolbar()


    def mouseMoveEvent(self, event: mpl_backend_bases.MouseEvent) -> None:
        '''
        Set focus on the canvas widget when mouse hovering. Necessary to detect
        the key attribute of the mouse wheel zoom event.

        Parameters
        ----------
        event : Matplotlib MouseEvent
            The mouse move event.

        '''
        super().mouseMoveEvent(event)
        if self.hasMouseTracking():
            self.setFocus()


    def get_navigation_context_menu(self, navtbar: 'NavTbar') -> QMenu:
        '''
        Return a default context menu with actions extracted from the provided
        navigation toolbar.

        Parameters
        ----------
        navtbar : NavTbar
            The navigation toolbar associated with this canvas.

        Returns
        -------
        menu : QMenu
            A menu populated with actions.

        '''
        menu = QMenu(self)
        menu.setStyleSheet(style.SS_MENU)

    # Get Navigation Toolbar true actions (exclude QWidgetActions, like coords)
        ntbar_actions = navtbar.getTrueActions()

    # Get the hidden 'Show toolbar' action
        hide_ntbar = navtbar.showToolbarAction()
        hide_ntbar.setText('Show navigation toolbar')

    # Add actions to menu
        menu.addAction(hide_ntbar)
        menu.addSeparator()
        menu.addActions(ntbar_actions)

        return menu
        

    def share_axis(self, ax: Axes, share: bool = True) -> None:
        '''
        Share an ax from a different canvas with the ax of this canvas, so that 
        zoom and pan operations on one ax are reflected on the other.

        Parameters
        ----------
        ax : Axes
            Sharing ax.
        share : bool, optional
            Toggle on/off sharing. The default is True.

        '''
        shared_x = self.ax._shared_axes['x'] # pyright: ignore[reportAttributeAccessIssue]
        shared_y = self.ax._shared_axes['y'] # pyright: ignore[reportAttributeAccessIssue]
    # For older version of Matplotlib
        # shared_x = self.ax._shared_x_axes
        # shared_y = self.ax._shared_y_axes

        if share:
            shared_x.join(self.ax, ax)
            shared_y.join(self.ax, ax)
        else:
            shared_x.remove(ax)
            shared_y.remove(ax)


    def _update_canvas(self) -> None:
        '''
        Generic canvas update actions.

        '''
    # Show the axis (borders and ticks)
        self.ax.axis('on')

    # Reset the navigation toolbar stack to fix unwanted glitches with clims 
        self.update_toolbar()


    # Inspired by https://gist.github.com/tacaswell/3144287
    # and https://github.com/mpl-extensions
    def wheelZoomHandler(self, base_scale: float = 1.5) -> Callable[[], None]:
        '''
        Wheel zoom factory.

        Parameters
        ----------
        base_scale : float, optional
            Base zoom scale. The default is 1.5.

        '''

        def zoom(event: mpl_backend_bases.MouseEvent) -> None:
            '''
            Apply zoom to canvas after mouse wheel scroll event.

            Parameters
            ----------
            event : Matplotlib Event
                Matplotlib's mouse scroll event.

            '''

        # Get the current x and y limits
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()

        # Get cursor location
            xdata = event.xdata
            ydata = event.ydata

        # Exit the function if the cursor is outside the figure
            if xdata is None or ydata is None: return

        # Reduce the scale when pressing Ctrl key (accurate zoom)
            scale = 1.1 if event.key and 'control' in event.key else base_scale

        # Set zoom in or out
            if event.button == 'down':
                scale_factor = 1/scale
            elif event.button == 'up':
                scale_factor = scale
            else:
                scale_factor = 1

        # Set new limits
            self.ax.set_xlim((
                xdata - (xdata - cur_xlim[0]) / scale_factor,
                xdata + (cur_xlim[1] - xdata) / scale_factor
            ))
            self.ax.set_ylim((
                ydata - (ydata - cur_ylim[0]) / scale_factor,
                ydata + (cur_ylim[1] - ydata) / scale_factor
            ))

        # Redraw the canvas
            self.draw_idle()

    # Attach the callback
        cid = self.mpl_connect("scroll_event", zoom) # pyright: ignore[reportArgumentType]

    # Return the disconnect callback function
        return lambda id_=cid: self.mpl_disconnect(id_)



class ImageCanvas(_CanvasBase):

    _title_pad = 15 # overrides parent class attribute

    def __init__(
        self,
        binary: bool = False,
        cbar: bool = True,
        size: tuple[float, float] = (10.0, 7.5),
        **kwargs
    ) -> None:
        '''
        A base class for any type of canvas displaying images and maps.

        Parameters
        ----------
        binary : bool, optional
            Adapt the canvas to plot binary data. The default is False.
        cbar : bool, optional
            Include a colorbar in the canvas. The default is True.
        size : tuple[float, float], optional
            Width and height (in inches) of the canvas. The default is
            (10.0, 7.5).
        **kwargs
            Parent class arguments (see _CanvasBase class).

        '''
        super().__init__(size=size, **kwargs)

    # Set the pixel coordinate format
        self.ax.format_coord = lambda x, y: f'X : {round(x)}, Y : {round(y)}'

    # Define the main attributes of the canvas
        self._is_binary = binary
        self._image: AxesImage | None = None
        self.scaled_clim = None # scaled clims are currently unused

    # Initialize the colormap
        self._cmap: mpl_colors.Colormap | None = None

    # Initialize the colorbar
        self._has_colorbar = cbar
        self._cbar: Colorbar | None = None

    # Define the colors of out-of-range and masked values for colormaps
        self._bad = {
            'Linear': mpl_colors.to_rgb(style.IVORY),
            'Discrete': mpl_colors.to_rgb(style.IVORY)
        }
        self._over = {
            'Linear': mpl_colors.to_rgb('0.2'),
            'Discrete': mpl_colors.to_rgb(style.IVORY)
        }
        self._under = {
            'Linear': mpl_colors.to_rgb('0.5'),
            'Discrete': mpl_colors.to_rgb(style.IVORY)
        }


    def is_empty(self) -> bool:
        '''
        Check if the canvas is empty.

        Returns
        -------
        bool
            Whether or not the canvas is empty.

        '''
        return self._image is None


    def _set_linear_colormap(
        self, name: str = 'Spectral_r') -> mpl_colors.LinearSegmentedColormap:
        '''
        Set and return a linear segmented colormap suited to display a heatmap.

        Parameters
        ----------
        name : str, optional
            Matplotlib linear colormap name. The default is 'Spectral_r'.

        Returns
        -------
        Colormap
            New linear colormap. 

        '''
    # Ensure requested colormap is a LinearSegmentedColormap
        self._cmap = mpl_colormaps['binary_r' if self._is_binary else name]
        if not isinstance(self._cmap, mpl_colors.LinearSegmentedColormap):
            self._cmap = None
            raise ValueError(f'{name} is not a valid heatmap colormap name.')

    # Set colors for outliers and masked data
        self._cmap.set_over(self._over['Linear'])
        self._cmap.set_under(self._under['Linear'])
        self._cmap.set_bad(self._bad['Linear'])
        
        return self._cmap


    def _set_discrete_colormap(
        self,
        colors: Iterable[tuple[int, int, int]],
        data: np.ndarray,
        name: str = 'CustomCmap'
    ) -> mpl_colors.ListedColormap:
        '''
        Set and return a discrete colormap suited to display a discrete map.

        Parameters
        ----------
        colors : Iterable[tuple[int, int, int]]
            Iterable of RGB triplets.
        data : np.ndarray | None
            Image map data as numerical class IDs.
        name : str, optional
            Colormap name. The default is 'CustomCmap'.

        Returns
        -------
        ListedColormap
            New descrete colormap.

        '''
    # Convert colorlist RGB triplets to float RGB values
        float_rgb = rgb_to_float(colors)

    # Set colors of masked classes as "bad" (required for colorbar rendering)
        bad = self._bad['Discrete']
        if np.ma.is_masked(data) and self._has_colorbar:
            unq = np.unique(np.ma.compressed(data)) # unmasked class IDs
            float_rgb = [x if i in unq else bad for i, x in enumerate(float_rgb)]

    # Set colormap
        self._cmap = mpl_colors.ListedColormap(float_rgb, name=name)

    # Set colors for outliers and masked data
        self._cmap.set_over(self._over['Discrete'])
        self._cmap.set_under(self._under['Discrete'])
        self._cmap.set_bad(bad)

        return self._cmap


    def alter_colormap(
            self, color_arg: str | Iterable[tuple[int, int, int]]) -> None:
        '''
        Change the current colormap.

        Parameters
        ----------
        color_arg : str or Iterable[tuple[int, int, int]]
            If string, it must be a valid matplotlib colormap name, and the
            resulting colormap will be suited to display a heatmap. If an
            Iterable of RGB triplets, the colormap will be suited to display a
            discrete map.

        Raises
        ------
        ValueError
            Raised if 'color_arg' is not a string nor an iterable.

        '''
        # Exit function if the canvas is empty
        if self._image is None: 
            return

        # Set a linear colormap assuming a valid colormap name
        if isinstance(color_arg, str):
            cmap = self._set_linear_colormap(color_arg)

        # Set a discrete colormap assuming iterable of RGB triplets
        elif isinstance(color_arg, Iterable):
            if (data := self._image.get_array()) is None:
                return
            cmap = self._set_discrete_colormap(color_arg, data)

        # Safety fallback
        else:
            raise ValueError(f'{color_arg} is not a valid argument.')

        # Set the new colormap and render it
        self._image.set_cmap(cmap)
        self.draw_idle()


    def has_discretemap(self) -> bool:
        '''
        Check if the canvas is currently displaying a discrete map.

        Returns
        -------
        bool
            Whether the currently displayed data is a discrete map.

        '''
        return isinstance(self._cmap, mpl_colors.ListedColormap)


    def has_heatmap(self) -> bool:
        '''
        Check if the canvas is currently displaying a heatmap.

        Returns
        -------
        bool
            Whether the currently displayed data is a heatmap.

        '''
        return isinstance(self._cmap, mpl_colors.LinearSegmentedColormap)


    def get_boundary_norm(self, n_colors: int) -> mpl_colors.BoundaryNorm:
        '''
        Return boundary norms based on the given amount of colors. Useful to 
        properly set a colormap for a discrete map.

        Parameters
        ----------
        n_colors : int
            Number of colors.

        Returns
        -------
        norm : BoundaryNorm
            The boundary norm.

        '''
        boundaries = [n - 0.5 for n in range(n_colors + 1)]
        norm = mpl_colors.BoundaryNorm(boundaries, n_colors)
        return norm


    def _clear_colorbar(self) -> None:
        '''
        Cleanly remove any trace of previously drawn colorbars.

        '''
        if self._cbar is not None:
            cax = self._cbar.ax
            cax.clear()
            self.fig.delaxes(cax)
            self._cbar = None


    def _set_colorbar(self) -> Colorbar | None:
        '''
        Set and return a colorbar based on current map image.

        Returns
        -------
        Colorbar | None
            The new colorbar or None, if the current image is empty. 
        
        '''
        # Always clear previously drawn colorbars to avoid memory leaks 
        if self._cbar is not None:
            self._clear_colorbar()

        if self._image is None:
            return None

        # Re-create the colorbar and its ax from scratch (best to avoid mpl 
        # glitches with qt backends)
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size=0.4, pad=0.15)
        # consider using mappable as matplotlib.cm.ScalarMappable instead
        # of self._image. From Matplotlib docs: "Note that one can create a
        # ScalarMappable 'on-the-fly' to generate colorbars not attached to
        # a previously drawn artist, e.g.:
        # fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)"
        self._cbar = self.fig.colorbar(self._image, cax=cax, extend='both')
        return self._cbar


    def _toggle_colorbar(self, toggled: bool) -> None:
        '''
        Enable or disable the colorbar. Requires canvas redraw to view changes.

        Parameters
        ----------
        toggled : bool
            Whether the colorbar should be enabled or not.

        '''
        if toggled:
            cbar = self._set_colorbar()
            if cbar is not None:
                cbar.ax.set_visible(True)
        else:
            self._clear_colorbar()


    def scale_clim(self, toggled: bool, arrays: list[np.ndarray] | None = None) -> None:
        '''
        Toggle scaled norm limits.

        Parameters
        ----------
        toggled : bool
            Enable/disable scaled norm limits.
        arrays : list[numpy ndarray] or None, optional
            List of arrays from which to extract the scaled norm limits. It
            must be a non-empty list if 'toggled' is True. The default is None.

        Raises
        ------
        ValueError
            The array list must be non-empty when toggled is True.

        '''
    # Enable scaled norm limits
        if toggled:
            if not arrays:
                err = '"arrays" must be a non-empty list if "toggled" is True.'
                raise ValueError(err)
            glb_min = min([arr.min() for arr in arrays])
            glb_max = max([arr.max() for arr in arrays])
            self.scaled_clim = (glb_min, glb_max)

    # Disable scaled norm limits
        else: 
            self.scaled_clim = None

    # Update the norm limits in any case
        self.update_clim()


    def update_clim(
        self,
        vmin: int | float | None = None,
        vmax: int | float | None = None
    ) -> None:
        '''
        Update the image norm limits.

        Parameters
        ----------
        vmin : int or float or None, optional
            Lower limit. The default is None.
        vmax : int or float or None, optional
            Upper limit. The default is None.

        '''
        # Exit function if canvas is empty
        if self._image is None:
            return
        
        # Replace invalid clims with unmasked min/max values from image array
        if vmin is None or vmax is None:
            if (array := self._image.get_array()) is None:
                return
            vmin, vmax = float(np.nanmin(array)), float(np.nanmax(array))

        # Set clims and norm of discrete maps to avoid glitches with boundaries
        # (fixes visual bugs with class highlighting in legend)
        if self.has_discretemap():
            # Always refresh norm using full cmap palette before setting clims 
            norm = self.get_boundary_norm(self._image.get_cmap().N)
            self._image.set_norm(norm)
            self._image.set_clim(vmin - 0.5, vmax + 0.5)

        # Clims and norm of heatmaps are simpler and will sync automatically
        elif self.has_heatmap():
            self._image.set_clim(vmin, vmax)

        # Safety fallback
        else:
            return

        # Reset the navigation toolbar stack to fix glitches with clims 
        self.update_toolbar()
        self.draw_idle()


    def clear_canvas(self) -> None:
        '''
        Clear out and reset the properties of the map-related artists in the 
        canvas.

        '''
        if self._image is not None:

        # Run generic shallow cleaning actions
            self._clear_canvas()

        # Remove the image and its colormap
            self._image.remove()
            self._image = None
            self._cmap = None

        # Remove the colorbar
            self._clear_colorbar()

        # Remove the title
            self.set_padded_title('')

        # Render the canvas
            self.draw_idle()


    def enable_picking(self, enabled: bool) -> None:
        '''
        Enable data picking from this canvas.

        Parameters
        ----------
        enabled : bool
            Enable/disable data picking.

        '''
        if self._image is not None:
            self._image.set_picker(True if enabled else None)
            # self.draw_idle()


    def reset_view(self) -> None:
        '''
        Show the original map view. Fixes issues when clicking home button in
        the Navigation Toolbar.

        '''
    # Exit function if canvas is empty
        if self._image is None:
            return

    # Exit function if image array data is invalid
        if (data := self._image.get_array()) is None:
            return

    # Fix axes extents
        extents = (-0.5, data.shape[1] - 0.5, data.shape[0] - 0.5, -0.5)
        self.ax.set_xlim(extents[:2]) 
        self.ax.set_ylim(extents[2:])
    
    # Fix image extents (correct aspect ratio)
        self._image.set_extent(extents)

    # Fix image zoom issues when pressing home button multiple times
        if self.fig.get_tight_layout():
            self.fig.tight_layout()

    # Render the canvas
        self.draw_idle()


    def zoom_to(self, x: int, y: int) -> None:
        '''
        Zoom to a specific pixel in the map.

        Parameters
        ----------
        x : int
            The x coordinate of the pixel.
        y : int
            The y coordinate of the pixel.

        '''
        if (
            not self._image is None
            and (data := self._image.get_array()) is not None
            and x <= (data.shape[1] - 1)
            and y <= (data.shape[0] - 1)
        ):
            self.ax.set_xlim((x - 1, x + 1))
            self.ax.set_ylim((y + 1, y - 1))
            self.draw_idle()


    def _format_cursor(self, data: Any, encoder: dict[int, str] | None = None) -> str:
        '''
        Format cursor data when mouse hovering on map image.

        Parameters
        ----------
        data : Any
            Image data of hovered pixel.
        encoder : dict[int, str] | None, optional
            Class encoder for discrete map data. Use None for heatmap data. The
            default is None.

        Returns
        -------
        str
            String representation of 'data'.

        '''
        if encoder is None:
            return f'Pixel Value = {data}'
        name = '--' if np.ma.is_masked(data) else encoder.get(int(data), '?')
        return f'Pixel Class = {name}'


    def _set_image(
        self,
        data: npt.NDArray[Any],
        cmap: mpl_colors.Colormap,
        norm: mpl_colors.Normalize
    ) -> AxesImage:
        '''
        Set and return a new map image. Canvas requires a redraw to show the
        changes.

        Parameters
        ----------
        data : NDArray[Any]
            Map data array.
        cmap : mpl_colors.Colormap
            Heatmap or discrete map colormap.
        norm : mpl_colors.Normalize
            Color mapping normalization.

        Returns
        -------
        AxesImage
            The new map image.

        '''
        # Set image
        if self._image is None:
            kwargs = {'cmap': cmap, 'norm': norm, 'interpolation': 'none'}
            self._image = self.ax.imshow(data, **kwargs)
        else:
            self._image.set(data=data, cmap=cmap, norm=norm)

        # Adjust image's aspect ratio
        extents = (-0.5, data.shape[1] - 0.5, data.shape[0] - 0.5, -0.5)
        self._image.set_extent(extents)

        return self._image


    def draw_discretemap(
        self,
        data: np.ndarray,
        encoder: dict[int, str], 
        colors: Collection[tuple[int, int, int]],
        title: str | None = None
    ) -> None:
        '''
        Update the canvas with a new discrete map.

        Parameters
        ----------
        data : np.ndarray
            A 2D array storing the image pixels as numerical class IDs.
        encoder : dict[int, str]
            Dictionary that links the class IDs (key) with the corresponding
            class names (values).
        colors : Collection[tuple[int, int, int]]
            RGB color triplets. The number of colors must match the length of
            'encoder'.
        title : str or None, optional
            The image title. If None, the previous image title will be used.
            The default is None.

        '''
    # Run generic update actions
        self._update_canvas()

    # Set image title
        if title is not None: 
            self.set_padded_title(title)
            
    # Draw new map with a discrete colormap and boundary normalization
        cmap = self._set_discrete_colormap(colors, data)
        norm = self.get_boundary_norm(len(colors))
        image = self._set_image(data, cmap, norm)

    # Set cursor data format (i.e., when hovering with mouse over pixels)
        image.format_cursor_data = partial(self._format_cursor, encoder=encoder)

    # Disable the colorbar (discrete maps are not expected to have one)
        self._toggle_colorbar(False)
        # self._toggle_colorbar(self._has_colorbar) # use this to show cbar

    # Redraw canvas
        self.draw_idle()


    def draw_heatmap(
        self,
        data: np.ndarray,
        title: str | None = None,
        cmap_name: str = 'Spectral_r'
    ) -> None:
        '''
        Update the canvas with a new heatmap.

        Parameters
        ----------
        data : numpy ndarray
            A 2D array storing the image pixel values.
        title : str or None, optional
            The image title. If None, the previous image title will be used. 
            The default is None.
        cmap_name : str
            A Matplotlib default colormap name. The default is "Spectral_r".

        '''
    # Run generic update actions
        self._update_canvas()

    # Set image title
        if title is not None: 
            self.set_padded_title(title)

    # Draw new map with a linear colormap and linear normalization
        cmap = self._set_linear_colormap(cmap_name)
        norm = mpl_colors.Normalize()
        image = self._set_image(data, cmap, norm)

    # Set cursor data format (i.e., when hovering with mouse over pixels)
        image.format_cursor_data = partial(self._format_cursor)

    # Enable the colorbar if required
        self._toggle_colorbar(self._has_colorbar)

    # Redraw canvas
        self.draw_idle()


    def has_masked_array(self) -> bool:
        '''
        Check if the canvas is displaying masked data.

        Returns
        -------
        bool
            Whether the displayed data is masked.

        '''
        if self._image is None:
            return False
        return np.ma.is_masked(self._image.get_array())


    def get_array(self) -> np.ndarray | np.ma.MaskedArray | None:
        '''
        Get the underlying array in the displayed image.

        Returns
        -------
        np.ndarray or None
            The image array or None if image is empty.

        '''
        if self._image is None:
            return None
        return self._image.get_array()


    @overload # when 'return_mask' is explicitly False (default)
    def get_unmasked_array(
        self,
        return_mask: Literal[False] = False
    ) -> np.ndarray | None: ...

    @overload # when 'return_mask' is explicitly True
    def get_unmasked_array(
        self,
        return_mask: Literal[True]
    ) -> tuple[np.ndarray | None, np.ndarray | None]: ...

    @overload # when 'return_mask' is a boolean variable
    def get_unmasked_array(
        self,
        return_mask: bool
    ) -> np.ndarray | None | tuple[np.ndarray | None, np.ndarray | None]: ...

    def get_unmasked_array(
        self,
        return_mask: bool = False
    ) -> np.ndarray | None | tuple[np.ndarray | None, np.ndarray | None]:
        '''
        Get an unmasked version of the underlying array in the displayed image
        and, optionally, its mask. To get the array "as is" use "get_array()".

        Parameters
        ----------
        return_mask : bool, optional
            Also return the detached mask, if existent. The default is False.

        Returns
        -------
        np.ndarray or None or tuple[np.ndarray or None, np.ndarray or None]
            The unmasked array (or None, if empty) and, optionally, its mask
            (or None, if empty).

        '''    
    # Return None array (and None mask, if required) when array is invalid
        if (array := self.get_array()) is None:
            return None if not return_mask else (None, None)

        unmasked_array = np.ma.getdata(array)
        mask = np.ma.getmaskarray(array) if np.ma.is_masked(array) else None
        return unmasked_array if not return_mask else (unmasked_array, mask)



class BarCanvas(_CanvasBase):

    _bar_properties = {'edgecolor': style.IVORY, 'linewidth': 0.5}
    _grid_properties = {'linestyle': ':', 'linewidth': 0.5, 'zorder': 0}
    _legend_properties = {'labelcolor': style.IVORY}

    def __init__(
        self,
        orientation: Literal['h', 'v'] = 'v',
        size: tuple[float, float] = (6.4, 3.6),
        layout: Literal['constrained', 'compressed', 'tight'] | None = 'tight',
        **kwargs
    ) -> None:
        '''
        A canvas class for plotting bar charts, including grouped bar charts
        with labels.

        Parameters
        ----------
        orientation : Literal['h', 'v'], optional
            Orientation of the bars. Can be horizontal ('h') or vertical ('v').
            The default is 'v'.
        size : tuple[float, float], optional
            Size of the canvas. The default is (6.4, 3.6)
        layout : Literal['constrained', 'compressed', 'tight'] | None, optional
            Layout engine of the figure. The default is 'tight'.
        **kwargs
            Parent class arguments (see _CanvasBase class).

        '''
        super().__init__(size=size, layout=layout, **kwargs)

    # Set main attributes
        self._orient = orientation
        self._plot: list[BarContainer] = []
        self._bar_thickness = 0.5

    # Set label amounts attributes
        self._decimal_precision = settings.manager.get('data/decimal_precision')
        self._visible_amounts = False
        self._label_amounts: list[list[Annotation]] = []
        self._label_amounts_properties: dict[str, Any] = {
            'fmt': f'%.{self._decimal_precision}f',
            'padding': 16,
            'label_type': 'center',
            'color': style.IVORY,
            'path_effects': [
                mpl_patheffects.withStroke(linewidth=2, foreground='k')
            ]
        }


    def is_empty(self) -> bool:
        '''
        Check if the canvas is empty.

        Returns
        -------
        bool
            Whether the canvas is empty.

        '''
        return not len(self._plot)


    def clear_canvas(self) -> None:
        '''
        Clear out the canvas and reset some properties.

        '''
    # Run generic deep cleaning actions
        self._clear_canvas(deep_clear=True)

    # Reset the references to the plotted data
        self._plot.clear()
        self._label_amounts.clear()
        self.draw_idle()


    def _show_label_amounts(self, visible: bool) -> None:
        '''
        Show/hide existing bar label amounts. Canvas requires a redraw to show
        the changes.

        Parameters
        ----------
        visible : bool
            Whether labels should be visible or not.

        '''
        for series in self._label_amounts:
            for lbl in series:
                lbl.set_visible(visible)


    def _set_label_amounts(self) -> None:
        '''
        Draw the bar labels showing amounts. Canvas requires a redraw to show
        the changes.

        '''
        kwargs = self._label_amounts_properties
        for series in self._plot:
            self._label_amounts.append(self.ax.bar_label(series, **kwargs))


    def toggle_amounts(self, toggled: bool) -> None:
        '''
        Show/hide label amounts displayed on top of the bars.

        Parameters
        ----------
        toggled : bool
            Whether or not to display the amounts.

        '''
        # Update the attribute in any case
        self._visible_amounts = toggled

        # Set visibility of label amounts, creating them if necessary
        if toggled and not len(self._label_amounts):
            self._set_label_amounts()
        else:
            self._show_label_amounts(toggled)

        # Redraw the canvas
        self.draw_idle()


    def set_bar_thickness(self, thickness: float) -> None:
        '''
        Set the thickness of plotted bars.

        Parameters
        ----------
        width : float
            The required thickness.

        '''
        self._bar_thickness = thickness


    def set_decimal_precision(self, precision: int) -> None:
        '''
        Set the number of decimal places displayed in labels.

        Parameters
        ----------
        precision : int
            Number of decimal places.
            
        '''
        self._decimal_precision = precision
        self._label_amounts_properties['fmt'] = f'%.{precision}f'


    def _get_ticklabels(self) -> list[Text]:
        '''
        Get current tick labels based on the orientation of the bar plot.

        Returns
        -------
        list[Text]
            X or Y tick labels.

        '''
        if self._orient == 'h':
            return self.ax.get_yticklabels()
        else:
            return self.ax.get_xticklabels()


    def change_bar_color(self, name: str, color: tuple[int, int, int]) -> None:
        '''
        Change the color of the bar named 'name'.

        Parameters
        ----------
        name : str
            Bar name, as displayed in the corresponding tick label.
        color : tuple[int, int, int]
            New color as RGB triplet.

        Raises
        ------
        ValueError
            Raised if using this method on a multi-bars plot.

        '''

        # Exit function if plot is empty
        if not (n_series := len(self._plot)):
            return

        # Raise error if this is a multi-bars plot
        if n_series > 1:
            raise ValueError(
                "Cannot use 'change_bar_color' in a multi-bars plot. " \
                "Use 'change_category_color' instead."
            )

        # Extract tick labels to identify the correct bar from given 'name'
        ticklabels = self._get_ticklabels()

        # Change the color of the corresponding bar
        for i, bar in enumerate(self._plot[0]):
            bar: Rectangle
            if name == ticklabels[i].get_text():
                (float_rgb,) = rgb_to_float((color,))
                bar.set_facecolor(float_rgb)
                break
        
        self.draw_idle()


    def change_category_color(self, name: str, color: tuple[int, int, int]) -> None:
        '''
        Change the color of all bars belonging to the category named 'name'.

        Parameters
        ----------
        name : str
            Category name, as displayed in the legend.
        color : tuple[int, int, int]
            New color as RGB triplet.

        Raises
        ------
        ValueError
            Raised if using this method on a single-bar plot.

        '''

        # Exit function if plot is empty
        if not (n_series := len(self._plot)):
            return

        # Raise error if this is a single-bar plot
        if n_series == 1:
            raise ValueError(
                "Cannot use 'change_category_color' in a single-bar plot. " \
                "Use 'change_bar_color' instead."
            )

        # Change the color of the bars that belong to the corresponding category
        for series in self._plot:
            if name == series.get_label():
                (float_rgb,) = rgb_to_float((color,))
                for bar in series:
                    bar: Rectangle
                    bar.set_facecolor(float_rgb)
                break

        # Redraw legend
        if (legend := self.ax.get_legend()) is not None:
            legend.remove()
        self.ax.legend(**self._legend_properties)
        self.draw_idle()


    def update_canvas(
        self,
        data: Sequence[Sequence[int | float]], 
        ticklabels: Collection[str],
        colors: Collection[tuple[int, int, int]] | None = None,
        categories: Sequence[str] | None = None,
        title: str | None = None
    ) -> None:
        '''
        Update the canvas with a new plot. The plot can either be a single-bar
        plot (one bar for each element) or a grouped-bars plot, where each 
        element displays multiple categories of values, using grouped bars.

        Parameters
        ----------
        data : Sequence[Sequence[int | float]]
            A sequence of sequences, where each inner sequence contains the 
            values of all elements for a single data category. Use multiple 
            inner sequences to include multiple categories for each element, 
            displayed as a grouped-bars plot. Each inner sequence must contain 
            the same number of elements.
        ticklabels : Collection[str]
            Tick labels for each element. The number of labels must match the 
            number of elements in the inner sequences. They must be unique.
        colors : Collection[tuple[int, int, int]] | None, optional
            A collection of RGB triplets. If one inner sequence of elements is
            passed to 'data' (single-bar plot), the number of colors must match
            the number of elements in that sequence. If multiple sequences are
            passed (grouped-bars plot), they are interpreted as category colors 
            and their amount must match the number of inner sequences. If None,
            the Matplotlib's default colors are used. The default is None.
        categories : Sequence[str] | None, optional
            Labels associated with each value category. This is mostly useful
            if multiple inner sequences are passed to 'data' (grouped-bars 
            plot). The number of category labels must match the number of inner
            sequences. If provided, category labels are displayed in a legend.
            If None, no legend will be displayed. The default is None.
        title : str | None, optional
            The plot title. If None, no title is shown. The default is None.

        Raises
        ------
        ValueError
            Raised if 'data' sequences have different lengths.
        ValueError
            Raised if an invalid number of 'ticklabels' is passed.
        ValueError
            Raised if an invalid number of 'colors' is passed.
        ValueError
            Raised if an invalid number of 'categories' is passed.

        '''
        # Run generic update actions
        self._update_canvas()

        # Clear the canvas and exit the function if data is empty
        if not (n_series := len(data)):
            return self.clear_canvas()
        
        # Check for inconsistencies in input parameters
        if len((series_length := set(len(d) for d in data))) > 1:
            raise ValueError('Data sequences must have the same length.')
        
        n_values = series_length.pop()
        if (ntl := len(set(ticklabels))) != n_values:
            raise ValueError(f'Got {ntl} unique tick labels, but {n_values} are required.')

        mpl_colors = None if colors is None else rgb_to_float(colors)
        if mpl_colors is not None:
            nc = len(mpl_colors)
            if n_series == 1 and nc != n_values:
                raise ValueError(f'Got {nc} colors, but {n_values} are required.')
            if n_series > 1 and nc != n_series:
                raise ValueError(f'Got {nc} colors, but {n_series} are required.')

        if categories is not None and (ncat := len(categories)) != n_series:
            raise ValueError(f'Got {ncat} categories, but {n_series} are required.')

        # Remove the previous plot, including label amounts and legend
        if not self.is_empty(): 
            self.ax.clear()
            self._plot.clear()
            self._label_amounts.clear()

        # Adjust the ax properties that are based on plot orientation
        ticks = np.arange(1, len(ticklabels) + 1)
        match self._orient:
            case 'v':
                plot_function = self.ax.bar
                self.ax.set_xticks(ticks, labels=ticklabels)
                self.ax.tick_params('x', which='both', length=0, rotation=-60)
                self.ax.grid(True, axis='y', **self._grid_properties)
            case 'h':
                plot_function = self.ax.barh
                self.ax.set_yticks(ticks, labels=ticklabels)
                self.ax.tick_params('y', which='both', length=0)
                self.ax.invert_yaxis()
                self.ax.grid(True, axis='x', **self._grid_properties)
            case _: # safety
                return

        # Set the title
        if title is not None: 
            self.set_padded_title(title)

        # Adjust the canvas for a multi-bars plot, if required
        shift = 0 if n_series == 1 else self._bar_thickness / n_series
        shift_step = np.linspace(-shift, shift, n_series)

        # Build the plot
        for i in range(n_series):
            cat = categories[i] if categories else None
            col = mpl_colors[i] if mpl_colors and n_series > 1 else mpl_colors
            series = plot_function(
                ticks + shift_step[i],
                data[i],
                self._bar_thickness,
                color=col,
                label=cat,
                **self._bar_properties
            )
            series.set_label(f'Series {i}' if cat is None else cat)
            self._plot.append(series)

        # Build the legend
        if categories is not None:
            self.ax.legend(**self._legend_properties)

        # Refresh label amounts if required
        self.toggle_amounts(self._visible_amounts)

        # Redraw the canvas
        self.draw_idle()



class HistogramCanvas(_CanvasBase):

    def __init__(
        self,
        density: bool = False,
        logscale: bool = False,
        size: tuple[float, float] = (3.0, 3.0),
        **kwargs
    ) -> None:
        '''
        A canvas class for plotting histograms.

        Parameters
        ----------
        density : bool, optional
            Whether to scale y axes to [0, 1]. The default is False.
        logscale : bool, optional
            Whether to use logarithmic scale. The default is False.
        size : tuple[float, float], optional
            Size of the canvas. The default is (3.0, 3.0)
        **kwargs
            Parent class arguments (see _CanvasBase class).

        '''
        super().__init__(size=size, **kwargs)

    # Set yaxis ticks properties
        self.ax.yaxis.set_ticks_position('both')

    # Set main attributes
        self._density = density
        self._log = logscale
        self._nbins = 50
        self._hist_data: np.ndarray | None = None
        self._roi_hist_data: np.ndarray | None = None


    def get_bins(self) -> int:
        '''
        Get the number of histogram bins.

        Returns
        -------
        int
            Number of bins.

        '''
        return self._nbins


    def set_bins(self, num: int) -> None:
        '''
        Set the number of histogram bins. The plot is refreshed automatically.

        Parameters
        ----------
        num : int
            Number of bins.

        '''
        self._nbins = num
        self.refresh_view()


    def is_empty(self) -> bool:
        '''
        Check if the canvas is empty.

        Returns
        -------
        bool
            Whether the canvas is empty.

        '''
        return self._hist_data is None


    def clear_canvas(self) -> None:
        '''
        Clear out the plot and reset some attributes.

        '''
    # Run generic deep cleaning actions
        self._clear_canvas(deep_clear=True)

        self._hist_data = None
        self._roi_hist_data = None
        self.draw_idle()


    def get_logscale(self) -> bool:
        '''
        Check if the scale is logarithmic or not.

        Returns
        -------
        bool
            Whether the scale is logarithmic or not.

        '''
        return self._log


    def set_logscale(self, toggled: bool) -> None:
        '''
        Toggle on/off logarithmic scale. The plot is refreshed automatically.

        Parameters
        ----------
        toggled : bool
            Toggle on/off.

        '''
        self._log = toggled
        self.refresh_view()


    def refresh_view(self) -> None:
        '''
        Refresh (re-render) the plot.

        '''
        if (data := self._hist_data) is not None:
            self.update_canvas(data, self._roi_hist_data, self.ax.get_title())


    def _to_hist_data(self, data: np.ndarray) -> npt.NDArray[Any]:
        '''
        Helper method to transform 2D array data to histogram-ready, flattened
        data. It accepts both masked and unmasked data arrays.

        Parameters
        ----------
        data : np.ndarray
            Masked or unmasked 2D data array.

        Returns
        -------
        NDArray[Any]
            1D flattened data array.

        '''
        return np.ma.compressed(data) if np.ma.is_masked(data) else data.flatten()


    def update_canvas(
        self,
        data: np.ndarray,
        roi_data: np.ndarray | None = None,
        title: str | None = None
    ) -> None:
        '''
        Update the canvas with a new plot.

        Parameters
        ----------
        data : numpy ndarray
            Data to plot in histogram.
        roi_data : numpy ndarray or None, optional
            The histogram data of a portion of a map enclosed by a ROI. Such
            data is plotted over the main plot. The default is None.
        title : str or None, optional
            The title of the plot. If None, the previous title will be used. 
            The default is None.

        Example
        -------
        How to extract ROI data
        >>> data = image_canvas.get_array()
        >>> r0, r1, c0, c1 = ROI.extents
        >>> roi_data = data[r0:r1, c0:c1]
        >>> histogram_canvas.update_canvas(data, roi_data)

        '''
    # Run generic update actions
        self._update_canvas()

    # Clear the canvas if it is already populated
        if not self.is_empty():
            self.ax.clear()

    # Set the title
        if title is not None:
            self.set_padded_title(title)

    # Plot the histogram data
        self._hist_data = self._to_hist_data(data)
        hist_kw = {'bins': self._nbins, 'density': self._density, 'log': self._log}
        self.ax.hist(self._hist_data, **hist_kw)

    # Plot the ROI histogram data, if requested
        if roi_data is None:
            self._roi_hist_data = None
        else:
            self._roi_hist_data = self._to_hist_data(roi_data)
            roi_hist_kw = {**hist_kw, 'facecolor': style.HIST_MASK}
            self.ax.hist(self._roi_hist_data, **roi_hist_kw)
    
    # Adjust the x and y lims
        self.update_xylim()

    # Redraw the canvas
        self.draw_idle()


    def reset_view(self) -> None:
        '''
        Show the original histogram view. Fixes issues when clicking home
        button in the Navigation Toolbar.

        '''
        if not self.is_empty():
        # Update the xy limits
            self.update_xylim()

        # Fix image zoom issues when pressing home button multiple times
            if self.fig.get_tight_layout():
                self.fig.tight_layout()

        # Redraw canvas
            self.draw_idle()


    def update_xylim(self) -> None:
        '''
        Update the plot xy limits.
        
        '''
        self.ax.relim()
        self.ax.autoscale()



class ConfMatCanvas(ImageCanvas):

    _annotation_properties = {
        'color': 'w',
        'horizontalalignment': 'center',
        'verticalalignment': 'center',
        'path_effects': [
            mpl_patheffects.withStroke(linewidth=2, foreground='k')
        ]
    }
    _confmat_colormap = 'cividis' # default colormap, may be a user preference
    _title_pad = 5

    def __init__(
        self,
        title: str = 'Confusion Matrix',
        xlab: str = 'Predicted classes',
        ylab: str = 'True classes',
        size: tuple[float, float] = (9.0, 9.0),
        **kwargs
    ) -> None:
        '''
        A canvas class for drawing confusion matrices.

        Parameters
        ----------
        title : str, optional
            Plot title. The default is 'Confusion Matrix'.
        xlab : str, optional
            Name of the x-axis. The default is 'Predicted classes'.
        ylab : str, optional
            Name of the y-axis. The default is 'True classes'.
        size : tuple[float, float], optional
            Size of the canvas. The default is (9, 9).
        **kwargs
            Parent class arguments (see _CanvasBase class).

        '''
    # Call the constructor of the parent class
        super().__init__(size=size, **kwargs)

    # Set main attributes (self._image and self._cbar are inherited from parent)
        self._matrix: npt.NDArray[Any] | None = None

    # Set title and axis labels attributes and initialize the ax
        self.title = title
        self.xlab = xlab
        self.ylab = ylab
        self._init_ax()


    def _init_ax(self) -> None:
        '''
        Reset the ax to default state.

        '''
        self.set_padded_title(self.title)
        self.ax.set_xlabel(self.xlab)
        self.ax.set_ylabel(self.ylab)


    def _set_colorbar(self) -> Colorbar | None:
        '''
        Reimplementation of parent '_set_colorbar' method. It sets a title to
        the colorbar.

        Returns
        -------
        Colorbar | None
            The new colorbar or None, if the current image is empty. 

        '''
        cbar = super()._set_colorbar()
        if cbar is not None:
            cbar.ax.set_title('%')
        return cbar


    def _compute_percentage(self, matrix: npt.NDArray[Any]) -> npt.NDArray[Any]:
        '''
        Transform 'matrix' data to percentage, rounded to one decimal place.

        Parameters
        ----------
        matrix : NDArray[Any]
            Original matrix data.

        Returns
        -------
        NDArray[Any]
            Percentage matrix data.

        '''
        sums = matrix.sum(1).reshape(-1, 1)
        perc = (matrix / sums) * 100
        matrix = np.round(perc, 1)
        return matrix


    def _set_annotations(self, perc: bool) -> None:
        '''
        Populate the canvas with the matrix values. Canvas requires a redraw to
        show the changes.

        Parameters
        ----------
        perc : bool
            Whether annotations should be displayed as percentages.

        '''
    # Exit function if image or matrix data are invalid
        if self._image is None or self._matrix is None:
            return
        
    # Compute percentages if required
        mtx = self._compute_percentage(self._matrix) if perc else self._matrix

    # Define a meshgrid from the matrix shape
        n_rows, n_cols = mtx.shape
        row_ind = np.arange(n_rows)
        col_ind = np.arange(n_cols)
        xx, yy = np.meshgrid(col_ind, row_ind)

    # Annotate each node with the corresponding value
        for t, x, y in zip(mtx.flatten(), xx.flatten(), yy.flatten()):
            self.ax.annotate(t, (x, y), **self._annotation_properties)


    def _remove_annotations(self) -> None:
        '''
        Remove any previous annotation on the canvas. Canvas requires a redraw
        to show the changes.

        '''
        for child in self.ax.get_children():
            if isinstance(child, Annotation):
                child.remove()


    def annotate(self, as_percentage: bool) -> None:
        '''
        Annotate matrix's cells with values. If 'as_percentage' is True, the
        values are displayed as percentages.

        Parameters
        ----------
        as_percentage : bool
            Whether annotations should be displayed as percentages or not.

        '''
        self._remove_annotations()
        self._set_annotations(as_percentage)
        self.draw_idle()


    def _set_ticks(self, labels: Collection[str]) -> None:
        '''
        Set ticks and ticks labels of the confusion matrix.

        Parameters
        ----------
        labels : Collection[str]
            Collection of labels.

        '''
        ticks = np.arange(len(labels))
        self.ax.set_xticks(ticks, labels=labels, fontsize='small', rotation=-60)
        self.ax.set_yticks(ticks, labels=labels, fontsize='small')
        self.ax.tick_params(labelbottom=True, labeltop=False)


    def _set_matrix(self, data: npt.NDArray[Any]) -> AxesImage:
        '''
        Set and return a new matrix image. Canvas requires a redraw to show the
        changes.

        Parameters
        ----------
        data : NDArray[Any]
            Matrix data array.

        Returns
        -------
        AxesImage
            The new matrix image.

        '''
        if self._image is None:
            kwargs = {'cmap': self._confmat_colormap, 'interpolation': 'none'}
            self._image = self.ax.matshow(data, **kwargs)
        else:
            self._image.set_data(data)
            self._image.set_clim(data.min(), data.max())
            extent = (-0.5, data.shape[1] - 0.5, data.shape[0] - 0.5, -0.5)
            self._image.set_extent(extent)

        return self._image 


    def update_canvas(
        self,
        data: npt.NDArray[Any],
        labels: Collection[str],
        perc_ann: bool = True
    ) -> None:
        '''
        Update the canvas with a new matrix.

        Parameters
        ----------
        data : NDArray[Any]
            Confusion matrix array of squared shape.
        labels : Collection[str]
            Names of true/predicted classes in confusion matrix.
        perc_ann : bool, optional
            Whether annotations should be displayed as percentages. The default
            is True.

        '''
        # Run generic update actions
        self._update_canvas()

        # Store original matrix data but use percentages for a nicer colormap
        self._matrix = data
        data = self._compute_percentage(data)
        self._set_matrix(data)

        # Draw true/predicted classes names as X and Y ticks labels
        self._set_ticks(labels)

        # Draw colorbar if requested
        self._toggle_colorbar(self._has_colorbar)

        # Draw annotations (this also redraws the canvas)
        self.annotate(perc_ann)


    def clear_canvas(self) -> None:
        '''
        Reimplemented 'clear_canvas' method that better fits the requirements
        of a matrix canvas.

        '''
    # Run generic deep cleaning actions
        self._clear_canvas(deep_clear=True)

    # Reset properties
        self._matrix = None
        self._image = None

    # Remove the colorbar
        self._clear_colorbar()

    # Re-initialize title and x-y labels
        self._init_ax()
        self.draw_idle()



class SilhouetteCanvas(_CanvasBase):

    _avg_line_properties = {
        'color': 'r',
        'linestyle': '--',
        'linewidth': 2, 
        'path_effects': [mpl_patheffects.withStroke(foreground=style.IVORY)]
    }
    _y_btm_init = 15

    def __init__(self, size: tuple[float, float] = (4.8, 6.4), **kwargs) -> None:
        '''
        A canvas object for displaying Silhouette scores.

        Parameters
        ----------
        size : tuple[float, float], optional
            Size of the canvas. The default is (4.8, 6.4)
        **kwargs
            Parent class arguments (see _CanvasBase class).

        '''
        super().__init__(size=size, init_blank=False, **kwargs)
        self._init_ax()


    def _init_ax(self) -> None:
        '''
        Reset the ax to default state.

        '''
        self.ax.set_xlabel('Silhouette Coefficient')
        self.ax.set_ylabel('Cluster')
        self.ax.set_yticklabels([])
        self.ax.set_yticks([])


    def alter_colors(self, palette: dict[int, tuple[int, int, int]]) -> None:
        '''
        Change colors of clusters.

        Parameters
        ----------
        palette : dict[int, tuple[int, int, int]]
            Palette dictionary (-> cluster_id : RGB_color).

        '''
    # Format the RGB palette colors as matplotlib compatible floating values
        colors = dict(zip(palette.keys(), rgb_to_float(palette.values())))

    # Iterate through clusters (artists) and use their label as identifier
        for artist in self.ax.get_children():
            if isinstance(artist, PolyCollection):
                id_ = int(str(artist.get_label()))
                artist.set_facecolor(colors[id_])

    # Redraw the canvas
        self.draw_idle()


    def update_canvas(
        self,
        sil_values: dict[int, np.ndarray],
        sil_avg: float,
        title: str, 
        palette: dict[int, tuple[int, int, int]]
    ) -> None:
        '''
        Draw a new silhouette plot.

        Parameters
        ----------
        sil_values : dict[int, numpy array]
            Sorted silhouette scores per cluster (-> cluster_id: scores_array). 
        sil_avg : float
            Average silhouette score.
        title : str
            Title label for the plot.
        palette : dict[int, tuple[int, int, int]]
            Palette dictionary (-> cluster_id: RGB_color).

        '''
    # Run generic update actions
        self._update_canvas()

    # Reset the ax and set the new title
        self.ax.clear()
        self._init_ax()
        self.set_padded_title(title)

    # Format the RGB palette colors as matplotlib compatible floating values
        colors = dict(zip(palette.keys(), rgb_to_float(palette.values())))

    # Adjust the initial vertical (y) padding
        y_btm = self._y_btm_init

    # Plot silhouettes, using y padding as a reference
        for clust_id, values in sil_values.items():
            y_top = y_btm + len(values)
            self.ax.fill_betweenx(
                y = np.arange(y_btm, y_top),
                x1 = 0,
                x2 = values,
                linewidth = 0.3,
                facecolor = colors[clust_id],
                edgecolor = style.IVORY, 
                label = str(clust_id)
            )
            y_btm = y_top + self._y_btm_init

    # Draw the average silhouette score
        self.ax.axvline(x=sil_avg, **self._avg_line_properties)

    # Render the plot
        self.draw_idle()


    def clear_canvas(self) -> None:
        '''
        Clear out the canvas.

        '''
        # Run generic deep cleaning actions and re-initialize X-Y labels 
        self._clear_canvas(deep_clear=True)
        self._init_ax()
        self.draw_idle()



class PieCanvas(_CanvasBase):

    def __init__(
        self,
        perc_fmt: str | None = '%d%%',
        tridimensional: bool = True, 
        size: tuple[float, float] = (5.0, 5.0),
        **kwargs
    ) -> None:
        '''
        A canvas class to display pie charts.

        Parameters
        ----------
        perc_fmt : str or None, optional
            String format for percentage values. The default is '%d%%'.
        tridimensional : bool, optional
            Whether the pie's wedges should pop out for a 3D effect. The 
            default is True.
        size : tuple[float, float], optional
             Width and height (in inches) of the canvas. The default is (5.0,
             5.0).
        **kwargs
            Parent class arguments (see _CanvasBase class).

        '''
        super().__init__(size=size, **kwargs)
    
    # Set a fixed size policy to avoid unwanted rescaling of the pie chart
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    # Set main attributes
        self._3d = tridimensional
        self._perc_fmt = perc_fmt


    def _init_ax(self) -> None:
        '''
        Reset the ax to default state.

        '''
        self._clear_canvas(deep_clear=True)

    # Legend must be cleared to avoid stacking of multiple legend boxes
        self.fig.legends.clear()
        
    # Equal aspect ratio ensures that the pie is drawn as a circle
        self.ax.axis('equal') 


    def update_canvas(
        self,
        data: Sequence[int | float],
        labels: Sequence[str],
        title: str = '', 
        legend: bool = True
    ) -> None:
        '''
        Draw a new pie chart.

        Parameters
        ----------
        data : Sequence[int | float]
            Values that populate the pie.
        labels : Sequence[str]
            Labels for each wedge.
        title : str, optional
            Plot title. The default is ''.
        legend : bool, optional
            Whether to include a legend. The default is True.

        '''
    # Intialize the canvas and the ax
        self._update_canvas()
        self._init_ax()
       
    # Set title
        if title:
            self.set_padded_title(title)
        
    # Adjust wedges and labels layout parameters
        expl = [0.1] * len(data) if self._3d else None
        lbldist = 0.4 if self._perc_fmt is None else 1.1

    # Draw pie chart
        self.ax.pie(
            x=data,
            explode=expl,
            autopct=self._perc_fmt,
            shadow=self._3d,
            labels=labels,
            labeldistance=None if legend else lbldist
        )
    
    # Draw legend
        if legend:
            self.fig.subplots_adjust(left=0.2, bottom=0.2)
            self.fig.legend(
                loc='lower left',
                fontsize='small',
                bbox_to_anchor=(0, 0)
            )
        
    # Render the plot
        self.draw_idle()


    def clear_canvas(self) -> None:
        '''
        Clear out the canvas.

        '''
        self._init_ax()
        self.draw_idle()



type CurvePlotData = tuple[Sequence[float | int], Sequence[float | int]] # (x_coords, y_coords)
class CurveCanvas(_CanvasBase):

    _grid_properties = {'linestyle': ':', 'linewidth': 0.5, 'zorder': 0}
    _legend_properties = {
        'facecolor': style.IVORY,
        'framealpha': 1,
        'frameon': True,
        'loc': 'best'
    }

    def __init__(
        self,
        title: str = '',
        xlab: str = '',
        ylab: str = '',
        grid: bool = True,
        **kwargs
    ) -> None:
        '''
        A canvas class to plot and dynamically update curves.

        Parameters
        ----------
        title : str, optional
            Plot title. The default is ''.
        xlab : str, optional
            X-axis label. The default is ''.
        ylab : str, optional
            Y-axis label. The default ''.
        grid : bool, optional
            Whether to render a grid. The default is True.
        **kwargs
            Parent class arguments (see _CanvasBase class).

        '''
        super().__init__(**kwargs)

    # Set main attributes
        self._title = title
        self._xlab = xlab
        self._ylab = ylab
        self._grid_on = grid

    # Initialize the ax
        self._init_ax()


    def _init_ax(self) -> None:
        '''
        Reset the ax to default state.

        '''
        self.set_padded_title(self._title)
        self.ax.set_xlabel(self._xlab)
        self.ax.set_ylabel(self._ylab)
        self.ax.grid(self._grid_on, **self._grid_properties)


    def _add_curve(
        self,
        xdata: npt.ArrayLike | float | int,
        ydata: npt.ArrayLike | float | int,
        label: str = '',
        color: MplColor | None = None
    ) -> None:
        '''
        Add a new curve. Warning: drawing is not performed by this method.

        Parameters
        ----------
        xdata : ArrayLike | float | int
            X coordinate(s) of the point(s) of the curve.
        ydata : ArrayLike | float | int
            Y coordinate(s) of the point(s) of the curve.
        label : str, optional
            Curve name. The default is ''.
        color : MplColor | None, optional
            A valid matplotlib color. If None, a default color will be selected
            based on current matplotlib style. The default is None.

        '''
        self.ax.plot(xdata, ydata, label=label, color=color)


    def has_curves(self) -> bool:
        '''
        Check if canvas is populated with curves.

        Returns
        -------
        bool
            Whether curves are currently drawn.

        '''
        return len(self.ax.lines) > 0


    def update_canvas(
        self,
        curves: Collection[CurvePlotData],
        labels: Collection[str], 
        colors: Collection[tuple[int, int, int] | None] | None = None
    ) -> None:
        '''
        Update existent curves in the plot. If curve is not existent, add it as
        a new curve. Curve existence is determined via its label.

        Parameters
        ----------
        curves : Collection[CurvePlotData]
            Collection of (x, y) coordinates for each curve to plot.
        labels : Collection[str]
           Unique labels for each curve.
        colors : Collection[tuple[int, int, int] | None] | None, optional
            List of colors for each curve as RGB triplets. If None or empty,
            default colors will be selected based on current matplotlib style.
            The default is None.

        Raises
        ------
        ValueError
            Raised if 'labels' contains multiple instances of the same value.
        ValueError
            Raised if 'data', 'labels' and 'colors' lists have different
            lengths. This is not raised when 'colors' is an empty list.

        '''
        self._update_canvas()

    # Check for unique labels
        if len(set(labels)) < len(labels):
            raise ValueError('"Labels" must be unique names.')
        
    # Set colors to a list of None if not specified
        if colors:
            mpl_col = [None if c is None else rgb_to_float((c,))[0] for c in colors]
        else:
            mpl_col = [None] * len(curves)

    # Check that all parameters have same length
        if not len(curves) == len(labels) == len(mpl_col):
            raise ValueError('"Data", "labels" and "colors" must have same length.')

    # Determine existent curves
        existent_curves = self.ax.lines
        existent_labels = [ec.get_label() for ec in existent_curves]

    # Add new curves and update existent ones
        for cur, lbl, col in zip(curves, labels, mpl_col):
            if lbl in existent_labels:
                existent_curves[existent_labels.index(lbl)].set_data(cur)
            else:
                self._add_curve(*cur, label=lbl, color=col)

    # Rescale the plot
        self.ax.relim()
        self.ax.autoscale_view()

    # Draw the legend
        self.ax.legend(**self._legend_properties)
    
    # Render the plot
        self.draw_idle()


    def reset_view(self) -> None:
        '''
        Show the original plot view. Fixes issues when clicking home button in
        the Navigation Toolbar.

        '''
        self.ax.relim()
        self.ax.autoscale()
        self.draw_idle()


    def clear_canvas(self) -> None:
        '''
        Clear out the canvas.
        
        '''
        self._clear_canvas(deep_clear=True)
        self._init_ax()
        self.draw_idle()



class NavTbar(backend_qt.NavigationToolbar2QT):

    def __init__(
        self,
        canvas: _CanvasBase,
        parent: QWidget | None = None, 
        orient: Qt.Orientation = Qt.Orientation.Horizontal,
        coords: bool = True
    ) -> None:
        '''
        A class to provide a navigation toolbar linked to a canvas.

        Parameters
        ----------
        canvas : _CanvasBase
            The canvas linked to the toolbar.
        parent : QWidget or None, optional
            The GUI parent of the toolbar. The default is None.
        orient : Qt.Orientation, optional
            The orientation of the toolbar. The default is Qt.Horizontal.
        coords : bool, optional
            Whether to display coordinates in the toolbar. The default is True.

        '''
        super().__init__(canvas, parent, coords)

    # Set the main attributes of the class
        self._orient = orient

    # Set custom icons
        icons_dict = {
            'Home': style.getIcon('ZOOM_DEFAULT'),
            'Pan' : style.getIcon('PAN'),
            'Zoom': style.getIcon('ZOOM'),
            'Save': style.getIcon('SAVE')
        }
        for a in self.actions():
            if icon := icons_dict.get(a.text()):
                a.setIcon(icon)

    # Set the orientation of the toolbar (vertical/horizontal)
        self.setOrientation(self._orient)

    # Avoid coords and data label to be cut off
        if coords:
            self.locLabel.setSizePolicy(style.SP_EXPANDING, style.SP_MINIMUM)

    # Set the toolbar style-sheet
        self.setStyleSheet(style.SS_TOOLBAR)

    
    @classmethod
    def imageCanvasDefault(
        cls,
        canvas: ImageCanvas,
        parent: QWidget | None = None,
        orient: Qt.Orientation = Qt.Orientation.Horizontal
    ) -> 'NavTbar':
        '''
        Convenient method to instantiate a Navigation Toolbar attached to an
        Image Canvas with default actions and fixed home action. 

        Parameters
        ----------
        canvas : ImageCanvas
            The image canvas object linked to the toolbar. 
        parent : QWidget or None, optional
            The GUI parent of the toolbar. The default is None.
        orient : Qt.Orientation, optional
            The orientation of the toolbar. The default is Qt.Horizontal.

        Returns
        -------
        NavTbar
            A new instance of Navigation Toolbar.

        '''
        navtbar = cls(canvas, parent, orient)
        navtbar.fixHomeAction()
        navtbar.removeToolByIndex([3, 4, 8, 9])
        return navtbar
    

    @classmethod
    def barCanvasDefault(
        cls,
        canvas: BarCanvas,
        parent: QWidget | None = None,
        orient: Qt.Orientation = Qt.Orientation.Horizontal
    ) -> 'NavTbar':
        '''
        Convenient method to instantiate a Navigation Toolbar attached to a Bar
        Canvas with default actions and custom show label percentage action. 

        Parameters
        ----------
        canvas : BarCanvas
            The barplot canvas object linked to the toolbar. 
        parent : QWidget or None, optional
            The GUI parent of the toolbar. The default is None.
        orient : Qt.Orientation, optional
            The orientation of the toolbar. The default is Qt.Horizontal.

        Returns
        -------
        NavTbar
            A new instance of Navigation Toolbar.

        '''
        navtbar = cls(canvas, parent, orient, coords=False)
        navtbar.removeToolByIndex(list(range(2, 10)))
        navtbar.insertLabelizeAction(10)
        return navtbar
    

    @classmethod
    def histCanvasDefault(
        cls,
        canvas: HistogramCanvas, 
        parent: QWidget | None = None,
        orient: Qt.Orientation = Qt.Orientation.Horizontal
    ) -> 'NavTbar':
        '''
        Convenient method to instantiate a Navigation Toolbar attached to a
        Histogram Canvas with default actions and custom log scale action.

        Parameters
        ----------
        canvas : HistogramCanvas
            The histogram canvas object linked to the toolbar. 
        parent : QWidget or None, optional
            The GUI parent of the toolbar. The default is None.
        orient : Qt.Orientation, optional
            The orientation of the toolbar. The default is Qt.Horizontal.

        Returns
        -------
        NavTbar
            A new instance of Navigation Toolbar.

        '''
        navtbar = cls(canvas, parent, orient, coords=False)
        navtbar.removeToolByIndex([3, 4, 8, 9])
        navtbar.fixHomeAction()
        navtbar.insertLogscaleAction(2)
        return navtbar
    

    def showToolbarAction(self) -> QAction:
        '''
        Return the hidden checkable 'show navigation toolbar' action.

        Returns
        -------
        QAction
            Show toolbar action.

        '''
        return self.findChildren(QAction)[1]
    

    def insertLabelizeAction(self, before_idx: int) -> None:
        '''
        Convenient method to include the labelize action into a Navigation 
        Toolbar which is linked to a Bar Canvas.

        Parameters
        ----------
        before_idx : int
            Before action index.

        '''
    # Do nothing if the linked canvas object is not a barplot canvas
        if not isinstance(self.canvas, BarCanvas):
            return
        
        self.lbl_action = QAction(style.getIcon('PERCENT'), 'Show amounts')
        self.lbl_action.setCheckable(True)
        self.lbl_action.toggled.connect(self.canvas.toggle_amounts)
        self.insertAction(before_idx, self.lbl_action)


    def insertLogscaleAction(self, before_idx: int) -> None:
        '''
        Convenient method to include the log scale action into a Navigation 
        Toolbar which is linked to a Histogram Canvas.

        Parameters
        ----------
        before_idx : int
            Before action index.

        '''
    # Do nothing if the linked canvas object is not a Histogram canvas
        if not isinstance(self.canvas, HistogramCanvas):
            return
        
        self.log_action = QAction(style.getIcon('LOG'), 'Log scale')
        self.log_action.setCheckable(True)
        self.log_action.setChecked(self.canvas.get_logscale())
        self.log_action.toggled.connect(self.canvas.set_logscale)
        self.insertAction(before_idx, self.log_action)
        

    def fixHomeAction(self) -> None:
        '''
        Fix the home action (zoom to default view) of the canvas. Canvas must
        implement the 'reset_view' method.

        '''
        # Check that canvas has implemented the method 'reset_view'
        reset_view = getattr(self.canvas, "reset_view", None)
        if callable(reset_view):
            for a in self.actions():
                if a.text() == 'Home':
                    a.triggered.connect(reset_view)
                    return


    def insertAction(self, before_idx: int, action: QAction) -> None:
        '''
        Convenient reimplementation of 'insertAction' method, that allows 
        accessing the 'before_action' parameter through its position (index).

        Parameters
        ----------
        before_idx : int
            Before action index.
        action : QAction
            Action to insert.

        '''
        before_action = self.findChildren(QAction)[before_idx]
        super().insertAction(before_action, action)


    def insertActions(self, before_idx: int, actions: list[QAction]) -> None:
        '''
        Convenient reimplementation of 'insertActions' method, that allows
        accessing the 'before_action' parameter through its position (index).

        Parameters
        ----------
        before_idx : int
            Before action index.
        actions : list of QAction
            Actions to insert.

        '''
        before_action = self.findChildren(QAction)[before_idx]
        super().insertActions(before_action, actions)

    
    def insertSeparator(self, before_idx: int) -> None:
        '''
        Convenient reimplementation of 'insertSeparator' method, that allows
        accessing the 'before_action' parameter through its position (index).

        Parameters
        ----------
        before_idx : int
            Before action index.

        '''
        before_action = self.findChildren(QAction)[before_idx]
        super().insertSeparator(before_action)


    def getTrueActions(self) -> list[QAction]:
        '''
        Returns a list of the true actions (= NOT QWidgetActions) held by the
        toolbar.

        Returns
        -------
        actions : list[QAction]
            True actions.

        '''
        actions = self.actions()
        actions = [a for a in actions if not isinstance(a, QWidgetAction)]
        return actions


    def removeToolByIndex(self, indices: list[int]) -> None:
        '''
        Remove actions from the toolbar by their indices.

        Parameters
        ----------
        indices : list of int
            List of indices of the actions that should be excluded.

        '''
        for i in indices:
            self.removeAction(self.findChildren(QAction)[i])


    def resizeEvent(self, event: QResizeEvent):
        '''
        Reimplementation of the 'resizeEvent' method, that updates the minimum
        height of the toolbar.

        Parameters
        ----------
        event : QResizeEvent
            The resize event.

        '''
        new_h = event.size().height()
        old_h = event.oldSize().height()
        if new_h > old_h and self._orient == Qt.Orientation.Horizontal:
            self.setMinimumHeight(new_h)



class RoiPatch(Rectangle):

    def __init__(
        self,
        bbox: tuple[float, float, int, int],
        color: tuple[int, int, int],
        filled: bool
    ) -> None:
        '''
        Reimplementation of a Matplotlib rectangle patch, useful to manage and 
        display ROIs.

        Parameters
        ----------
        bbox : tuple[float, float, int, int]
            ROI bounding box (x0, y0, width, height).
        color : tuple[int, int, int]
            Patch color as an RGB triplet.
        filled : bool
            Whether the ROI patch should be filled.

        '''
        x0, y0, w, h = bbox
        (mpl_col,) = rgb_to_float((color,))
        super().__init__((x0, y0), w, h, linewidth=2, color=mpl_col, fill=filled)



class RoiAnnotation(Annotation):

    def __init__(
        self,
        text: str,
        anchor_patch: RoiPatch,
        xy: tuple[float, float] = (0.0, 1.0)
    ) -> None:
        '''
        Reimplementation of a Matplotlib text annotation, useful to manage and 
        display ROIs names.

        Parameters
        ----------
        text : str
            ROI name. If the hyphen symbol ('-') is used, the annotation is 
            hidden.
        anchor_patch : RoiPatch
            ROI to which attach the annotation.
        xy : tuple[float, float], optional
            Point of anchor with respect to the anchor_patch. For example, 
            (0.0, 1.0) means top left corner. The default is (0.0, 1.0).

        '''
        super().__init__(
            text = '' if text == '-' else text, # hyphen hides ROI annotation
            xy = xy,
            xycoords = anchor_patch,
            bbox = dict(boxstyle='round', fc=style.IVORY, ec=style.BLACK_PEARL),
            annotation_clip=True
        )
        

    def set_text(self, s: str) -> None:
        '''
        Reimplementation of parent 'set_text' method. Passing an hyphen string
        will clear the annotation text, hiding the annotation box. 

        Parameters
        ----------
        s : str
            New text string.

        '''
        super().set_text('' if s == '-' else s)



class Crosshair(mpl_widgets.MultiCursor): # !!! Not used yet

    def __init__(self, axes: list[Axes] | tuple[Axes, ...]) -> None:
        '''
        A reimplemented Matplotlib multicursor widget [NOT USED YET].

        Parameters
        ----------
        axes : list[Axes] or tuple[Axes, ...]
            List of axis where the cursor should be rendered.

        '''
        super().__init__(
            canvas=None, # deprecated (see 'MultiCursor' class for details)
            axes=axes,
            useblit=True,
            horizOn=True,
            vertOn=True,
            color=style.BLACK_PEARL,
            linewidth=1
        )



class PolySel(mpl_widgets.PolygonSelector): # future improvement to ROIs

    def __init__(self, ax: Axes, onselect: Callable, interactive: bool = True) -> None:
        '''
        A reimplemented Matplotlib polygon selector widget [NOT USED YET]

        Parameters
        ----------
        ax : Axes
            The ax where the selector must be drawn.
        onselect : Callable
            Callback function that is called after the selection is created.
        interactive: bool, optional
            If True, a bounding box will be drawn around the selector once it
            is complete. This box can be used to move and resize the selector.
            The default is True

        '''
        # Customize the appearence of the polygon selector
        kwargs = {
            'useblit': True,
            'grab_range': 10,
            'props': {
                'color': style.BLACK_PEARL,
                'alpha': 0.6,
                'linewidth': 2,
            },
            'handle_props': {
                'markerfacecolor': style.BLOSSOM,
                'markeredgecolor': style.BLACK_PEARL,
                'alpha': 1
            },
            'draw_bounding_box': interactive,
            'box_props': {
                'facecolor': style.BLOSSOM_LIGHT,
                'edgecolor': style.BLACK_PEARL,
                'alpha': 0.5, 
                'linewidth': 1,
                'fill': True
            },
            'box_handle_props': {
                'markerfacecolor': style.CASPER_DARK,
                'markeredgecolor': style.BLACK_PEARL,
                'alpha': 0.6
            }
        }
        super().__init__(ax, onselect, **kwargs)
    
    # By default the selector is turned off
        self.set_active(False)


    def update(self) -> None:
        '''
        Reimplementation of the default 'update' method. It also calls the
        'updateCursor' method after the default update operations are done.

        '''
        super().update()
        self.updateCursor()


    def updateCursor(self) -> None:
        '''
        Updates the cursor depending on the state of the selector. When it is
        active, the pointing hand cursor is set, otherwise the arrow cursor.

        '''
        if self.canvas is None:
            return

        if self.active:
            cursor = Qt.CursorShape.PointingHandCursor
        else:
            cursor = Qt.CursorShape.ArrowCursor

        self.canvas.setCursor(QCursor(cursor)) # pyright: ignore[reportAttributeAccessIssue]



class RectSel(mpl_widgets.RectangleSelector):

    def __init__(
        self,
        ax: Axes,
        onselect: Callable,
        interactive: bool = True,
        btns: list[int] | None = None
    ) -> None:
        '''
        A reimplemented Matplotlib rectangle selector widget, tailored for ROI 
        selection.

        Parameters
        ----------
        ax : Axes
            The ax where the selector must be drawn.
        onselect : Callable
            Callback function that is called after the selection is created.
        interactive : bool, optional
            Whether a drawn rectangle selector can be moved or resized. The 
            default is True.
        btns : list[int] or None, optional
            List of mouse buttons that can trigger the drawing event. Left = 1,
            Middle = 2 and Right = 3. If None, all buttons are included. The
            default is None.

        '''
    # Customize the appearence of the rectangle selector
    # drag_from_anywhere=True causes a rendering glitch when a former selection
    # is active, you draw a new one and, without releasing the left button, you
    # start resizing it. [==> This seems to be fixed now!]
        kwargs = {
            'minspanx': 1,
            'minspany': 1,
            'useblit': True,
            'props': {
                'facecolor': style.BLACK_PEARL,
                'edgecolor': style.BLOSSOM,
                'alpha': 0.6, 
                'linewidth': 2,
                'fill': True
            },
            'spancoords': 'data',
            'button': btns,
            'grab_range': 10,
            'handle_props': {
                'markerfacecolor': style.BLOSSOM,
                'markeredgecolor': style.BLACK_PEARL,
                'alpha': 1
            },
            'interactive': interactive,
            'drag_from_anywhere': True
        }
        super().__init__(ax, onselect, **kwargs)

    # By default the selector is turned off
        self.set_active(False)


    def fixed_extents(
        self,
        shape: tuple[int, int],
        fmt: Literal['matrix', 'xy'] = 'matrix',
        mode: Literal['full'] = 'full'
    ) -> tuple[int, int, int, int] | None:
        '''
        Get integer extents of the selector after checking if it lies within
        the provided 'shape'. Different output extents format can be selected.

        Parameters
        ----------
        shape : tuple[int, int]
            Control shape. It must be provided as (rows, cols).
        fmt : Literal['matrix', 'xy'], optional
            The output format of coords. If 'matrix' the coords format is
            (row_min, row_max, col_min, col_max). If 'xy' the coords format is
            (x_min, x_max, y_min, y_max). The default is 'matrix'.
        mode : Literal['full'], optional
            How to treat pixels at the border of the selector region. At the
            moment the only supported mode is 'full', which means that only
            the pixels that are entirely covered by the selector region will be
            included. The default is 'full'.

        Returns
        -------
        extents : tuple[int, int, int, int] or None
            Fixed extents as integer indices. If the selector region falls
            entirely outside the map area, extents will be None.

        Raises
        ------
        ValueError
            Raised if 'fmt' parameter is not 'matrix' or 'xy'.
        ValueError
            Raised if 'mode' parameter is not 'full'.  

        '''
    # Get the default extents
        xmin, xmax, ymin, ymax = self.extents

    # Exclude pixels not entirely selected --> 'mode' : 'full'
    # ymax and xmax should be decreased by one (-1) before being rounded
    # but this is skipped due to range and/or array slice mechanics,
    # where the second index is always excluded -> [xmin, xmax)
        match mode:
            case 'full':
                xmin = round(xmin + 1)
                xmax = round(xmax)
                ymin = round(ymin + 1)
                ymax = round(ymax)
            case _:
                raise ValueError(f'{mode} is not a valid mode.')

    # Exit function if the extents are completely outside the map
        if xmax < 0 or xmin > shape[1] or ymax < 0 or ymin > shape[0]:
            return None

    # Fix extents that are partially outside the map borders
        if xmin < 0: xmin = 0
        if xmax > shape[1]: xmax = shape[1]
        if ymin < 0: ymin = 0
        if ymax > shape[0]: ymax = shape[0]

    # Return the fixed extents formatted according to 'fmt'
        match fmt:
            case 'matrix':
                extents = (ymin, ymax, xmin, xmax)
            case 'xy':
                extents = (xmin, xmax, ymin, ymax)
            case _:
                raise ValueError(f'{fmt} is not a valid format.')

        return extents


    def fixed_rect_bbox(
        self,
        shape: tuple[int, int],
        mode: Literal['full'] = 'full'
    ) -> tuple[float, float, int, int] | None:
        '''
        Get the bounding box of the selector after checking if it lies within
        the provided shape.

        Parameters
        ----------
        shape : tuple[int, int]
            Control shape provided as (rows, cols).
        mode : Literal['full'], optional
            How to treat pixels at the border of the selector region.
            Default is 'full'.

        Returns
        -------
        bbox : tuple[float, float, int, int] or None
            Fixed bounding box (x0, y0, width, height). If the selector region
            falls entirely outside the map area, returns None.

        '''
        # Get fixed_extents in 'xy' format
        extents = self.fixed_extents(shape, fmt='xy', mode=mode)
        if extents is None:
            return None

        # Convert extents to bbox
        xmin, xmax, ymin, ymax = extents
        x0 = xmin - 0.5
        y0 = ymin - 0.5
        w = xmax - xmin
        h = ymax - ymin

        # Bbox is invalid if it has non-positive height or width (safety)
        if w < 1 or h < 1:
            return None 
        return (x0, y0, w, h)


    # def fixed_rect_bbox(
    #     self,
    #     shape: tuple[int, int],
    #     mode: Literal['full'] = 'full'
    # ) -> tuple[float, float, int, int] | None:
    #     '''
    #     Get the bounding box of the selector after checking if it lies within
    #     the provided shape.

    #     Parameters
    #     ----------
    #     shape : tuple[int, int]
    #         Control shape. It must be provided as (rows, cols).
    #     mode : Literal['full'], optional
    #         How to treat pixels at the border of the selector region. At the
    #         moment the only supported mode is 'full', which means that only
    #         the pixels that are entirely covered by the selector region will be
    #         included. The default is 'full'.

    #     Returns
    #     -------
    #     bbox : tuple[float, float, int, int] or None
    #         Fixed bounding box (x0, y0, width, height). If the selector region
    #         falls entirely outside the map area, it will be None.

    #     Raises
    #     ------
    #     ValueError
    #         Raised if 'mode' parameter is not 'full'.  

    #     '''
    # # Get the default bounding box
    #     x0, y0, w, h = self._rect_bbox

    # # Exit function if the bounding box is completely outside the map
    #     if (
    #         x0 + 0.5 > shape[1]
    #         or x0 + w < -0.5
    #         or y0 + 0.5 > shape[0]
    #         or y0 + h < -0.5
    #     ):
    #         return None

    # # Fix extents that are partially outside the map borders
    #     if x0 < -0.5:
    #         w += x0 + 0.5
    #         x0 = -0.5
    #     if x0 + w + 0.5 > shape[1]:
    #         w = shape[1] - 0.5 - x0
    #     if y0 < -0.5:
    #         h += y0 + 0.5
    #         y0 = -0.5
    #     if y0 + h + 0.5 > shape[0]:
    #         h = shape[0] - 0.5 - y0

    # # Exclude pixels not entirely selected --> 'mode' : 'full'
    #     match mode:
    #         case 'full':
    #             _x0 = round(x0 + 1) - 0.5
    #             _y0 = round(y0 + 1) - 0.5
    #             w = int(w + x0 - _x0)
    #             h = int(h + y0 - _y0)
    #         case _:
    #             raise ValueError(f"'{mode}' is not a valid mode.")
    
    # # Bbox is invalid if it has non-positive height or width (safety)
    #     if w < 1 or h < 1:
    #         return None

    # # Return the fixed bounding box
    #     bbox = (_x0, _y0, w, h)
    #     return bbox


    def update(self) -> None:
        '''
        Reimplementation of the default 'update' method. It also calls the
        'updateCursor' method after the default update operations are done.

        '''
        super().update()
        self.updateCursor()


    def updateCursor(self) -> None:
        '''
        Updates the cursor depending on the state of the selector. When it is
        active, the pointing hand cursor is set, otherwise the arrow cursor.

        '''
        if self.canvas is None:
            return

        if self.active:
            cursor = Qt.CursorShape.PointingHandCursor
        else:
            cursor = Qt.CursorShape.ArrowCursor
    
        self.canvas.setCursor(QCursor(cursor)) # pyright: ignore[reportAttributeAccessIssue]



class HeatmapScaler(mpl_widgets.SpanSelector):

    def __init__(
        self,
        ax: Axes,
        onselect: Callable,
        interactive: bool = True,
        btns: list[int] | None = None
    ) -> None:
        '''
        A reimplemented Matplotlib span selector widget, tailored for scaling
        histograms linked to heatmaps.

        Parameters
        ----------
        ax : Axes
            The ax where the span selection is performed.
        onselect : Callable
            The function that triggers after a selection event.
        interactive : bool, optional
            Whether the selector is interactive. The default is True.
        btns : list[int] or None, optional
            List of buttons that can trigger the selection. Can be left mouse 
            button [1], wheel mouse button [2], right mouse button [3] or None,
            which means all the buttons. The default is None.

        '''
    # Customize the appearence of the span selector
        kwargs = {
            'direction': 'horizontal',
            'minspan': 0,
            'useblit': True,
            'props': {
                'facecolor': style.BLOSSOM,
                'edgecolor': style.BLOSSOM,
                'alpha': 0.6,
                'fill': True
            },
            'interactive': interactive,
            'button': btns,
            'handle_props': {
                'color': style.BLOSSOM,
                'linewidth': 2
            },
            'grab_range': 10,
            'drag_from_anywhere': True
        }
        super().__init__(ax, onselect, **kwargs)

    # By default the selector is turned off
        self.set_active(False)



def rgb_to_float(
    rgb_list: Iterable[tuple[int, int, int]]
) -> list[tuple[float, float, float]]:
    '''
    Convert RGB values to Matplotlib compatible floating values, ranging in
    [0, 1].

    Parameters
    ----------
    rgb_list : Iterable[tuple[int, int, int]]
        An iterable of RGB triplets.

    Returns
    -------
    list[tuple[float, float, float]]
        A list of converted float RGB triplets.

    '''
    float_rgb = [(r/255, g/255, b/255) for (r, g, b) in rgb_list]
    return float_rgb