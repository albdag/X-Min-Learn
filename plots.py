# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 09:14:45 2023

@author: albdag
"""

from collections.abc import Callable, Iterable

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor, QResizeEvent
from PyQt5.QtWidgets import QAction, QMenu, QSizePolicy, QWidget, QWidgetAction

import numpy as np

from matplotlib.axes import Axes
from matplotlib.backends import backend_qtagg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.text import Annotation
from matplotlib import backend_bases as mpl_backend_bases
from matplotlib import colormaps as mpl_colormaps
from matplotlib import colors as mpl_colors
from matplotlib import patheffects as mpl_patheffects
from matplotlib import style as mpl_style
from matplotlib import widgets as mpl_widgets
from mpl_interactions import panhandler
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import preferences as pref
import style

# mpl.use('Qt5Agg')


class PanHandler(panhandler):

    _pressCursor = QCursor(Qt.SizeAllCursor)
    _releaseCursor = QCursor(Qt.ArrowCursor)

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
        Reimplementation of the '_press' method. It just changes the cursor.

        Parameters
        ----------
        event : Matplotlib MouseEvent
            The mouse press event.

        '''
        super()._press(event)
        if event.button == self.wheelButton:
            self._releaseCursor = self.fig.canvas.cursor()
            self.fig.canvas.setCursor(self._pressCursor)


    def _release(self, event: mpl_backend_bases.MouseEvent) -> None:
        '''
        Reimplementation of the '_release' method. It just changes the cursor.

        Parameters
        ----------
        event : Matplotlib MouseEvent
            The mouse release event.

        '''
        super()._release(event)
        if event.button == self.wheelButton:
            self.fig.canvas.setCursor(self._releaseCursor)



class _CanvasBase(backend_qtagg.FigureCanvasQTAgg):

    _title_pad = 5

    def __init__(
        self,
        size: tuple[float, float] = (6.4, 4.8),
        layout: str = 'none',
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
        layout : str, optional
            Layout engine of the figure. One of ['constrained', 'compressed', 
            'tight' and 'none']. The default is 'none'.
        wheel_zoom : bool, optional
            Whether or not allow zooming with mouse wheel. The default is True.
        wheel_pan : bool, optional
            Whether or not allow drag-pan with pressed mouse wheel. The default
            is True.
        init_blank : bool, optional
            Whether ax should be initialized as blank. The default is True.

        '''
    # Define the figure and the ax of the matplotlib canvas
        self.fig = Figure(
            figsize=size,
            facecolor=style.IVORY, 
            edgecolor=style.BLACK_PEARL,
            linewidth=2, 
            layout=layout
        )
        self.ax = self.fig.add_subplot(111, facecolor=style.BLACK_PEARL)
        self.ax.patch.set(edgecolor=style.SAN_MARINO, linewidth=3)
        
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
        self.setContextMenuPolicy(Qt.CustomContextMenu)


    def has_toolbar(self) -> bool:
        '''
        Check if canvas has a linked navigation toolbar.

        Returns
        -------
        bool
            Whether the canvas has a toolbar.

        '''
        return hasattr(self, "toolbar") and self.toolbar is not None


    def clear_canvas(self, deep_clear: bool = False) -> None:
        '''
        Generic canvas clearing actions. To be reimplemented in each subclass.

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
        if self.has_toolbar():
            self.toolbar.update()


    def mouseMoveEvent(self, event: mpl_backend_bases.MouseEvent) -> None:
        '''
        Set focus on the canvas widget when mouse hovering. Necessary to detect
        the key attribute of the mouse wheel zoom event. See 'wheel_zoom' 
        method for further details.

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
        menu = QMenu()
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
        shared_x = self.ax._shared_axes['x']
        shared_y = self.ax._shared_axes['y']
    # For older version of Matplotlib
        # shared_x = self.ax._shared_x_axes
        # shared_y = self.ax._shared_y_axes

        if share:
            shared_x.join(self.ax, ax)
            shared_y.join(self.ax, ax)
        else:
            shared_x.remove(ax)
            shared_y.remove(ax)



    def update_canvas(self) -> None:
        '''
        Generic canvas update actions. To be reimplemented in each subclass.

        '''
    # Show the axis (borders and ticks)
        self.ax.axis('on')

    # Reset the navigation toolbar stack to fix unwanted glitches with clims 
        if self.has_toolbar():
            self.toolbar.update()


    # Inspired by https://gist.github.com/tacaswell/3144287
    # and https://github.com/mpl-extensions
    def wheelZoomHandler(self, base_scale: float = 1.5) -> None:
        '''
        Wheel zoom factory.

        Parameters
        ----------
        base_scale : float, optional
            Base zoom scale. The default is 1.5.

        '''

        def zoom(event: mpl_backend_bases.MouseEvent) -> Callable:
            '''
            Apply zoom to canvas after mouse wheel event.

            Parameters
            ----------
            event : Matplotlib MouseEvent
                The mouse wheel event.

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
            scale = 1.1 if 'ctrl' in event.modifiers else base_scale

        # Set zoom in or out
            if event.button == 'down':
                scale_factor = 1/scale
            elif event.button == 'up':
                scale_factor = scale
            else:
                scale_factor = 1

        # Set new limits
            self.ax.set_xlim([xdata - (xdata-cur_xlim[0]) / scale_factor,
                              xdata + (cur_xlim[1]-xdata) / scale_factor])
            self.ax.set_ylim([ydata - (ydata-cur_ylim[0]) / scale_factor,
                              ydata + (cur_ylim[1]-ydata) / scale_factor])

        # Redraw the canvas
            self.draw_idle()

    # Attach the callback
        cid = self.mpl_connect("scroll_event", zoom)

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
        self.is_binary = binary
        self.image = None
        self.scaled_clim = None

    # Initialize the colormap
        self.cmap = None

    # Initialize the colorbar
        self.has_cbar = cbar
        self.cbar, self.cax = None, None


    def is_empty(self) -> bool:
        '''
        Check if the canvas is empty.

        Returns
        -------
        bool
            Whether or not the canvas is empty.

        '''
        return self.image is None


    def set_cbar(self) -> None:
        '''
        Set the colorbar ax.

        '''
        divider = make_axes_locatable(self.ax)
        self.cax = divider.append_axes("right", size="5%", pad=0.1)
        # consider using mappable as matplotlib.cm.ScalarMappable instead of self.image
        # From Matplotlib docs: "Note that one can create a ScalarMappable 
        #"on-the-fly" to generate colorbars not attached to a previously drawn
        # artist, e.g.: 
        #fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)"
        self.cbar = self.fig.colorbar(self.image, cax=self.cax, extend='both')


    def set_heatmap_cmap(self, cmap_name: str = 'Spectral_r') -> None:
        '''
        Set a colormap suited to display a heatmap.

        Parameters
        ----------
        cmap_name : str, optional
            Matplotlib colormap name. The default is 'Spectral_r'.

        '''
    # Set colormap
        self.cmap = mpl_colormaps['binary_r' if self.is_binary else cmap_name]

    # Set outliers as grayed-out
        self.cmap.set_over('0.2')
        self.cmap.set_under('0.5')

    # Set masked data as ivory
        self.cmap.set_bad(style.IVORY)


    def set_discretemap_cmap(
        self,
        colors: Iterable[tuple[int, int, int]],
        name: str = 'CustomCmap'
    ) -> None:
        '''
        Set a colormap suited to display a discrete map.

        Parameters
        ----------
        colors : Iterable[tuple[int, int, int]]
            Iterable of RGB triplets.
        name : str, optional
            Colormap name. The default is 'CustomCmap'.

        '''
    # Convert colorlist RGB triplets to float RGB values
        floatRGBA = rgb_to_float(colors)

    # # Filter the colorlist if the map data is masked (required as bug fix)
    #     if self.contains_masked_data():
    #         data, mask = self.get_map(return_mask=True)
    #         floatRGBA = [floatRGBA[u] for u in np.unique((data[~mask]))]

    # Set colormap
        self.cmap = mpl_colors.ListedColormap(floatRGBA, name=name)

    # Set outliers as ivory
        self.cmap.set_over(style.IVORY)
        self.cmap.set_under(style.IVORY)

    # Set masked data as ivory
        self.cmap.set_bad(style.IVORY)

    
    def set_boundary_norm(self, n_colors: int) -> mpl_colors.BoundaryNorm:
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


    def alter_cmap(self, color_arg: str | Iterable[tuple[int, int, int]]) -> None:
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
        if self.is_empty(): 
            return

    # If color_arg is a string, set a heatmap cmap. If it is a list of RGB 
    # triplets, set a discrete cmap. If neither of those, raise a ValueError.
        if isinstance(color_arg, str):
            self.set_heatmap_cmap(color_arg)
        elif isinstance(color_arg, Iterable):
            self.set_discretemap_cmap(color_arg)
        else:
            raise ValueError(f'{color_arg} is not a valid argument.')

    # Set the new colormap and render it
        self.image.set_cmap(self.cmap)
        self.draw_idle()


    def contains_masked_data(self) -> bool:
        '''
        Check if the canvas is displaying masked data.

        Returns
        -------
        bool
            Whether the displayed data is masked.

        '''
        if self.is_empty():
            return False
        else:
            return np.ma.is_masked(self.image.get_array())


    def contains_discretemap(self) -> bool:
        '''
        Check if the canvas is currently displaying a discrete map.

        Returns
        -------
        bool
            Whether the currently displayed data is a discrete map.

        '''
        return isinstance(self.cmap, mpl_colors.ListedColormap)


    def contains_heatmap(self) -> bool:
        '''
        Check if the canvas is currently displaying a heatmap.

        Returns
        -------
        bool
            Whether the currently displayed data is a heatmap.

        '''
        return isinstance(self.cmap, mpl_colors.LinearSegmentedColormap)


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
    # Do nothing if canvas is empty
        if self.is_empty():
            return

    # We use the vmin, vmax args if provided
        if vmin is not None and vmax is not None:
            self.image.set_clim(vmin, vmax)
        # Set colorbar clim. Fixes a bug with item highlight of legends 
            if self.contains_discretemap() and self.cbar is not None:
                self.cbar.mappable.set_clim(vmin, vmax)

    # Otherwise we use the current image clims (reset clims)
        else:
            array = self.get_map()
            vmin, vmax = np.nanmin(array), np.nanmax(array)
            self.image.set_clim(vmin, vmax)
        # Re-apply boundary norm if image is a discrete map
            if self.contains_discretemap():
                norm = self.set_boundary_norm(int(vmax + 1))
                self.image.set_norm(norm)

    # Reset the navigation toolbar stack to fix unwanted glitches with clims 
        if self.has_toolbar:
            self.toolbar.update()


    def clear_canvas(self) -> None:
        '''
        Clear out certain elements of the canvas and resets their properties.

        '''
        if not self.is_empty():

        # Call the parent method to run generic cleaning actions
            super().clear_canvas()

        # ??? Reset the image and its colormap
            self.image.remove()
            self.image, self.cmap = None, None

        # Reset the colorbar and its ax
            self.cbar = None
            if self.cax is not None:
                self.cax.set_visible(False)

        # Remove the title
            self.ax.set_title('')

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
        if not self.is_empty():
            self.image.set_picker(True if enabled else None)
            # self.draw_idle()


    def reset_view(self) -> None:
        '''
        Show the original map view. Fixes issues when clicking home button in
        the Navigation Toolbar.

        '''
    # Do nothing if canvas is empty
        if self.is_empty():
            return
        
        data = self.image.get_array()
        extents = (-0.5, data.shape[1]-0.5, data.shape[0]-0.5, -0.5)
    
    # Fix axes extents
        self.ax.set_xlim(extents[:2])
        self.ax.set_ylim(extents[2:])
    
    # Fix image extents (correct aspect ratio)
        self.image.set_extent(extents)
    
    # Force re-applying current clims (fix colormap bug of discrete maps)
        self.update_clim(*self.image.get_clim())
   
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
        if not self.is_empty():
            data = self.image.get_array()
            if x <= (data.shape[1] - 1) and y <= (data.shape[0] - 1):
                self.ax.set_xlim((x - 1, x + 1))
                self.ax.set_ylim((y + 1, y - 1))
                self.draw_idle()


    def draw_discretemap(
        self,
        data: np.ndarray,
        encoder: dict[int, str], 
        colors: list[tuple[int, int, int]],
        title: str | None = None
    ) -> None:
        '''
        Update the canvas with a new discrete map.

        Parameters
        ----------
        data : numpy ndarray
            A 2D array storing the image pixels as numerical class IDs.
        encoder : dict[int, str]
            Dictionary that links the class IDs (key) with the corresponding
            class names (values).
        colors : list[tuple[int, int, int]]
            List of RGB triplets. The length of the list must match the length
            of the encoder dictionary.
        title : str or None, optional
            The image title. If None, the previous image title will be used.
            The default is None.

        '''
    # Call parent method to run generic update actions
        super().update_canvas()

    # Set image title
        if title is not None: 
            self.ax.set_title(title, pad=self._title_pad)

    # Set color map
        self.set_discretemap_cmap(colors)

    # Define boundary norm
        norm = self.set_boundary_norm(len(colors))

    # Set image
        if self.is_empty():
            img_kw = {'cmap': self.cmap, 'norm': norm, 'interpolation': 'none'}
            self.image = self.ax.imshow(data, **img_kw)
        else:
            self.image.set(data=data, cmap=self.cmap, norm=norm)

    # Adjust aspect ratio
        self.image.set_extent((-0.5,data.shape[1]-0.5, data.shape[0]-0.5,-0.5))

    # Set cursor data format (i.e., when hovering with mouse over pixels)
        self.image.format_cursor_data = lambda k: (
            f'Pixel Class = {"--" if np.ma.is_masked(k) else encoder[k]}'
        )

    # Hide colorbar
        if self.cbar is not None: 
            self.cax.set_visible(False)

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
    # Call the parent method to run generic update actions
        super().update_canvas()

    # Set image title
        if title is not None: 
            self.ax.set_title(title, pad=self._title_pad)

    # Set color map
        self.set_heatmap_cmap(cmap_name)

    # Set linear boundary norm
        norm = mpl_colors.Normalize()

    # Set image
        if self.is_empty():
            img_kw = {'cmap': self.cmap, 'norm': norm, 'interpolation': 'none'}
            self.image = self.ax.imshow(data, **img_kw)
        else:
            self.image.set(data=data, cmap=self.cmap, norm=norm)

    # # Update norm limits
    #     self.update_clim(data.min(), data.max())
   
    # Set and show colorbar
        if self.has_cbar:
            if self.cbar is None: self.set_cbar()
            self.cax.set_visible(True)

    # Adjust aspect ratio
        self.image.set_extent((-0.5,data.shape[1]-0.5, data.shape[0]-0.5,-0.5))

    # Set cursor data format (i.e., when hovering with mouse over pixels)
        self.image.format_cursor_data = lambda v: f"Pixel Value = {v}"

    # Redraw canvas
        self.draw_idle()


    def get_map(self, return_mask: bool = False) -> (
        np.ndarray | np.ma.MaskedArray | None
        | tuple[np.ndarray | np.ma.MaskedArray | None, np.ndarray | None]
    ):
        '''
        Get an unmasked version of the array displayed in the canvas and,
        optionally, its mask. If you just want the array "as is" use instead
        "self.image.get_array()".

        Parameters
        ----------
        return_mask : bool, optional
            Whether the mask of a masked map array should be returned. The
            default is False.

        Returns
        -------
        array : numpy ndarray or numpy MaskedArray or None
            The displayed map array or None if no map is displayed.
        mask : numpy ndarray or None, optional
            The mask array, if 'return_mask' is True and the array is masked.

        '''
        if self.is_empty(): return None

        array = self.image.get_array()

        if np.ma.is_masked(array):
            mask = array.mask
            array = array.data
        else:
            mask = None

        if return_mask:
            return array, mask
        else:
            return array



class BarCanvas(_CanvasBase):

    def __init__(
        self,
        orientation: str = 'v',
        size: tuple[float, float] = (6.4, 3.6),
        layout: str = 'tight',
        **kwargs
    ) -> None:
        '''
        A canvas class for plotting bar charts, including grouped bar charts
        with labels.

        Parameters
        ----------
        orientation : str, optional
            Orientation of the bars. Can be 'h' (horizontal) or 'v' (vertical).
            The default is 'v'.
        size : tuple[float, float], optional
            Size of the canvas. The default is (6.4, 3.6)
        layout : str, optional
            Layout engine of the figure. One of ['constrained', 'compressed', 
            'tight' and 'none']. The default is 'tight'.
        **kwargs
            Parent class arguments (see _CanvasBase class).

        '''
        super().__init__(size=size, layout=layout, **kwargs)

    # Set main attributes
        self.orient = orientation
        self.plot = None
        self.bar_width = 0.5
        self.decimal_precision = pref.get_setting('data/decimal_precision')
        self.label_amounts = []
        self.visible_amounts = False


    def is_empty(self) -> bool:
        '''
        Check if the canvas is empty.

        Returns
        -------
        bool
            Whether the canvas is empty.

        '''
        return self.plot is None


    def clear_canvas(self) -> None:
        '''
        Clear out the canvas and reset some properties.

        '''
    # Call the parent method to run generic cleaning actions
        super().clear_canvas(deep_clear=True)

    # Reset the references to the plotted data
        self.plot = None
        self.label_amounts.clear()
        self.draw_idle()


    def show_amounts(self, enabled: bool) -> None:
        '''
        Show/hide amounts on top of the bars.

        Parameters
        ----------
        enabled : bool
            Whether or not to display the amounts.

        '''
    # Update the attribute in any case
        self.visible_amounts = enabled

    # Exit function if the canvas is empty
        if self.is_empty():
            return

    # Show the amounts
        if enabled:
            pe = [mpl_patheffects.withStroke(linewidth=2, foreground='k')]
            kwargs = {
                'fmt': f'%.{self.decimal_precision}f',
                'padding': 16,
                'label_type': 'center',
                'color': style.IVORY,
                'path_effects': pe
            }
            self.label_amounts = self.ax.bar_label(self.plot, **kwargs)

    # Or hide the amounts
        else:
            for lbl in self.label_amounts:
                lbl.remove()
            self.label_amounts.clear()

    # Redraw the canvas
        self.draw_idle()


    def set_bar_width(self, width: float) -> None:
        '''
        Set the width of plotted bars.

        Parameters
        ----------
        width : float
            The required width.

        '''
        self.bar_width = width


    def set_decimal_precision(self, precision: int) -> None:
        '''
        Set the number of decimal places displayed in labels.

        Parameters
        ----------
        precision : int
            Number of decimal places.
            
        '''
        self.decimal_precision = precision


    def update_canvas(
        self,
        data: list[int | float] | list[list[int | float]], 
        tickslabels: list[str] | None = None,
        title: str | None = None,
        colors: list[tuple[int, int, int]] | None = None,
        multibars: bool = False, 
        labels: list[str] | None = None
    ) -> None:
        '''
        Update the canvas with a new plot.

        Parameters
        ----------
        data : list[int or float] or list[list[int or float]]
            X-axis values. If 'multibars' is True, this should be provided as a
            list of lists, where len(data) is the number of categories (or 
            groups). Sub-lists must share the same length.
        tickslabels : list[str] or None, optional
            X-ticks labels. The length of the list must match the length of 
            'data' (or of its sub-lists if 'multibars' is True). If None, no 
            tick label is shown. The default is None.
        title : str or None, optional
             The plot title. If None, no title is shown. The default is None.
        colors : list[tuple[int, int, int]] or None, optional
            A list of RGB triplets. The length of the list must match the 
            length of 'data' (or of its sub-lists if 'multibars' is True). If 
            None, Matplotlib's default colors are used. The default is None.
        multibars : bool, optional
            Whether to adapt the plot to display bars in categories or groups. 
            The default is False.
        labels : list[str] or None, optional
            A list of labels associated with each bar or each category if 
            'multibars' is True. If provided, they are displayed in a legend. 
            The length of the list must match the length of 'data' (or of its 
            sub-lists if 'multibars' is True). If None, no legend will be 
            displayed. The default is None.

        '''
    # Call the parent method to run generic update actions
        super().update_canvas()

    # Clear the canvas and exit the function if data is empty
        if not len(data): 
            self.clear_canvas()
            return

    # Remove the previous plot if there is one
        if not self.is_empty(): 
            self.ax.clear()

    # Set the grid on for y or x axis
        gridax = 'y' if self.orient == 'v' else 'x'
        self.ax.grid(True, axis=gridax, ls=':', lw=0.5, zorder=0)

    # Adjust the ticks and tickslabels properties
        if tickslabels is not None:
            ticks = np.arange(1, len(tickslabels)+1)
            if self.orient == 'v':
                self.ax.set_xticks(ticks, labels=tickslabels, rotation=-60)
                self.ax.tick_params(axis='x', which='both', length=0)
            else:
                self.ax.set_yticks(ticks, labels=tickslabels)
                self.ax.tick_params(axis='y', which='both', length=0)
                self.ax.invert_yaxis()

    # Set the title
        if title is not None: 
            self.ax.set_title(title, pad=self._title_pad)

    # Convert the bar colors to a matplotlib compatible format
        if colors is not None: 
            colors = rgb_to_float(colors)

    # Adjust the canvas for a multibar plot, if required (multibars is used as 
    # a boolean switch)
        shift = self.bar_width/len(data) * multibars
        shift_step = np.linspace(-shift, shift, len(data))
        n_iter = multibars * len(data)

    # Build the plot
        if not n_iter:
            data = (data, )
            n_iter = 1

        for i in range(n_iter):
            lbl = labels if labels is None else labels[i]
            args = (ticks + shift_step[i], data[i], self.bar_width)
            kw = {'color': colors, 'ec': style.IVORY, 'lw': 0.5, 'label': lbl}

            if self.orient == 'v':
                self.plot = self.ax.bar(*args, **kw)
            else:
                self.plot = self.ax.barh(*args, **kw)

    # Build the legend
        if labels is not None: 
            self.ax.legend() # this will probably need tweaks (legends stacking?)

    # Refresh label amounts if required
        self.show_amounts(self.visible_amounts)

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
        self.density = density
        self.log = logscale
        self.nbins = 50
        self.hist_data, self.roi_hist_data = None, None
        self.hist, self.roi_hist = None, None


    def is_empty(self) -> bool:
        '''
        Check if the canvas is empty.

        Returns
        -------
        bool
            Whether the canvas is empty.

        '''
        return self.hist is None


    def clear_canvas(self) -> None:
        '''
        Clear out the plot and reset some attributes.

        '''
    # Call the parent method to run generic cleaning actions
        super().clear_canvas(deep_clear=True)

        self.hist, self.roi_hist = None, None
        self.hist_data, self.roi_hist_data = None, None
        self.draw_idle()


    def toggle_logscale(self, toggled: bool) -> None:
        '''
        Toggle on/off logarithmic scale. The plot is refreshed automatically.

        Parameters
        ----------
        toggled : bool
            Toggle on/off.

        '''
        self.log = toggled
        self.refresh_view()


    def set_nbins(self, num: int) -> None:
        '''
        Set the number of histogram bins. The plot is refreshed automatically.

        Parameters
        ----------
        num : int
            Number of bins.

        '''
        self.nbins = num
        self.refresh_view()


    def refresh_view(self) -> None:
        '''
        Refresh (re-render) the plot.

        '''
        if not self.is_empty():
            data = self.hist_data
            roi_data = self.roi_hist_data
            title = self.ax.get_title()
            self.update_canvas(data, roi_data, title)


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
            The histogram data.
        roi_data : numpy ndarray or None, optional
            The histogram data of a portion of a map enclosed by a ROI. Such
            data is plotted over the main plot. The default is None.
        title : str or None, optional
            The title of the plot. If None, the previous title will be used. 
            The default is None.

        Example of roi_data
        -----------------------------
        data = MAP.image.get_array()
        r0, r1, c0, c1 = ROI.extents
        roi_data = data[r0:r1, c0:c1]

        '''
    # Call the parent method to run generic update actions
        super().update_canvas()

    # Clear the canvas if it is already populated
        if not self.is_empty():
            self.ax.clear()

    # Set the title
        if title is not None:
            self.ax.set_title(title, pad=self._title_pad)

    # Plot the histogram data
        self.hist_data = data.flatten()
        hist_kw = {'bins': self.nbins, 'density': self.density, 'log': self.log}
        self.hist = self.ax.hist(self.hist_data, **hist_kw)

    # Plot the ROI histogram data, if requested
        self.roi_hist_data = roi_data
        if roi_data is not None:
            self.roi_hist_data = roi_data.flatten()
            roi_hist_kw = {**hist_kw, 'fc': style.HIST_MASK}
            self.roi_hist = self.ax.hist(self.roi_hist_data, **roi_hist_kw)
    
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



class ConfMatCanvas(_CanvasBase):

    def __init__(
        self,
        title: str = 'Confusion Matrix',
        xlab: str = 'Predicted classes',
        ylab: str = 'True classes',
        cbar: bool = True,
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
        cbar : bool, optional
            Whether to include a colorbar in the canvas. The default is True.
        size : tuple[float, float], optional
            Size of the canvas. The default is (9, 9).
        **kwargs
            Parent class arguments (see _CanvasBase class).

        '''
    # Call the constructor of the parent class
        super().__init__(size=size, **kwargs)

    # Set main attributes
        self.matrix = None
        self.matplot = None
        self.show_cbar = cbar
        self.cbar = None
        self.cax = None

    # Set title and axis labels attributes and initialize the ax
        self.title = title
        self.xlab = xlab
        self.ylab = ylab
        self._init_ax()


    def _init_ax(self) -> None:
        '''
        Reset the ax to default state.

        '''
        self.ax.set_title(self.title, pad=self._title_pad)
        self.ax.set_xlabel(self.xlab)
        self.ax.set_ylabel(self.ylab)


    def set_cbar(self) -> None:
        '''
        Initialize the colorbar ax.

        '''
        divider = make_axes_locatable(self.ax)
        self.cax = divider.append_axes("right", size="5%", pad=0.1)
        self.cbar = self.fig.colorbar(self.matplot, cax=self.cax)
        self.cbar.ax.set_title('%')


    def _compute_percentage(self, matrix: np.ndarray) -> np.ndarray:
        '''
        Transform 'matrix' data to percentage, rounded to one decimal place.

        Parameters
        ----------
        matrix : numpy ndarray
            Original matrix data.

        Returns
        -------
        numpy ndarray
            Transformed matrix data.

        '''
        sums = matrix.sum(1).reshape(-1, 1)
        perc = (matrix / sums) * 100
        matrix = np.round(perc, 1)
        return matrix


    def annotate(self, as_percent: bool) -> None:
        '''
        Populate the canvas with the matrix values.

        Parameters
        ----------
        as_percent : bool
            Whether annotations should be displayed as percentages.

        '''
    # Get the matrix data. If the matrix plot is None, exit function
        if self.matplot:
            matrix = self.matrix
        else:
            return
        
    # Compute percentages if required
        if as_percent:
            matrix = self._compute_percentage(self.matrix)

    # Define a meshgrid from the matrix shape
        n_rows, n_cols = matrix.shape
        row_ind = np.arange(n_rows)
        col_ind = np.arange(n_cols)
        xx, yy = np.meshgrid(col_ind, row_ind)

    # Annotate each node with the corresponding value
        pe = [mpl_patheffects.withStroke(linewidth=2, foreground='k')]
        kw = {'va': 'center', 'ha': 'center', 'color': 'w', 'path_effects': pe}
        for t, x, y in zip(matrix.flatten(), xx.flatten(), yy.flatten()):
            self.ax.annotate(t, (x, y), **kw)


    def clear_canvas(self) -> None:
        '''
        Clear out the plot and reset some properties.

        '''
    # Call the parent method to run generic cleaning actions
        super().clear_canvas(deep_clear=True)

    # Hide colorbar
        if self.cbar is not None: 
            self.cax.set_visible(False)

    # Reset some properties
        self.matrix = None
        self.matplot = None
        self._init_ax()

    # Redraw the canvas
        self.draw_idle()


    def remove_annotations(self) -> None:
        '''
        Remove any previous annotation on the canvas.

        '''
        for child in self.ax.get_children():
            if isinstance(child, Annotation):
                child.remove()


    def set_ticks(self, labels: list[str] | tuple[str, ...], axis: str = 'both') -> None:
        '''
        Set the ticks and their labels for the confusion matrix (true/predicted 
        classes).

        Parameters
        ----------
        labels : list[str] or tuple[str, ...]
            List of labels.
        axis : str, optional
            Axis to be populated with labels. The available choices are 'x',
            'y' or 'both'. The default is 'both'.

        Raises
        ------
        AssertionError
            Raised if 'axis' argument is not one of ['x', 'y', 'both'].

        '''
        assert axis in ('x', 'y', 'both')
        ticks = np.arange(len(labels))

        if axis in ('x', 'both'):
            self.ax.set_xticks(
                ticks, labels=labels, fontsize='small', rotation=-60)
            self.ax.tick_params(labelbottom=True, labeltop=False)
        if axis in ('y', 'both'):
            self.ax.set_yticks(ticks, labels=labels, fontsize='small')

        self.draw_idle()


    def update_canvas(self, data: np.ndarray, percent_annotations: bool = True) -> None:
        '''
        Update the canvas with a new matrix.

        Parameters
        ----------
        data : numpy ndarray
            Confusion matrix array of squared shape.
        percent_annotations : bool, optional
            Whether annotations should be displayed as percentages. The default
            is True.

        '''
    # Call the parent method to run generic update actions
        super().update_canvas()

    # Always compute matrix data percentage for a nicer colormap
        self.matrix = data
        data = self._compute_percentage(data)

    # If the plot is empty, build the matrix and the colorbar (if required)
        if self.matplot is None:
            self.matplot = self.ax.matshow(
                data, cmap='cividis', interpolation='none')
            if self.show_cbar:
                if self.cbar is None: 
                    self.set_cbar()
                self.cax.set_visible(True)

    # If the matrix is not empty, refresh it
        else:
            self.matplot.set_data(data)
            self.matplot.set_clim(data.min(), data.max())
            self.remove_annotations()
            # Set extent to allow plotting a different shaped matrix
            extent = (-0.5, data.shape[1]-0.5, data.shape[0]-0.5, -0.5)
            self.matplot.set_extent(extent)

    # Add new annotations
        self.annotate(percent_annotations)

    # Redraw the canvas
        self.draw_idle()



class SilhouetteCanvas(_CanvasBase):

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

        self.y_btm_init = 15
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
        pal = dict(zip(palette.keys(), rgb_to_float(palette.values())))

    # Iterate through clusters (artists) and use their label as identifier
        for artist in self.ax.get_children():
            if isinstance(artist, PolyCollection):
                id_ = int(artist.get_label())
                artist.set(fc = pal[id_])

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
    # Call the parent method to run generic update actions
        super().update_canvas()

    # Reset the ax and set the new title
        self.ax.cla()
        self._init_ax()
        self.ax.set_title(title, pad=self._title_pad)

    # Format the RGB palette colors as matplotlib compatible floating values
        pal = dict(zip(palette.keys(), rgb_to_float(palette.values())))

    # Adjust the initial vertical (y) padding
        y_btm = self.y_btm_init

    # Plot silhouettes, using y padding as a reference
        for clust_id, values in sil_values.items():
            y_top = y_btm + len(values)
            self.ax.fill_betweenx(
                y = np.arange(y_btm, y_top),
                x1 = 0,
                x2 = values,
                lw = 0.3,
                fc = pal[clust_id],
                ec = style.IVORY, 
                label = str(clust_id)
            )
            y_btm = y_top + self.y_btm_init

    # Draw the average silhouette score as a red vertical dashed line
        pe = [mpl_patheffects.withStroke(foreground=style.IVORY)]
        self.ax.axvline(x=sil_avg, color='r', ls='--', lw=2, path_effects=pe)

    # Render the plot
        self.draw_idle()


    def clear_canvas(self) -> None:
        '''
        Clear out the canvas.

        '''
    # Call the parent method to run generic cleaning actions
        super().clear_canvas(deep_clear=True)

    # Initialize ax
        self._init_ax()
        
    # Redraw the canvas
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
        self.perc_fmt = perc_fmt


    def _init_ax(self) -> None:
        '''
        Reset the ax to default state.

        '''
        super().clear_canvas(deep_clear=True)

    # Legend must be cleared to avoid stacking of multiple legend boxes
        self.fig.legends.clear()
        
    # Equal aspect ratio ensures that the pie is drawn as a circle
        self.ax.axis('equal') 


    def update_canvas(
        self,
        data: np.ndarray,
        labels: list[str],
        title: str = '', 
        legend: bool = True
    ) -> None:
        '''
        Draw a new pie chart.

        Parameters
        ----------
        data : numpy ndarray
            Values that populate the pie.
        labels : list[str]
            A sequence of strings providing the labels for each wedge.
        title : str, optional
            Plot title. The default is ''.
        legend : bool, optional
            Whether to include a legend. The default is True.

        '''
    # Intialize the canvas and the ax
        super().update_canvas()
        self._init_ax()
       
    # Set title
        if title:
            self.ax.set_title(title, pad=self._title_pad)
        
    # Adjust wedges and labels layout parameters
        expl = [0.1] * len(data) if self._3d else None
        lbldist = 1.1 if self.perc_fmt else 0.4

    # Draw pie chart
        self.ax.pie(
            x = data,
            explode = expl,
            autopct = self.perc_fmt,
            shadow = self._3d,
            labels = labels,
            labeldistance = None if legend else lbldist
        )
    
    # Draw legend
        if legend:
            self.fig.subplots_adjust(left=0.2, bottom=0.2)
            hand, lbls = self.ax.get_legend_handles_labels()
            self.fig.legend(hand, lbls, loc='lower left', fontsize='small',
                            bbox_to_anchor=(0, 0))
        
    # Render the plot
        self.draw_idle()


    def clear_canvas(self) -> None:
        '''
        Clear out the canvas.

        '''
        self._init_ax()
        self.draw_idle()



class CurveCanvas(_CanvasBase):

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
        self.title = title
        self.xlab = xlab
        self.ylab = ylab
        self.grid_on = grid

    # Initialize the ax
        self._init_ax()


    def _init_ax(self) -> None:
        '''
        Reset the ax to default state.

        '''
        self.ax.set_title(self.title, pad=self._title_pad)
        self.ax.set_xlabel(self.xlab)
        self.ax.set_ylabel(self.ylab)
        self.ax.grid(self.grid_on, ls=':', lw=0.5, zorder=0)


    def add_curve(
        self,
        xdata: np.ndarray | float | int,
        ydata: np.ndarray | float | int,
        label: str = '',
        color: Iterable[float] | float | str | None = None
    ) -> None:
        '''
        Add a new curve. Warning: drawing is not performed by this method.

        Parameters
        ----------
        xdata : numpy ndarray or float or int
            X coordinate(s) of the point(s) of the curve.
        ydata : numpy ndarray or float or int
            Y coordinate(s) of the point(s) of the curve.
        label : str, optional
            Curve name. The default is ''.
        color : Iterable[float] or float or str or None, optional
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
        curves: list[tuple[float, float]],
        labels: list[str], 
        colors: list[tuple[float, float, float] | float | str] | None = None
    ) -> None:
        '''
        Update existent curves in the plot. If curve is not existent, add it as
        a new curve. Curve existence is determined via its label.

        Parameters
        ----------
        curves : list[tuple[float, float]]
            List of (x, y) cooords for each curve.
        labels : list[str]
            A sequence of unique strings providing the label for each curve.
        colors : list[tuple[float, float, float] or float or str] or None, optional
            List of valid Matplotlib colors for each curve. If None or empty,
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
        super().update_canvas()

    # Check for unique labels
        if len(set(labels)) < len(labels):
            raise ValueError('"Labels" must be unique names.')
        
    # Set colors to a list of None if not specified
        if not colors:
            colors = [None] * len(curves)

    # Check that all parameters have same length
        if not len(curves) == len(labels) == len(colors):
            raise ValueError('"Data", "labels" and "colors" must have same length.')

    # Determine existent curves
        existent_curves = self.ax.lines
        existent_labels = [ec.get_label() for ec in existent_curves]

    # Add new curves, update existent ones.
        for cur, lbl, col in zip(curves, labels, colors):
            if lbl in existent_labels:
                existent_curves[existent_labels.index(lbl)].set_data(cur)
            else:
                self.add_curve(*cur, label=lbl, color=col)

    # Rescale the plot
        self.ax.relim()
        self.ax.autoscale_view()

    # Render the legend in the most suitable position
        self.ax.legend(
            loc='best', frameon=True, facecolor=style.IVORY, framealpha=1)
    
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
        super().clear_canvas(deep_clear=True)
        self._init_ax()
        self.draw_idle()



class NavTbar(backend_qtagg.NavigationToolbar2QT):

    def __init__(
        self,
        canvas: _CanvasBase,
        parent: QWidget | None = None, 
        orient: Qt.Orientation = Qt.Horizontal,
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
        super().__init__(canvas, parent, coordinates=coords)

    # Set the main attributes of the class
        self.canvas = canvas
        self.orient = orient

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
        self.setOrientation(self.orient)

    # Avoid coords and data label to be cut off
        if coords:
            self.locLabel.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Minimum)

    # Set the toolbar style-sheet
        self.setStyleSheet(style.SS_TOOLBAR)

    
    @classmethod
    def imageCanvasDefault(
        cls,
        canvas: ImageCanvas,
        parent: QWidget | None = None,
        orient: Qt.Orientation = Qt.Horizontal
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
        orient: Qt.Orientation = Qt.Horizontal
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
        orient: Qt.Orientation = Qt.Horizontal
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
        self.lbl_action.toggled.connect(self.canvas.show_amounts)
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
        self.log_action.setChecked(self.canvas.log)
        self.log_action.toggled.connect(self.canvas.toggle_logscale)
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
        if new_h > old_h and self.orient == Qt.Horizontal:
            self.setMinimumHeight(new_h)



class RoiPatch(Rectangle):

    def __init__(
        self,
        bbox: tuple[float, float, int, int],
        color: tuple[float, float, float] | float | str,
        filled: bool
    ) -> None:
        '''
        Reimplementation of a Matplotlib rectangle patch, useful to manage and 
        display ROIs.

        Parameters
        ----------
        bbox : tuple[float, float, int, int]
            ROI bounding box (x0, y0, width, height).
        color : tuple[float, float, float] or float or str
            A valid Matplotlib color.
        filled : bool
            Whether the ROI patch should be filled.

        '''
        x0, y0, w, h = bbox
        super().__init__((x0, y0), w, h, linewidth=2, color=color, fill=filled)



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
        

    def set_text(self, text: str) -> None:
        '''
        Change the text annotation.

        Parameters
        ----------
        text : str
            New text.

        '''
    # Hyphen hides ROI annotation
        super().set_text('' if text == '-' else text)



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
            canvas = None, # deprecated (see 'MultiCursor' class for details)
            axes = axes,
            useblit = True,
            horizOn  =True,
            vertOn = True,
            color = style.BLACK_PEARL,
            lw = 1
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
        cursor = Qt.PointingHandCursor if self.active else Qt.ArrowCursor
        self.canvas.setCursor(QCursor(cursor))



class RectSel(mpl_widgets.RectangleSelector):

    def __init__(
        self,
        ax: Axes,
        onselect: Callable,
        interactive: bool = True,
        btns: list[int] | None = [1]
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
            default is [1].

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
        fmt: str = 'matrix',
        mode: str = 'full'
    ) -> tuple[int, int, int, int] | None:
        '''
        Get integer extents of the selector after checking if it lies within
        the provided 'shape'. Different output extents format can be selected.

        Parameters
        ----------
        shape : tuple[int, int]
            Control shape. It must be provided as (rows, cols).
        fmt : str, optional
            The output format of coords. If 'matrix' the coords format is
            (row_min, row_max, col_min, col_max). If 'xy' the coords format is
            (x_min, x_max, y_min, y_max). The default is 'matrix'.
        mode : str, optional
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
                raise ValueError(f'No mode available for {mode}')

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
                raise ValueError('No format available for {fmt}')

        return extents


    def fixed_rect_bbox(
        self,
        shape: tuple[int, int],
        mode: str = 'full'
    ) -> tuple[float, float, int, int] | None:
        '''
        Get the bounding box of the selector after checking if it lies within
        the provided shape.

        Parameters
        ----------
        shape : tuple[int, int]
            Control shape. It must be provided as (rows, cols).
        mode : str, optional
            How to treat pixels at the border of the selector region. At the
            moment the only supported mode is 'full', which means that only
            the pixels that are entirely covered by the selector region will be
            included. The default is 'full'.

        Returns
        -------
        bbox : tuple[float, float, int, int] or None
            Fixed bounding box (x0, y0, width, height). If the selector region
            falls entirely outside the map area, it will be None.

        Raises
        ------
        ValueError
            Raised if 'mode' parameter is not 'full'.  

        '''
    # Get the default bounding box
        x0, y0, w, h = self._rect_bbox

    # Exit function if the bounding box is completely outside the map
        if (
            x0 + 0.5 > shape[1]
            or x0 + w < -0.5
            or y0 + 0.5 > shape[0]
            or y0 + h < -0.5
        ):
            return None

    # Fix extents that are partially outside the map borders
        if x0 < -0.5:
            w += x0 + 0.5
            x0 = -0.5
        if x0 + w + 0.5 > shape[1]:
            w = shape[1] - 0.5 - x0
        if y0 < -0.5:
            h += y0 + 0.5
            y0 = -0.5
        if y0 + h + 0.5 > shape[0]:
            h = shape[0] - 0.5 - y0

    # Exclude pixels not entirely selected --> 'mode' : 'full'
        match mode:
            case 'full':
                _x0 = round(x0 + 1) - 0.5
                _y0 = round(y0 + 1) - 0.5
                w = int(w + x0 - _x0)
                h = int(h + y0 - _y0)
            case _:
                raise ValueError(f'No mode available for {mode}')
    
    # Bbox is invalid if it has non-positive height or width (safety)
        if w < 1 or h < 1:
            return None

    # Return the fixed bounding box
        bbox = (_x0, _y0, w, h)
        return bbox


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
        cursor = Qt.PointingHandCursor if self.active else Qt.ArrowCursor
        self.canvas.setCursor(QCursor(cursor))



class HeatmapScaler(mpl_widgets.SpanSelector):

    def __init__(
        self,
        ax: Axes,
        onselect: Callable,
        interactive: bool = True,
        btns: list[int] | None = [1]
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
            which means all the buttons. The default is [1].

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
    ) -> list[tuple[float, float, float]] | tuple[float, float, float]:
    '''
    Convert RGB values to Matplotlib compatible floating values, ranging in
    [0, 1].

    Parameters
    ----------
    rgb_list : Iterable[tuple[int, int, int]]
        A list of RGB triplets.

    Returns
    -------
    list[tuple[float, float, float]] or tuple[float, float, float]
        A list of float RGB triplets or a single float RGB triplet if only one
        color is passed.

    '''
    float_rgb = [(r/255, g/255, b/255) for (r, g, b) in rgb_list]
    if len(float_rgb) == 1:
        return float_rgb[0]
    else:
        return float_rgb