# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 09:14:45 2023

@author: albdag
"""

from PyQt5.QtGui import QCursor, QMouseEvent
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtWidgets import QAction, QSizePolicy, QWidgetAction

import preferences as pref
import customObjects as cObj

import numpy as np
import os
from weakref import proxy

import matplotlib as MPL
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_interactions import panhandler


# MPL.use('Qt5Agg')





class PanHandler(panhandler):
    '''Reimplemantation of the <panhandler> class from mpl_interactions library. It simply adds
    a couple of functions to control the type of cursor'''

    _pressCursor = QCursor(Qt.ClosedHandCursor)
    _releaseCursor = QCursor(Qt.ArrowCursor)

    def __init__(self, fig, button=2):
        super(PanHandler, self).__init__(fig, button)
        self.wheelButton = button

    def press(self, event):
        if event.button == self.wheelButton:
            self._releaseCursor = self.fig.canvas.cursor()
            self.fig.canvas.setCursor(self._pressCursor)
            super(PanHandler, self).press(event)

    def release(self, event):
        if event.button == self.wheelButton:
            self.fig.canvas.setCursor(self._releaseCursor)
            super(PanHandler, self).release(event)




class NavTbar(NavigationToolbar2QT):
    '''A class to provide a navigation toolbar linked to a canvas object.'''
    instances = []

    def __init__(self, canvas, QtParent=None, orient=Qt.Horizontal, coords=True):
        '''
        Constructor of the class NavTbar.

        Parameters
        ----------
        canvas : _CanvasBase object or one of its children classes.
            The canvas object linked to the navigation toolbar.
        QtParent : QWidget or None, optional
            Parent widget of the toolbar in the GUI. The default is None.
        orient : Qt orientation, optional
            The orientation of the navigation toolbar. The default is
            Qt.Horizontal.
        coords : bool, optional
            Display the coordinates in the navigation toolbar. The default is
            True.

        Returns
        -------
        None.

        '''
    # Weakly track all class instances
        self.__class__.instances.append(proxy(self))

    # Call the constructor of the parent class
        super(NavTbar, self).__init__(canvas, QtParent, coordinates=coords)

    # Set the main attributes of the class
        self.canvas = canvas
        self.orient = orient

    # Set the icons size
        size = pref.get_setting('plots/NTBsize', 20, type=int)
        self.setIconSize(QSize(size, size))

    # Set the orientation of the toolbar (vertical/horizontal)
        self.setOrientation(self.orient)

    # Avoid coords and data label to be cut off
        if coords:
            self.locLabel.setSizePolicy(QSizePolicy.Expanding,
                                        QSizePolicy.Minimum)

    # Set the toolbar style-sheet
        self.setStyleSheet(pref.SS_toolbar)


    def fixHomeAction(self):
        '''
        Function to fix the default zoom action of a canvas. Canvas must
        implement the reset_view function.

        Returns
        -------
        None.

        '''
        # Check that canvas has implemented the function reset_view
        reset_view = getattr(self.canvas, "reset_view", None)
        if callable(reset_view):
            home = self.findChildren(QAction)[2]
            home.triggered.connect(reset_view)


    def getTrueActions(self):
        '''
        Returns a list of the true actions (= NOT QWidgetActions) held by the
        toolbar.

        Returns
        -------
        actions : list
            True actions.

        '''
        actions = self.actions()
        actions = [a for a in actions if not isinstance(a, QWidgetAction)]
        return actions


    def removeToolByIndex(self, list_of_indices):
        '''
        Remove actions from the navigation toolbar by their indices.

        Parameters
        ----------
        list_of_indices : list
            List of indices of the actions that should be excluded.

        Returns
        -------
        None.

        '''
        for i in list_of_indices:
            self.removeAction(self.findChildren(QAction)[i])


    def resizeEvent(self, event):
        '''
        (EXPERIMENTAL) # !!!
        Reimplementation of the resizeEvent function, that updates the minimum
        height of the toolbar.

        Parameters
        ----------
        event : QResizeEvent
            The resize event.

        Returns
        -------
        None.

        '''
        new_h = event.size().height()
        old_h = event.oldSize().height()
        if new_h > old_h and self.orient == Qt.Horizontal:
            self.setMinimumHeight(new_h)









class _CanvasBase(FigureCanvasQTAgg):
    '''A base class for any type of plot.'''

    def __init__(self, size=(6.4, 4.8), layout='none', 
                 wheelZoomEnabled=True, wheelPanEnabled=True):
        '''
        _CanvasBase class constructor.

        Parameters
        ----------
        size : tuple, optional
            Width and height (in inches) of canvas. The default is (6.4, 4.8).
        layout : str, optional
            Layout engine of the figure. One of ['constrained', 'compressed', 
            'tight' and 'none']. The default is 'none'.
        wheelZoomEnabled : bool, optional
            Wether or not allow zooming with mouse wheel. The default is True.
        wheelPanEnabled : TYPE, optional
            Wether or not allow drag-pan with mouse buttons. The default is True.

        Returns
        -------
        None.

        '''


    # Define the figure and the ax of the matplotlib canvas
        self.fig = MPL.figure.Figure(figsize=size, facecolor=pref.IVORY,
                                     edgecolor='#19232D', linewidth=2,
                                     layout=layout)
        self.ax = self.fig.add_subplot(111, facecolor=pref.IVORY)
        self.ax.axis('off')

    # Call the constructor of the parent class
        super(_CanvasBase, self).__init__(self.fig)

    # Set the default style
        MPL.style.use('seaborn-v0_8-colorblind') # preferences to change style

    # Set events connections for mouse-wheel zoom and mouse-wheel pan, if enabled
        if wheelZoomEnabled:
            self._id_zoom = self.wheelZoom_handler()
            self.setMouseTracking(True)
        if wheelPanEnabled:
            self._id_pan = PanHandler(self.fig)

    # Enable custom context menu request (when right-clicking on canvas)
        self.setContextMenuPolicy(Qt.CustomContextMenu)


    def clear_canvas(self, deep_clear=False):
        '''
        Generic canvas clearing actions. Must be reimplemented in each subclass.

        Parameters
        ----------
        deep_clear : bool, optional
            Fully reset the ax. The default is False.

        Returns
        -------
        None.

        '''
    # Fully reset the ax if requested:
        if deep_clear: self.ax.clear()

    # Hide the axis (borders and ticks)
        self.ax.axis('off')


    def mouseMoveEvent(self, event):
        '''
        Set focus on the canvas widget when mouse hovering. Necessary to detect the
        key attribute of the mouse wheel zoom event. See wheel_zoom() function for
        further details.

        Parameters
        ----------
        event : Matplotlib.backend_bases.MouseEvent
            The Matplotlib mouse event (version 3.5.1).

        Returns
        -------
        None.

        '''
        super(_CanvasBase, self).mouseMoveEvent(event)
        if self.hasMouseTracking():
            self.setFocus()


    def get_navigation_context_menu(self, navTbar):
        '''
        Return a default context menu with actions extracted from the provided
        navigation toolbar.

        Parameters
        ----------
        navTbar : NavTbar
            The navigation toolbar associated with this canvas.

        Returns
        -------
        menu
            A StyledMenu() populated with actions.

        '''

        menu = cObj.StyledMenu()

    # Get Navigation Toolbar true actions (exclude QWidgetActions, like coords)
        ntbar_actions = navTbar.getTrueActions()

    # Get the hidden 'Show toolbar' action
        hide_ntbar = navTbar.findChildren(QAction)[1]
        hide_ntbar.setText('Show navigation toolbar')

    # Add actions to menu
        menu.addAction(hide_ntbar)
        menu.addSeparator()
        menu.addActions(ntbar_actions)

        return menu


    def RGB_to_float(self, RGB_list):
        '''
        Convert RGB values to matplotlib-compatible floating values ranging in [0, 1].

        Parameters
        ----------
        RGB_list : list
            A list of RGB triplets.

        Returns
        -------
        list
            A list of float RGB triplets.

        '''
        floatRGB = [(r/255, g/255, b/255) for (r,g,b) in RGB_list]
        if len(floatRGB) == 1:
            return floatRGB[0]
        else:
            return floatRGB


    def update_canvas(self):
        '''
        Generic canvas update actions. Must be reimplemented in each subclass.

        Returns
        -------
        None.

        '''
    # Show the axis (borders and ticks)
        self.ax.axis('on')

        # try:
        #     # disconnect the zoom factory
        #     self._id_zoom()
        #     # reconnect the zoom factory
        #     self._id_zoom = self.wheelZoom_handler()
        # except AttributeError:
        #     # Wheelzoom is not enabled
        #     pass


    # Inspired by https://gist.github.com/tacaswell/3144287
    # and https://github.com/mpl-extensions
    def wheelZoom_handler(self, base_scale = 1.5):
        '''
        Wheel zoom factory.

        Parameters
        ----------
        base_scale : float, optional
            Base zoom scale. The default is 1.5.

        Returns
        -------
        None.

        '''

        def zoom(event):

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

        # Push the current view limits and position onto their respective stacks
            self.toolbar.push_current() # ???

        # Set new limits
            self.ax.set_xlim([xdata - (xdata-cur_xlim[0]) / scale_factor,
                              xdata + (cur_xlim[1]-xdata) / scale_factor])
            self.ax.set_ylim([ydata - (ydata-cur_ylim[0]) / scale_factor,
                              ydata + (cur_ylim[1]-ydata) / scale_factor])



        # Redraw the canvas
            self.draw_idle()
            # self.flush_events() gets more framy

    # Attach the call back
        cid = self.mpl_connect("scroll_event", zoom)

        def disconnect_zoom():
            self.mpl_disconnect(cid)

    # Return the disconnect function
        return disconnect_zoom









class ImageCanvas(_CanvasBase):
    '''A base class for any type of canvas displaying images and maps'''

    def __init__(self, binary=False, cbar=True, size=(10, 7.5), **kwargs):
        '''
        ImageCanvas class constructor.

        Parameters
        ----------
        binary : bool, optional
            Format the canvas to plot binary data. The default is False.
        cbar : bool, optional
            Include a colorbar in the canvas. The default is True.
        size : tuple, optional
            Width and height (in inches) of the canvas. The default is (10, 7.5).
        **kwargs
            Parent class arguments (see _CanvasBase constructor).

        Returns
        -------
        None.

        '''
    # Call the constructor of the parent class
        super(ImageCanvas, self).__init__(size, **kwargs)

    # Set the pixel coordinate format
        # self.ax.format_coord = lambda x, y: f"X : {round(x)}, Y : {round(y)}  |  " \
        #                                     f"R : {round(y)}, C : {round(x)}"
        self.ax.format_coord = lambda x, y: f'X : {round(x)}, Y : {round(y)}'

    # Define the main attributes of the canvas
        self.isBinary = binary
        self.image = None
        self.scaled_clim = None

    # Initialize the colormap
        self.cmap = None

    # Initialize the colorbar
        self.has_cbar = cbar
        self.cbar, self.cax = None, None

    def is_empty(self):
        return self.image is None


    def set_cbar(self):
        '''
        Set the colorbar ax.

        Returns
        -------
        None.

        '''
        divider = make_axes_locatable(self.ax)
        self.cax = divider.append_axes("right", size="5%", pad=0.1)
        self.cbar = self.fig.colorbar(self.image, cax=self.cax, extend='both')


    def set_heatmap_cmap(self, MPLcmap='Spectral_r'):
    # Set colormap
        self.cmap = MPL.colormaps['binary_r' if self.isBinary else MPLcmap]
    # Set outliers as grayed-out
        self.cmap.set_over('0.2')
        self.cmap.set_under('0.5')
    # Set masked data as ivory
        self.cmap.set_bad(pref.IVORY)


    def set_discretemap_cmap(self, colorlist, name='Custom_cmap'):
    # Convert colorlist RGB triplets to float RGB values
        floatRGBA = self.RGB_to_float(colorlist)
    # # Filter the colorlist if the map data is masked (required as bug fix)
    #     if self.contains_masked_data():
    #         data, mask = self.get_map(return_mask=True)
    #         floatRGBA = [floatRGBA[u] for u in np.unique((data[~mask]))]
    # Set colormap
        self.cmap = MPL.colors.ListedColormap(floatRGBA, name=name)
    # Set outliers as ivory
        self.cmap.set_over(pref.IVORY)
        self.cmap.set_under(pref.IVORY)
    # Set masked data as ivory
        self.cmap.set_bad(pref.IVORY)

    
    def set_boundary_norm(self, n_classes):
        norm = MPL.colors.BoundaryNorm(range(n_classes+1), n_classes)
        return norm


    def alter_cmap(self, col_arg):
        if self.is_empty(): return

        if (t := type(col_arg)) == str:
            self.set_heatmap_cmap(col_arg)
        elif hasattr(t, '__iter__'):
            self.set_discretemap_cmap(col_arg)
        else:
            return

        self.image.set_cmap(self.cmap)
        self.draw_idle()
        # self.flush_events()


    def contains_masked_data(self):
        if self.is_empty():
            return False
        else:
            return np.ma.is_masked(self.image.get_array())


    def contains_discretemap(self):
        return isinstance(self.cmap, MPL.colors.ListedColormap)


    def contains_heatmap(self):
        return isinstance(self.cmap, MPL.colors.LinearSegmentedColormap)


    def scale_clim(self, toggled, array_list=None):
        '''
        Toggle scaled norm limits.

        Parameters
        ----------
        toggled : bool
            Enable/disable scaled norm limits.
        array_list : list, optional
            List of NumPy ndarrays containg the data from which to extract the
            scaled norm limits. It must be a non-ampty list if <toggled> is True.
            The default is None.

        Returns
        -------
        None.

        '''
    # Enable scaled norm limits
        if toggled:
            if not array_list:
                raise ValueError('<array_list> must be a non-empty list if <toggled> is True')
            glb_min = min([arr.min() for arr in array_list])
            glb_max = max([arr.max() for arr in array_list])
            self.scaled_clim = (glb_min, glb_max)

    # Disable scaled norm limits
        else: self.scaled_clim = None

    # Update the norm limits in any case
        self.update_clim()


    def update_clim(self, vmin=None, vmax=None):
        '''
        Update the image norm limits.

        Parameters
        ----------
        vmin : int or float or None, optional
            Lower limit. The default is None.
        vmax : int or float or None, optional
            Upper limit. The default is None.

        Returns
        -------
        None.

        '''
        if not self.is_empty():

        # We use the vmin, vmax args if provided
            if vmin is not None and vmax is not None:
                self.image.set_clim(vmin, vmax)

        # If vmin, vmax are None, we use the scaled clims if they are present
            elif self.scaled_clim is not None:
                self.image.set_clim(*self.scaled_clim)

        # Otherwise we use the current image clims (reset clims)
            else:
                data = self.image.get_array()
                self.image.set_clim(data.min(), data.max())
            # Re-apply boundary norm if image is a discrete map
                if self.contains_discretemap():
                    norm = self.set_boundary_norm(len(np.unique(data)))
                    self.image.set_norm(norm)


    def clear_canvas(self):
        '''
        Clear out certain elements of the canvas and resets their properties.

        Returns
        -------
        None.

        '''

        if not self.is_empty():
            # shape = self.image.get_array().shape
        # We set the data as an array of NaNs for compatibility with the colormaps
        # behavior, where NaN data is colored in white (see build_cmap func of DiscreteClassCanvas sub-class)
            # self.image.set_data(np.empty(shape) * np.nan)

        # Call the parent function to run generic cleaning actions
            super(ImageCanvas, self).clear_canvas()

        # ??? Reset the image and its colormap
            self.image.remove()
            self.image, self.cmap = None, None

        # Reset the colorbar and its ax
            self.cbar = None
            if self.cax is not None:
                self.cax.set_visible(False)

        # Remove the title
            self.ax.set_title('')

            self.draw_idle()
            # self.flush_events()


    def enable_picking(self, enabled):
        '''
        Enable data picking from this canvas.

        Parameters
        ----------
        enabled : bool
            Enable/disable data picking.

        Returns
        -------
        None.

        '''
        if not self.is_empty():
            self.image.set_picker(True if enabled else None)
            # self.draw_idle()
            # self.flush_events()


    def reset_view(self):
        '''
        Show the original map view. Fixes issues when clicking home button in
        the Navigation Toolbar.

        Returns
        -------
        None.

        '''
        if not self.is_empty():
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

            self.draw_idle()
            # self.flush_events()


    def zoom_to(self, x, y):
        '''
        Zoom to a specific pixel in the map.

        Parameters
        ----------
        x : int
            The x coordinate of the pixel.
        y : int
            The y coordinate of the pixel.

        Returns
        -------
        None.

        '''
        if not self.is_empty():
            data = self.image.get_array()
            if x <= (data.shape[1] - 1) and y <= (data.shape[0] - 1):
                self.ax.set_xlim((x - 1, x + 1))
                self.ax.set_ylim((y + 1, y - 1))
                self.draw_idle()
                # self.flush_events()



    def draw_discretemap(self, data, encoder, colors, title=None):
        '''
        Update the canvas with a new classified image.

        Parameters
        ----------
        data : NumPy ndarray
            A 2D array storing the image pixels as nmerical class IDs.
        encoder : dict
            Dictionary that links the class IDs (key) with the corresponding
            class names (values).
        colors : list
            List of RGB tuples. The length of the list must match the length of
            the encoder dictionary.
        title : str, optional
            The image title. If None, the previous image title will be used.
            The default is None.

        Returns
        -------
        None.

        '''

    # Call parent function to run generic update actions
        super(ImageCanvas, self).update_canvas()

    # Set image title
        if title is not None: self.ax.set_title(title)

    # Set color map
        self.set_discretemap_cmap(colors)

    # Define boundary norm
        norm = self.set_boundary_norm(len(colors))

    # Set image
        if self.is_empty():
            self.image = self.ax.imshow(data, cmap=self.cmap, norm=norm,
                                        interpolation='none')
        else:
            self.image.set(data=data, cmap=self.cmap, norm=norm)

    # # Set image
    #     if self.image is None:
    #         self.image = self.ax.imshow(data, interpolation='none',
    #                                     vmin=data.min(), vmax=data.max())
    #     else:
    #         self.image.set(data=data, clim=(data.min(), data.max()))

    # # Set color map. We must set it after the data is set in order to solve a
    # # rendering bug with cmaps that happens while discrete data is masked
    #     self.set_discretemap_cmap(colors)
    #     self.image.set_cmap(self.cmap)

    # Adjust aspect ratio
        self.image.set_extent((-0.5,data.shape[1]-0.5, data.shape[0]-0.5,-0.5))

    # Set cursor data format (i.e., when hovering with mouse over pixels)
        self.image.format_cursor_data = lambda k: "Pixel Class = "\
            f"{'--' if np.ma.is_masked(k) else encoder[k]}"

    # Hide colorbar
        if self.cbar is not None: self.cax.set_visible(False)

    # Redraw canvas
        self.draw_idle()
        self.flush_events()
        # # <self.flush_events()> da problemi al rectSel del phase refiner
        # # (rettangolo che si sposta dopo l'acquisizione dei suoi vertici)
        # if not self.img.pickable():
        #     self.flush_events()


    def draw_heatmap(self, data, title=None, MPLcmap='Spectral_r'):
        '''
        Update the canvas with a new heatmap.

        Parameters
        ----------
        data : NumPy ndarray
            A 2D array storing the image pixel values.
        title : str, optional
            The image title. If None, the previous image title will be used. The default is None.
        MPLcmap : str
            A Matplotlib colormap string. The default is "Spectral_r".

        Returns
        -------
        None.

        '''
    # Call the parent function to run generic update actions
        super(ImageCanvas, self).update_canvas()

    # Set image title
        if title is not None: self.ax.set_title(title)

    # Set color map
        self.set_heatmap_cmap(MPLcmap)

    # Set linear boundary norm
        norm = MPL.colors.Normalize()

    # Set image
        if self.is_empty():
            self.image = self.ax.imshow(data, cmap=self.cmap, norm=norm,
                                        interpolation='none')
        else:
            self.image.set(data=data, cmap=self.cmap, norm=norm)

    # Set and show colorbar
        if self.has_cbar:
            if self.cbar is None : self.set_cbar()
            self.cax.set_visible(True)

    # ??? Update norm limits. Is this really useful?
        # self.update_clim()

    # Adjust aspect ratio
        self.image.set_extent((-0.5,data.shape[1]-0.5, data.shape[0]-0.5,-0.5))

    # Set cursor data format (i.e., when hovering with mouse over pixels)
        self.image.format_cursor_data = lambda v: f"Pixel Value = {v}"

    # Redraw canvas
        self.draw_idle()
        self.flush_events()

        # if not self.img.pickable():
        #     self.flush_events()

    def get_map(self, return_mask=False):
        '''
        Get an unmasked version of the array displayed in the canvas and,
        optionally, its mask. If you just want the array "as is" use instead
        self.image.get_array().

        Parameters
        ----------
        return_mask : bool, optional
            Whether the mask of a masked map array should be returned. The
            default is False.

        Returns
        -------
        array : numpy ndarray or numpy MaskedArray or None
            The displayed map array or None if no map is displayed.
        mask : numpy ndarray, optional
            The mask, if <return_mask> is True and the array is masked.

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




# class _ImageCanvas(_CanvasBase):
#     '''A base class for any type of canvas displaying images and maps'''

#     def __init__(self, size=(10, 7.5), **kwargs):
#         '''
#         _ImageCanvas class constructor.

#         Parameters
#         ----------
#         size : tuple, optional
#             Width and height (in inches) of the canvas. The default is (10, 7.5).
#         **kwargs
#             Parent class arguments (see _CanvasBase constructor).

#         Returns
#         -------
#         None.

#         '''
#     # Call the constructor of the parent class
#         super(_ImageCanvas, self).__init__(size, **kwargs)

#     # Set the pixel coordinate format
#         self.ax.format_coord = lambda x, y: f"X : {round(x)}, Y : {round(y)}  |  " \
#                                             f"R : {round(y)}, C : {round(x)}"

#     # Define the main attributes of the class
#         self.image = None
#         self.lock_zoom = False


#     def clear_canvas(self):
#         '''
#         Clear out the canvas. The canvas is populated with an empty array with
#         the shape of the previous displayed array.

#         Returns
#         -------
#         None.

#         '''
#         if self.image is not None:
#             shape = self.image.get_array().shape
#         # We set the data as an array of NaNs for compatibility with the colormaps
#         # behavior, where NaN data is colored in white (see build_cmap func of DiscreteClassCanvas sub-class)
#             self.image.set_data(np.empty(shape) * np.nan)
#             self.ax.set_title('')
#             self.draw_idle()


#     def enable_picking(self, enabled):
#         '''
#         Enable data picking from this canvas.

#         Parameters
#         ----------
#         enabled : bool
#             Enable/disable data picking.

#         Returns
#         -------
#         None.

#         '''
#         if self.image is not None:
#             self.image.set_picker(True if enabled else None)
#             self.draw_idle()


#     def reset_zoom(self):
#         '''
#         Zoom to the original map view. Fixes issues when clicking home button in the Navigation Toolbar

#         Returns
#         -------
#         None.

#         '''
#         if self.image is not None:
#             data = self.image.get_array()
#             self.ax.set_xlim((-0.5, data.shape[1]-0.5))
#             self.ax.set_ylim((data.shape[0]-0.5, -0.5))
#             self.draw_idle()


#     def toggle_zoomLock(self, signal):
#         '''
#         Lock the zoom when changing the displayed data.

#         Parameters
#         ----------
#         signal : bool
#             Enable/disable the zoom lock.

#         Returns
#         -------
#         None.

#         '''
#         self.lock_zoom = signal


#     def zoom_to(self, x, y):
#         '''
#         Zoom to a specific pixel in the map.

#         Parameters
#         ----------
#         x : int
#             The x coordinate of the pixel.
#         y : int
#             The y coordinate of the pixel.

#         Returns
#         -------
#         None.

#         '''
#         if self.image is not None:
#             data = self.image.get_array()
#             if x <= (data.shape[1] - 1) and y <= (data.shape[0] - 1):
#                 self.ax.set_xlim((x - 1, x + 1))
#                 self.ax.set_ylim((y + 1, y - 1))
#                 self.draw_idle()
#                 # self.flush_events()



# class DiscreteImageCanvas(_ImageCanvas):
#     '''A canvas object for displaying image data with discrete pixel values (=classes)'''

#     def __init__(self, **kwargs):
#         '''
#         Constructor of the DiscreteImageCanvas class.

#         Parameters
#         ----------
#         **kwargs
#             Parent class arguments (see _ImageCanvas constructor).

#         Returns
#         -------
#         None.

#         '''
#     # Call the parent class constructor
#         super(DiscreteImageCanvas, self).__init__(**kwargs)

#     # Initialize the colormap
#         self.cmap = None


#     def alter_cmap(self, colors):
#         '''
#         A function to change the colormap.

#         Parameters
#         ----------
#         colors : list
#             List of RGB tuples.

#         Returns
#         -------
#         None.

#         '''
#         if self.image is not None:
#             self.cmap = self.build_cmap(colors)
#             self.image.set_cmap(self.cmap)
#             self.draw_idle()
#             # self.flush_events()


#     def build_cmap(self, colorList, name='Custom_cmap'):
#         '''
#         Generate a matplotlib-compatible colormap.

#         Parameters
#         ----------
#         colorList : list
#             List of RGB tuples.
#         name : str, optional
#             A name to identify the colormap. Not useful at the moment. The default is 'Custom_cmap'.

#         Returns
#         -------
#         cmap : matplotlib.colors.Colormap
#             The colormap.

#         '''
#         floatRGBA = self.RGB_to_float(colorList)
#         cmap = MPL.colors.ListedColormap(floatRGBA, name=name)
#     # NaN data will be colored in white regardless of the colormap colors
#         cmap.set_bad('w')
#         return cmap


#     def export_array(self, filepath, encoder=None):
#         '''
#         A funtion to export the image data array in text format.

#         Parameters
#         ----------
#         filepath : str or PATH-like object
#             The filepath in which to store the array.
#         encoder : dict or None, optional
#             Wether or not to also store a text file containing a data decoder.
#             If None, the file is not generated. The default is None.

#         Returns
#         -------
#         None.

#         '''
#         if self.image is not None:
#             data = self.image.get_array()
#             np.savetxt(filepath, data, fmt='%d')
#             if encoder is not None:
#                 encoder_path = os.path.splitext(filepath)[0] + '_transDict.txt'
#                 with open(encoder_path, 'w') as ep:
#                     for ID, lbl in encoder:
#                         ep.write(f'{ID} :\t{lbl}\n')


#     def update_canvas(self, data, encoder, colors, title=None):
#         '''
#         Update the canvas with a new classified image.

#         Parameters
#         ----------
#         data : NumPy ndarray
#             A 2D array storing the image pixels as nmerical class IDs.
#         encoder : dictu
#             Dictionary that links the class IDs (key) with the corresponding class names (values).
#         colors : list
#             List of RGB tuples. The length of the list must match the length of the encoder dictionary.
#         title : str, optional
#             The image title. If None, the previous image title will be used. The default is None.

#         Returns
#         -------
#         None.

#         '''

#     # Call the parent function to update the zoom factory
#         super(DiscreteImageCanvas, self).update_canvas()

#     # Set image title
#         if title is not None: self.ax.set_title(title)

#     # Build the color map
#         self.cmap = self.build_cmap(colors)

#     # Show Image
#         if self.image is None:
#             self.image = self.ax.imshow(data, cmap=self.cmap, interpolation='none',
#                                         vmin=data.min(), vmax=data.max())
#         else:
#             self.image.set(data=data, cmap=self.cmap, clim=(data.min(), data.max()))

#         # To manage map extent when arrays with different shape are loaded
#             if not self.lock_zoom:
#                 self.image.set_extent((-0.5, data.shape[1]-0.5, data.shape[0]-0.5, -0.5))

#     # Set the cursor data format (i.e., when hovering with mouse over pixels)
#         self.image.format_cursor_data = lambda k: f"Pixel Class = {encoder[k]}"

#     # Redraw the canvas
#         self.draw_idle()
#         # # <self.flush_events()> da problemi al rectSel del phase refiner
#         # # (rettangolo che si sposta dopo l'acquisizione dei suoi vertici)
#         # if not self.img.pickable():
#         #     self.flush_events()



# class HeatmapCanvas(_ImageCanvas):
#     '''A canvas object for displaying image data with continuos or binary pixel values'''

#     def __init__(self, binary=False, cbar=True, **kwargs):
#         '''
#         Constructor of the HeatmapCanvas class.

#         Parameters
#         ----------
#         binary : bool, optional
#             Format the canvas to plot binary data. The default is False.
#         cbar : bool, optional
#             Include a colorbar in the canvas. The default is True.
#         **kwargs
#             Parent class arguments (see _ImageCanvas constructor).

#         Returns
#         -------
#         None.

#         '''
#     # Call the parent class contructor
#         super(HeatmapCanvas, self).__init__(**kwargs)

#     # Define the heatmap's main attributes
#         self.has_cbar = cbar
#         self.cbar, self.cax = None, None
#         self.scaled_clim = None

#     # Set a colormap and set its outliers as "Grayed-out"
#         self.cmap = MPL.colormaps['binary_r' if binary else 'Spectral_r']
#         self.cmap.set_over('0.2')
#         self.cmap.set_under('0.5')


#     def _init_cbar(self):
#         '''
#         Initialize the colorbar ax.

#         Returns
#         -------
#         None.

#         '''
#         divider = make_axes_locatable(self.ax)
#         self.cax = divider.append_axes("right", size="5%", pad=0.1)
#         self.cbar = self.fig.colorbar(self.image, cax=self.cax, extend='both')


#     def get_global_clim(self, array_list):
#         '''
#         Extract the global min-max pixel values (norm limits) from a list of arrays.

#         Parameters
#         ----------
#         array_list : list
#             A list of NumPy ndarrays.

#         Returns
#         -------
#         glb_min : int or float
#             Global minimum value.
#         glb_max : int or float
#             Global maximum value.

#         '''
#         glb_min = min([arr.min() for arr in array_list])
#         glb_max = max([arr.max() for arr in array_list])
#         return (glb_min, glb_max)


#     def scale_clim(self, toggled, maps_list=None):
#         '''
#         Toggle scaled norm limits.

#         Parameters
#         ----------
#         toggled : bool
#             Enable/disable scaled norm limits.
#         maps_list : list, optional
#             List of NumPy ndarrays containg the data from which to extract the
#             scaled norm limits. It must be a non-ampty list if <toggled> is True.
#             The default is None.

#         Returns
#         -------
#         None.

#         '''
#     # Enable scaled norm limits
#         if toggled:
#             if not maps_list:
#                 raise ValueError('<maps_list> must be a non-empty list if <toggled> is True')
#             self.scaled_clim = self.get_global_clim(maps_list)

#     # Disable scaled norm limits
#         else: self.scaled_clim = None

#     # Update the norm limits in any case
#         self.update_clim()


#     def update_clim(self):
#         '''
#         Update the image norm limits.

#         Returns
#         -------
#         None.

#         '''
#         if self.image is not None:

#         # We use the scaled clims if they are present
#             if self.scaled_clim is not None:
#                 self.image.set_clim(*self.scaled_clim)

#         # Else we use the current image clims
#             else:
#                 data = self.image.get_array()
#                 self.image.set_clim(data.min(), data.max())


#     def update_canvas(self, data, title=None):
#         '''
#         Update the canvas with a new image.

#         Parameters
#         ----------
#         data : NumPy ndarray
#             A 2D array storing the image pixel values.
#         title : str, optional
#             The image title. If None, the previous image title will be used. The default is None.

#         Returns
#         -------
#         None.

#         '''
#     # Call the parent function to update the zoom factory
#         super(HeatmapCanvas, self).update_canvas()

#     # Set image title
#         if title is not None: self.ax.set_title(title)

#     # Show the image
#         if self.image is None:
#             self.image = self.ax.imshow(data, self.cmap, interpolation='none')
#             self.image.format_cursor_data = lambda v: f"Pixel Value = {v}"
#             if self.has_cbar: self._init_cbar()

#         else:
#             self.image.set_data(data)
#             self.update_clim()
#             self.ax.autoscale(enable = not self.lock_zoom)

#         # Manage map extent when arrays with different shape are loaded
#             if not self.lock_zoom:
#                 self.image.set_extent((-0.5, data.shape[1]-0.5, data.shape[0]-0.5, -0.5))
#                 self.ax.relim()

#     # Redraw the canvas
#         self.draw_idle()

#         # if not self.img.pickable():
#         #     self.flush_events()






class BarCanvas(_CanvasBase):
    '''A canvas object for plotting bar charts, including grouped bar charts with labels.'''

    def __init__(self, orientation='v', size=(6.4, 3.6), **kwargs):
        '''
        Constructor of the BarCanvas class.

        Parameters
        ----------
        orientation : 'v' or 'h', optional
            Orientation of the bars. The default is 'v' (vertical).

        size : tuple, optional
            Size of the canvas. The default is (6.4, 3.6)

        **kwargs
            Parent class arguments (see _CanvasBase constructor).

        Returns
        -------
        None.

        '''
    # Call the constructor of the parent class
        super(BarCanvas, self).__init__(**kwargs)

    # Define the main attributes of the class
        self.orient = orientation
        self.plot = None
        self.barWidth = 0.5
        self.label_amounts = []
        self.visibleAmounts = False


    def clear_canvas(self):
        '''
        Clear out the plot and reset some properties.

        Returns
        -------
        None.

        '''
    # Call the parent function to run generic cleaning actions
        super(BarCanvas, self).clear_canvas(deep_clear=True)

    # Reset the references to the plotted data
        self.plot = None
        self.label_amounts.clear() # it is a list object
        self.draw_idle()
        # self.flush_events()


    def show_amounts(self, enabled):
        '''
        Show/hide amounts on top of the bars.

        Parameters
        ----------
        enabled : bool
            Wether or not to display the amounts.

        Returns
        -------
        None.

        '''
        self.visibleAmounts = enabled

        if self.plot is not None:

        # Show the amounts
            if enabled:
                prec = pref.get_setting('plots/legendDec', 3, type=int)
                pe = [MPL.patheffects.withStroke(linewidth=2, foreground='k')]
                self.label_amounts = self.ax.bar_label(self.plot,
                                                       fmt=f'%.{prec}f',
                                                       label_type='center',
                                                       padding=16, color='w',
                                                       path_effects=pe)

        # Remove the amounts
            else:
                for lbl in self.label_amounts:
                    lbl.remove()
                self.label_amounts.clear()

        # Redraw the canvas
            self.draw_idle()


    def set_barWidth(self, width):
        '''
        Set the width of plotted bars.

        Parameters
        ----------
        width : float
            The required width.

        Returns
        -------
        None.

        '''
        self.barWidth = width


    def update_canvas(self, data, tickslabels=None, title=None, colors=None,
                      multibars=False, labels=None):
        '''
        Update the canvas with a new plot.

        Parameters
        ----------
        data : list
            X-axis values. If <multibars> is True, this should be provided as a list of lists,
            where len(data) is the number of categories (or groups). Sub-lists must share the same
            length.
        tickslabels : list, optional
            X-ticks labels. The length of the list must match the length of <data> (or of its
            sub-lists if <multibars> is True). If None, no tick label is shown. The default is None.
        title : str, optional
             The plot title. If None, no title will be shown. The default is None.
        colors : list, optional
            A list of RGB tuples. The length of the list must match the length of <data> (or of its
            sub-lists if <multibars> is True). If None, default matplotlib colors are used.
            The default is None.
        multibars : bool, optional
            Format the plot to display bars in categories or groups. The default is False.
        labels : list, optional
            A list of labels associated with each bar or each category if <multibars> is True.
            If provided, they will be displayed in a legend. The length of the list must match
            the length of <data> (or of its sub-lists if <multibars> is True). If None, no legend
            will be displayed. The default is None.

        Returns
        -------
        None.

        '''
    # Call the parent function to run generic update actions
        super(BarCanvas, self).update_canvas()

    # Clear the canvas and exit the function if data is empty
        if not len(data): return self.clear_canvas()

    # Remove the previous plot if there is one
        if self.plot is not None: self.ax.clear()

    # Set the grid on for y or x axis
        gridax = 'y' if self.orient == 'v' else 'x'
        self.ax.grid(True, axis=gridax)

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
        if title is not None: self.ax.set_title(title)

    # Convert the bar colors to a matplotlib compatible format
        if colors is not None: colors = self.RGB_to_float(colors)

    # Adjust the canvas for a multibar plot, if required (<multibars> is a boolean switch)
        shift = self.barWidth/len(data) * multibars
        shift_step = np.linspace(-shift, shift, len(data))
        n_iter = multibars * len(data)

    # Show the plot
        if not n_iter:
            data = (data,)
            n_iter = 1
        for i in range(n_iter):
            if self.orient == 'v':
                self.plot = self.ax.bar(ticks+shift_step[i], data[i],
                                        self.barWidth, color=colors,
                                        edgecolor='k', label=labels if labels is None else labels[i])
            else:
                self.plot = self.ax.barh(ticks+shift_step[i], data[i],
                                         self.barWidth, color=colors,
                                         edgecolor='k', label=labels if labels is None else labels[i])

    # Show the legend
        if labels is not None: self.ax.legend()

    # Refresh label amounts if required
        self.show_amounts(self.visibleAmounts)

    # Redraw the canvas
        self.draw_idle()
        self.flush_events()



class HistogramCanvas(_CanvasBase):
    '''A canvas object for plotting histograms.'''

    def __init__(self, density=False, logscale=False, size=(3, 3), **kwargs):
        '''
        Constructor of the HistogramCanvas class.

        Parameters
        ----------
        density : bool, optional
            Scale y axes to [0, 1]. The default is False.
        logscale : bool, optional
            Use logarithmic scale. The default is False.

        **kwargs
            Parent class arguments (see _CanvasBase constructor).

        Returns
        -------
        None.

        '''
    # Call the constructor of the parent class
        super(HistogramCanvas, self).__init__(**kwargs)

    # Set yaxis ticks properties
        self.ax.yaxis.set_ticks_position('both')

    # Define the main attributes of the class
        self.density = density
        self.log = logscale
        self.nbins = 50
        self.histData, self.ROI_histData = None, None
        self.hist, self.ROI_hist = None, None


    def is_empty(self):
        return self.hist is None


    def clear_canvas(self):
        '''
        Clear out the plot and reset some attributes.

        Returns
        -------
        None.
        '''
    # Call the parent function to run generic cleaning actions
        super(HistogramCanvas, self).clear_canvas(deep_clear=True)

        self.hist, self.ROI_hist = None, None
        self.histData, self.ROI_histData = None, None
        self.draw_idle()
        # self.flush_events()


    def toggle_logscale(self, toggled):
        self.log = toggled
        self.refresh_view()


    def set_nbins(self, num):
        '''
        Set the number of histogram bins. The plot is refreshed automatically.

        Parameters
        ----------
        num : int
            Number of bins.

        Returns
        -------
        None.

        '''
        self.nbins = num
        self.refresh_view()



    def refresh_view(self):
        if not self.is_empty():
            data = self.histData
            ROI_data = self.ROI_histData
            title = self.ax.get_title()
            self.update_canvas(data, ROI_data, title)


    def update_canvas(self, data, ROI_data=None, title=None):
        '''
        Update the canvas with a new plot.

        Parameters
        ----------
        data : NumPy ndarray
            The histogram data.
        ROI_data : NumPy ndarray, optional
            The histogram of a ROI, plotted over the main plot. The default is
            None.
        title : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        Example of ROI_data
        -----------------------------
        data = MAP.image.get_array()
        r0,r1, c0,c1 = ROI.extents
        ROI_data = data[r0:r1, c0:c1]
        '''
    # Call the parent function to run generic update actions
        super(HistogramCanvas, self).update_canvas()

    # Clear the canvas if it is already populated
        if not self.is_empty():
            self.ax.clear()

    # Set the title
        if title is not None: self.ax.set_title(title)

    # Plot the histogram data
        self.histData = data.flatten()
        self.hist = self.ax.hist(self.histData, bins=self.nbins,
                                 density=self.density, log=self.log)

    # Plot the ROI histogram data, if requested
        self.ROI_histData = ROI_data
        if ROI_data is not None:
            self.ROI_histData = ROI_data.flatten()
            self.ROI_hist = self.ax.hist(self.ROI_histData, bins=self.nbins,
                                         density=self.density,
                                         log=self.log, fc=pref.HIST_MASK)
    # Adjust the x and y lims
        self.update_xylim()

    # Redraw the canvas
        self.draw_idle()
        # self.flush_events()


    def reset_view(self):
        '''
        Show the original histogram view. Fixes issues when clicking home
        button in the Navigation Toolbar.

        Returns
        -------
        None.

        '''
        if not self.is_empty():
        # Update the xy limits
            self.update_xylim()
        # Fix image zoom issues when pressing home button multiple times
            if self.fig.get_tight_layout():
                self.fig.tight_layout()
        # Redraw canvas
            self.draw_idle()
            # self.flush_events()

    def update_xylim(self):
        self.ax.relim()
        self.ax.autoscale()








class ConfMatCanvas(_CanvasBase):
    '''Canvas object specific for plotting confusion matrices'''
    def __init__(self, size=(9, 9), cbar=True,
                 title='', xlab='', ylab='', **kwargs):
        '''
        Constructor of the ConfMatCanvas class.

        Parameters
        ----------
        size : tuple, optional
            Width and height (in inches) of the canvas. The default is (9, 9).
        cbar : bool, optional
            Include a colorbar in the canvas. The default is True.
        title : str, optional
            The title of the plot. The default is ''.
        xlab : str, optional
            Name of the x-axis. The default is ''.
        ylab : str, optional
            Name of the y-axis. The default is ''.
        **kwargs
            Parent class arguments (see _CanvasBase constructor).

        Returns
        -------
        None.

        '''
    # Call the constructor of the parent class
        super(ConfMatCanvas, self).__init__(size, **kwargs)

    # Define main attributes
        self.mtx = None
        self.showCbar = cbar
        self.cbar = None
        self.cax = None

    # Set title and axis labels attributes and initialize the ax
        self.title = title
        self.xlab = xlab
        self.ylab = ylab
        self._init_ax()


    def _init_ax(self):
        '''
        Populate the ax with the title and the labels.

        Returns
        -------
        None.

        '''
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlab)
        self.ax.set_ylabel(self.ylab)


    def _init_cbar(self):
        '''
        Initialize the colorbar ax.

        Returns
        -------
        None.

        '''
        divider = make_axes_locatable(self.ax)
        self.cax = divider.append_axes("right", size="5%", pad=0.1)
        self.cbar = self.fig.colorbar(self.mtx, cax=self.cax)


    def annotate(self, data):
        '''
        Populate the canvas with the matrix values.

        Parameters
        ----------
        data : NumPy ndarray
            Confusion matrix array of shape (n, n).

        Returns
        -------
        None.

        '''
    # Define a meshgrid from the matrix shape
        n_rows, n_cols = data.shape
        row_ind = np.arange(n_rows)
        col_ind = np.arange(n_cols)
        x_coord, y_coord = np.meshgrid(col_ind, row_ind)

    # Annotate each node with the corresponding value
        for txt, x, y in zip(data.flatten(), x_coord.flatten(), y_coord.flatten()):
            self.ax.annotate(txt, (x, y), va='center', ha='center', color='w',
                             path_effects=[MPL.patheffects.withStroke(linewidth=2, foreground='k')])


    def clear_canvas(self):
        '''
        Clear out the plot and reset some properties.

        Returns
        -------
        None.

        '''
    # Call the parent function to run generic cleaning actions
        super(ConfMatCanvas, self).clear_canvas(deep_clear=True)

        self.mtx = None
        self._init_ax()
        if self.cax is not None: self.cax.clear()
        self.draw_idle()
        #self.flush_events()


    def remove_annotations(self):
        '''
        Remove any previous annotation on the canvas.

        Returns
        -------
        None.

        '''
        for child in self.ax.get_children():
            if isinstance(child, MPL.text.Annotation):
                child.remove()


    def set_ticks(self, labels, axis='both'):
        '''
        Set the ticks and their labels for the confusion matrix (true/predicted classes).

        Parameters
        ----------
        labels : list
            List of labels strings.
        axis : str, optional
            Axis to be populated with labels. The available choices are 'x',
            'y' or 'both'. The default is 'both'.

        Returns
        -------
        None.

        '''
        assert axis in ('x', 'y', 'both')
        ticks = np.arange(len(labels))

        if axis in ('x', 'both'):
            self.ax.set_xticks(ticks, labels=labels, fontsize='x-small', rotation=-60)
            self.ax.tick_params(labelbottom=True, labeltop=False)
        if axis in ('y', 'both'):
            self.ax.set_yticks(ticks, labels=labels, fontsize='x-small')

        self.draw_idle()
        #self.flush_events()


    def update_canvas(self, data):
        '''
        Update the canvas with a new matrix.

        Parameters
        ----------
        data : NumPy ndarray
            Confusion matrix array of shape (n, n).

        Returns
        -------
        None.

        '''
    # Call the parent function to run generic update actions
        super(ConfMatCanvas, self).update_canvas()

    # If the matrix is empty, build the matrix and the colorbar (if required)
        if self.mtx is None:
            self.mtx = self.ax.matshow(data, cmap='inferno', interpolation='none')
            if self.showCbar: self._init_cbar()

    # If the matrix is not empty, refresh it
        else:
            self.mtx.set_data(data)
            self.mtx.set_clim(data.min(), data.max())
        # Set extent to allow plotting a different shaped matrix without calling clear_canvas() before
            self.mtx.set_extent((-0.5, data.shape[1]-0.5, data.shape[0]-0.5, -0.5))
            self.remove_annotations()

    # Add new annotations
        self.annotate(data)
    # Redraw the canvas
        self.draw_idle()
        self.flush_events()




















