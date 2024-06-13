"""This file is just a temporary space where new features can be tested"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn.cluster
import sklearn.metrics
from scipy import ndimage
import os, sys
from _base import *

from PyQt5.QtWidgets import QApplication, QMainWindow, QGroupBox, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import QPropertyAnimation, QRect

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setGeometry(100, 100, 400, 300)
        self.setWindowTitle("QPropertyAnimation QGroupBox Example")

        # Creazione di un layout principale e aggiunta alla finestra principale
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Creazione di un QGroupBox
        self.group_box = QGroupBox("Group Box", self)
        self.group_box.setCheckable(True)
        self.group_box.setChecked(True)
        self.layout.addWidget(self.group_box)

        # Creazione di un layout per il QGroupBox
        self.group_box_layout = QVBoxLayout(self.group_box)

        # Aggiungi widget al QGroupBox (esempio)
        self.group_box_layout.addWidget(QPushButton("Button 1", self.group_box))
        self.group_box_layout.addWidget(QPushButton("Button 2", self.group_box))
        self.group_box_layout.addWidget(QPushButton("Button 3", self.group_box))

        # Creazione di un pulsante per mostrare/nascondere il QGroupBox
        self.toggle_button = QPushButton("Toggle Group Box", self)
        self.layout.addWidget(self.toggle_button)

        # Creazione dell'animazione per l'altezza massima
        self.animation = QPropertyAnimation(self.group_box, b"geometry")
        self.animation.setDuration(500)  # Durata in millisecondi (0.5 secondi)

        # Connessione del pulsante per avviare l'animazione
        self.toggle_button.clicked.connect(self.toggle_group_box)

    def toggle_group_box(self):
        start_rect = self.group_box.geometry()

        if self.group_box.isChecked():
            end_rect = QRect(start_rect.x(), start_rect.y(), start_rect.width(), 0)
        else:
            # Calcola l'altezza massima basata sul sizeHint del QGroupBox
            end_height = self.group_box.sizeHint().height()
            end_rect = QRect(start_rect.x(), start_rect.y(), start_rect.width(), end_height)

        self.animation.setStartValue(start_rect)
        self.animation.setEndValue(end_rect)
        self.animation.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



### ------------------ A U T O   R O I   D E T E C T O R ------------------ ###
###                                    |                                    ### 
###                                   \ /                                   ###
###                                    v                                    ###

# cwd = r'Examples\Samples\S32_b\maps'
# paths = filter(lambda f: f.endswith('.gz'), os.listdir(cwd))
# arrays = [np.loadtxt(os.path.join(cwd, p), dtype='int32') for p in paths]

# # Compute npv for each map
# NPVS = []
# for arr in arrays:
#     # Median filter (probably not required)
#     med = ndimage.median_filter(arr, 1)

#     # Neighborhood pixel variance (NPV)
#     r = 9
#     c_idx = r**2//2
#     npv = ndimage.generic_filter(med, lambda a: np.sum(np.abs(a-a[c_idx])), r) # use sum or median or mean

#     # NPV rescaling
#     npv = npv/(arr.max() - arr.min())

#     # Exclude borders
#     bord = r//2
#     mask = np.ones(npv.shape)
#     mask[bord:-bord, bord:-bord] = 0
#     npv[mask==1] = np.nan

#     # Append to list
#     NPVS.append(npv)

# # Get the cumulative NPV array, rescaled to [0, 1]
# npv_sum = sum(NPVS)
# npv_sum = npv_sum/np.nanmax(npv_sum)






# # Highlight best ROIs
# distance = 30 # rois minimum inter-distance
# d = bord + distance
# n_roi = 25
# rois_found = 0
# roimap = np.zeros(npv_sum.shape)
# idx = np.argsort(npv_sum, axis=None) 

# for i in idx:
#     row, col = np.unravel_index(i, npv_sum.shape)
#     if np.isnan(npv_sum[row, col]):
#         continue
#     y0, y1, x0, x1 = row-d, row+d, col-d, col+d
#     if y0 < 0: y0 = 0
#     if y1 > npv_sum.shape[0]: y1 = npv_sum.shape[0]
#     if x0 < 0: x0 = 0
#     if x1 > npv_sum.shape[1]: x1 = npv_sum.shape[1] 
    
#     if np.all(roimap[y0:y1, x0:x1] != 1): # if not overlaps
#         roimap[row-bord:row+bord+1, col-bord:col+bord+1] = 1
#         rois_found += 1
#         if rois_found == n_roi:
#             print(npv_sum[row, col])
#             break





# mmap = MineralMap.load(r'Examples\Samples\S32_b\class\MineralMap0.mmp')
# er = ndimage.binary_erosion(roimap, np.ones((3, 3)))
# minmap = mmap.minmap_encoded
# minmap[(roimap - er)==1] = len(mmap.encoder)



# fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True)

# col = [(r/255, g/255, b/255) for (r,g,b) in mmap.palette.values()]
# roi_cmap = ListedColormap(col + [(1., 0., 0.)] , name='roi_cmap')

# ax0.set_title('Cumulative Neighborhood Pixel Variance map')
# ax0.imshow(npv_sum, cmap='Spectral_r', interpolation='none')
# ax1.set_title('Auto-detected training areas\n(displayed on top of mineral map)')
# ax1.imshow(minmap, cmap=roi_cmap, interpolation='none')

# plt.show()




# for x in range(0):
#     # Highlight best ROIs
#     distance = 15 # rois minimum inter-distance
#     d = bord + distance
#     n_roi = 70
#     rois_found = 0
#     idx = np.argsort(npv_sum, axis=None) 

#     for i in idx:
#         row, col = np.unravel_index(i, npv_sum.shape)
#         if npv_sum[row, col] == np.nan:
#             continue
#         y0, y1, x0, x1 = row-d, row+d, col-d, col+d
#         if y0 < 0: y0 = 0
#         if y1 > npv_sum.shape[0]: y1 = npv_sum.shape[0]
#         if x0 < 0: x0 = 0
#         if x1 > npv_sum.shape[1]: x1 = npv_sum.shape[1] 
        
#         if np.all(roimap[y0:y1, x0:x1] != 1): # if not overlaps
#             roimap[row-bord:row+bord, col-bord:col+bord] = 1
#             rois_found += 1
#             if rois_found == n_roi:
#                 print(npv_sum[row, col])
#                 break


#     er = ndimage.binary_erosion(roimap, np.ones((3, 3)))
#     minmap = mmap.minmap_encoded
#     minmap[(roimap - er)==1] = len(mmap.encoder)



#     fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True)

#     col = [(r/255, g/255, b/255) for (r,g,b) in mmap.palette.values()]
#     roi_cmap = ListedColormap(col + [(1., 0., 0.)] , name='roi_cmap')

#     ax0.set_title('Cumulative Neighborhood Pixel Variance map')
#     ax0.imshow(npv_sum, cmap='Spectral_r', interpolation='none')
#     ax1.set_title('Auto-detected training areas\n(displayed on top of mineral map)')
#     ax1.imshow(minmap, cmap=roi_cmap, interpolation='none')

    # plt.show()

###                                    ^                                    ### 
###                                   / \                                   ###
###                                    |                                    ###        
### ------------------ A U T O   R O I   D E T E C T O R ------------------ ### 



### ------------------- S I L H O U E T T E  S C O R E -------------------- ###
###                                    |                                    ### 
###                                   \ /                                   ###
###                                    v                                    ###

# cwd = r'Examples\Samples\S32_b\maps'
# paths = filter(lambda f: f.endswith('.gz'), os.listdir(cwd))
# inmaps = [InputMap.load(os.path.join(cwd, p)) for p in paths]
# feat = InputMapStack(inmaps).get_feature_array()


# seed = 3003
# kmeans = sklearn.cluster.KMeans(7, random_state=seed)
# print('Clustering')
# pred = kmeans.fit_predict(feat)

# print('Silhouette score')
# size = int(0.05 * pred.size)
# subset_idx = np.random.default_rng(seed).permutation(pred.size)[:size]
# feat = feat[subset_idx, :]
# pred = pred[subset_idx]

# sil_score = sklearn.metrics.silhouette_score(feat, pred)
# print(sil_score)
# sil_sample = sklearn.metrics.silhouette_samples(feat, pred)
# print(sil_sample[np.argmin(sil_sample)])

###                                    ^                                    ### 
###                                   / \                                   ###
###                                    |                                    ###        
### ------------------- S I L H O U E T T E  S C O R E -------------------- ### 

