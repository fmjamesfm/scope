#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application to display and save microscope USB camera images with the option to overlay a calibrated grid. 

@author: fmjamesfm@gmail.com

"""

import sys
import cv2

from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtWidgets import QFileDialog
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QPoint, QThread, pyqtSlot
from PyQt6.QtGui import QPainter, QImage, QTextCursor, QPixmap


from pyqtgraph.widgets.RawImageWidget import RawImageWidget

import numpy as np
import os
import logging


basepath, _ = os.path.split(__file__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#%%
#import matplotlib.pyplot as plt
IMG_SIZE    = 2592, 1944 # 640,480 or 1280,720 or 1920,1080
SCALE = 2

SCALED_SIZE = int(IMG_SIZE[0] / SCALE), int(IMG_SIZE[1] / SCALE)

IMG_FORMAT  = QImage.Format.Format_RGB888


CAMERA = 1

def get_mask(file, size=IMG_SIZE):
    cal = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    mask = np.zeros(cal.shape[:2], dtype="uint8")
    mask[cal[:,:,3] > 0] = 255
    calmask = cv2.resize(mask, size)
    _, calmask = cv2.threshold(calmask, 50, 255, cv2.THRESH_BINARY)
    return calmask 

OVERLAY_FILES = ('./images/cal4x.png', './images/cal10x.png',  './images/cal40x.png',  './images/cal100x.png')
OVERLAYS = [get_mask(file) for file in OVERLAY_FILES]

def list_cams():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


class Camera:
    def __init__(self, camera):
        self.camera = camera
        self.vc = None

    def openCamera(self):
        self.vc = cv2.VideoCapture(self.camera)
        # self.vc.set(5, 30)  #set FPS
        self.vc.set(3, IMG_SIZE[0])  # set width
        self.vc.set(4, IMG_SIZE[1])  # set height

        if not self.vc.isOpened():
            print('failure')
            msgBox = QtWidgets.QMessageBox()
            msgBox.setText("Failed to open camera.")
            msgBox.exec_()
            return
        
    def read(self):
        return self.vc.read()
        
    
    def close(self):
        if not(self.vc is None):
            self.vc.release()

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, cam, *args, **kw):
        super().__init__(*args, **kw)
        self.running = True
        self.cam = cam
        
    def run(self):
        # capture from web cam
        self.cam.openCamera()
        
        while self.running:
            ret, cv_img = self.cam.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        self.cam.close()
        
    def stop(self):
        self.running=False

class ImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
 
    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()
 
    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()


class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        uic.loadUi(os.path.join(basepath, 'ui/main_win.ui'), self)
        
        self.setWindowTitle("Camera")
        self.display_width = IMG_SIZE[0]
        self.display_height = IMG_SIZE[1]
        
        self.camView = ImageWidget(self)
        self.gridLayout.addWidget(self.camView)
        # create the video capture thread
        self.cam = Camera(CAMERA)
        self.thread = VideoThread(self.cam)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()
        self.overlay_alpha = 0.9
        self.currentImage = None
        self.freeze = False
    
        self.objectiveLensChanged(self.lensSlider.value())
        
    def overlayOpacityChanged(self, val):
        self.overlay_alpha = val/100.0
        
    def changeResolution(self):
        self.cam.set(3, IMG_SIZE[0])  # set width
        self.cam.set(4, IMG_SIZE[1])  # set height
    
    def objectiveLensChanged(self, val):
        self.overlay_mask = OVERLAYS[val]
        
    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    def takeSnapshot(self):
        
        self.freeze = not(self.freeze)
        
        if self.freeze:
            self.freezeButton.setText('Unfreeze')
        else:
            self.freezeButton.setText('Freeze')
            
    def saveImage(self):
        filename_suggest = self.sampleName.text() + '.png'
                
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self,"Save image","","PNG Files (*.png)", options=options)
        
        if not(fileName.endswith('.png')):
            fileName += '.png'
        
        if self.includeOverlay.isChecked():
            cv2.imwrite(fileName, self.applyOverlay(self.currentImage))

        else:
            cv2.imwrite(fileName, self.currentImage)

        
    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
    
        # do processing here
        if not(self.freeze):
            self.currentImage = cv_img 
    
        qt_img = self.convert_cv_qt(self.applyOverlay(self.currentImage))
        self.camView.setImage(qt_img)
        
    
    def applyOverlay(self, img):
        cop = self.currentImage.copy()
        cop[self.overlay_mask==255] = (0,0,255)
        return cv2.addWeighted(cop, self.overlay_alpha, self.currentImage, 1-self.overlay_alpha, 0)
        
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, IMG_FORMAT)
        
        scale = 0.01 * self.scaleSlider.value()
        
        p = convert_to_Qt_format.scaled(SCALED_SIZE[0], SCALED_SIZE[1], Qt.KeepAspectRatio)
        # out = QPixmap.fromImage(p)
        return p

if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec())
