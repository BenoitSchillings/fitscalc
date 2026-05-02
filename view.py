import sys
import os
import json
import socket
import threading
import numpy as np
import cv2
from astropy.io import fits
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QTimer, Qt, QObject, pyqtSignal
from util import * # Assuming util.py with fit_gauss_circular exists
from ser import Ser

import pyqtgraph as pg
import argparse


SOCK_PATH = f"/tmp/fitscalc-viewer-{os.getuid()}.sock"


class IpcServer(QObject):
    """Background-thread Unix-socket server that emits filenames into the GUI thread."""
    files_received = pyqtSignal(list)

    def __init__(self, sock_path=SOCK_PATH):
        super().__init__()
        self.sock_path = sock_path
        self._stop = threading.Event()
        self._sock = None
        self._thread = None

    def start(self):
        try:
            os.unlink(self.sock_path)
        except FileNotFoundError:
            pass
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.bind(self.sock_path)
            self._sock.listen(4)
            self._sock.settimeout(0.5)
        except OSError as e:
            print(f"viewer IPC disabled (bind failed): {e}", file=sys.stderr)
            self._sock = None
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        try:
            os.unlink(self.sock_path)
        except FileNotFoundError:
            pass

    def _loop(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                buf = b""
                conn.settimeout(2.0)
                try:
                    while True:
                        chunk = conn.recv(8192)
                        if not chunk:
                            break
                        buf += chunk
                except socket.timeout:
                    continue
            try:
                msg = json.loads(buf.decode())
                files = msg.get("files", [])
                if isinstance(files, list) and files:
                    self.files_received.emit([str(f) for f in files])
            except Exception as e:
                print(f"viewer IPC: bad message: {e}", file=sys.stderr)

def compute_hfd(image):
    """
    Compute the half flux diameter (HFD) of a star image in a 2D array.
    
    Args:
        image (numpy.ndarray): 2D array containing the star image.
        
    Returns:
        float: The half flux diameter (HFD) of the star image.
    """
    # Find the centroid (center of mass) of the star image
    total_flux = np.sum(image)
    if total_flux == 0:
        return 0 # Avoid division by zero
    y, x = np.indices(image.shape)
    y_centroid = np.sum(y * image) / total_flux
    x_centroid = np.sum(x * image) / total_flux
    
    # Sort the pixel values in descending order
    sorted_pixels = np.sort(image.ravel())[::-1]
    
    # Calculate the cumulative sum of the sorted pixel values
    cumsum = np.cumsum(sorted_pixels)
    
    # Find the radius at which the cumulative sum reaches half of the total flux
    half_flux = total_flux / 2
    idx = np.searchsorted(cumsum, half_flux, side='right')
    radius = np.sqrt((idx - 1) / np.pi)
    
    # The HFD is twice the radius
    return 2 * radius

class ImageNavigator(QtWidgets.QMainWindow):
    def __init__(self, filenames):
        super().__init__()
        # Store file paths instead of image data to save memory
        self.filenames = filenames
        self.currentImage = None # This will hold the data for the current image
        self.currentIndex = 0
        self.histLevels = None  # Store histogram levels
        self.prevTransform = None  # Initialize variable to store previous transformation

        # SER file support
        self.ser_file = None  # Current open SER file
        self.ser_frame_index = 0  # Current frame within SER file

        self.initUI()
        
    def click(self, event):
        event.accept()   
        if self.currentImage is None:
            return
            
        self.pos = event.pos()
        x = int(self.pos.x())
        y = int(self.pos.y())
        
        # Ensure coordinates are within image bounds
        if 0 <= x < self.currentImage.shape[0] and 0 <= y < self.currentImage.shape[1]:
            # Extract a 20x20 box around the click
            data = self.currentImage[max(0, x-10):x+10, max(0, y-10):y+10]
            if data.size == 0:
                return

            data = data - np.min(data)
            print("--- Star Analysis ---")
            print(f"Clicked at: ({x}, {y})")
            print(f"Max value in box: {np.max(data)}")
            print(f"HFD = {compute_hfd(data):.2f}")
            fwhm = fit_gauss_circular(data)
            print(f"FWHM = {fwhm:.2f}")
            print("---------------------")

    def initUI(self):
        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)
        
        self.layout = QtWidgets.QVBoxLayout()
        self.centralWidget.setLayout(self.layout)
        
        self.imageView = pg.ImageView()
        self.layout.addWidget(self.imageView)
        
        self.buttonLayout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.buttonLayout)

        self.prevButton = QtWidgets.QPushButton("Previous")
        self.prevButton.clicked.connect(self.prevImage)
        self.buttonLayout.addWidget(self.prevButton)

        self.nextButton = QtWidgets.QPushButton("Next")
        self.nextButton.clicked.connect(self.nextImage)
        self.buttonLayout.addWidget(self.nextButton)

        # SER frame slider (hidden by default)
        self.frameLayout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(self.frameLayout)

        self.frameLabel = QtWidgets.QLabel("Frame:")
        self.frameLayout.addWidget(self.frameLabel)

        self.frameSlider = QtWidgets.QSlider(Qt.Horizontal)
        self.frameSlider.setMinimum(0)
        self.frameSlider.setMaximum(0)
        self.frameSlider.valueChanged.connect(self.onFrameSliderChanged)
        self.frameLayout.addWidget(self.frameSlider)

        self.frameIndexLabel = QtWidgets.QLabel("0 / 0")
        self.frameIndexLabel.setMinimumWidth(100)
        self.frameLayout.addWidget(self.frameIndexLabel)

        # Hide SER controls initially
        self.frameLabel.hide()
        self.frameSlider.hide()
        self.frameIndexLabel.hide()

        self.resize(1600, 1200) # Set a default window size

        # --- Setup Actions and Shortcuts ---
        self.toggleFullScreenAction = QtWidgets.QAction("Toggle Full-Screen", self)
        self.toggleFullScreenAction.setShortcut("f")
        self.toggleFullScreenAction.triggered.connect(self.toggleFullScreen)
        self.addAction(self.toggleFullScreenAction)

        self.deleteFileAction = QtWidgets.QAction("Delete File", self)
        self.deleteFileAction.setShortcut("d")
        self.deleteFileAction.triggered.connect(self.deleteFile)
        self.addAction(self.deleteFileAction)

        self.showFilenameAction = QtWidgets.QAction("Show Filename", self)
        self.showFilenameAction.setShortcut("n")
        self.showFilenameAction.triggered.connect(self.showFilename)
        self.addAction(self.showFilenameAction)

        # New auto-scale action with 'A' shortcut
        self.autoScale90Action = QtWidgets.QAction("Auto Scale to 90%", self)
        self.autoScale90Action.setShortcut("A")
        self.autoScale90Action.triggered.connect(self.setAutoScale90)
        self.addAction(self.autoScale90Action)

        self.imageView.getImageItem().mouseClickEvent = self.click

        # IPC: accept new file lists from fitscalc on a Unix socket
        self.ipc = IpcServer()
        self.ipc.files_received.connect(self.replaceFiles)
        self.ipc.start()

        # Load the first image
        self.updateImage()

    def replaceFiles(self, files):
        """Replace the navigator's file list with `files` and refresh."""
        self.closeSer()
        self.filenames = list(files)
        self.currentIndex = 0
        # Reset cached display state so the new sequence auto-scales fresh
        self.histLevels = None
        self.prevTransform = None
        self.updateImage()
        # Bring the window forward
        if self.isMinimized():
            self.showNormal()
        self.raise_()
        self.activateWindow()

    def setAutoScale90(self):
        """Set image levels to background (median) and 90th percentile."""
        if self.currentImage is None:
            return
            
        flat_data = self.currentImage.flatten()
        background = np.median(flat_data)
        max_level = np.percentile(flat_data, 90)
        
        # Fallback for low-contrast images
        if background >= max_level:
            max_level = np.percentile(flat_data, 99)

        self.imageView.setLevels(background, max_level)
        self.histLevels = (background, max_level)

    def toggleFullScreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def prevImage(self):
        if self.currentIndex > 0:
            self.saveHistogramSettings()
            self.saveViewTransform()
            self.currentIndex -= 1
            self.updateImage()
        
    def nextImage(self):
        # Use self.filenames to check length
        if self.currentIndex < len(self.filenames) - 1:
            self.saveHistogramSettings()
            self.saveViewTransform()
            self.currentIndex += 1
            self.updateImage()
        
    def saveHistogramSettings(self):
        # Save the current histogram settings
        self.histLevels = self.imageView.getHistogramWidget().getLevels()
        
    def saveViewTransform(self):
        # Save the current view transformation matrix
        self.prevTransform = self.imageView.view.getViewBox().transform()

    def closeSer(self):
        """Close any open SER file."""
        if self.ser_file is not None:
            self.ser_file.close()
            self.ser_file = None

    def updateImage(self):
        """Loads and displays the image at the current index."""
        if not self.filenames or self.currentIndex < 0:
            self.closeSer()
            self.imageView.clear()
            self.currentImage = None
            self.setWindowTitle("Image Viewer")
            self.hideSERControls()
            return

        filename = self.filenames[self.currentIndex]
        ext = os.path.splitext(filename)[1].lower()

        if ext == '.ser':
            self.loadSERFile(filename)
        else:
            self.loadFITSFile(filename)

    def loadFITSFile(self, filename):
        """Load and display a FITS file."""
        self.closeSer()
        self.hideSERControls()

        try:
            with fits.open(filename) as hdul:
                data = hdul[0].data
                if data is None:
                    raise ValueError("No data in primary HDU.")
                self.currentImage = np.rot90(data)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load FITS file {filename}:\n{e}')
            self.imageView.clear()
            self.currentImage = None
            self.setWindowTitle(f"Error loading {os.path.basename(filename)}")
            return

        title = f"{os.path.basename(filename)} ({self.currentIndex + 1}/{len(self.filenames)})"
        self.setWindowTitle(title)

        self.imageView.setImage(self.currentImage, autoHistogramRange=False)

        if self.histLevels:
            self.imageView.setLevels(*self.histLevels)
        else:
            self.setAutoScale90()

        self.restoreViewTransform()

    def loadSERFile(self, filename):
        """Load and display a SER file with frame navigation."""
        self.closeSer()

        try:
            self.ser_file = Ser(filename)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load SER file {filename}:\n{e}')
            self.imageView.clear()
            self.currentImage = None
            self.setWindowTitle(f"Error loading {os.path.basename(filename)}")
            self.hideSERControls()
            return

        # Show SER controls and configure slider
        self.showSERControls()
        self.frameSlider.blockSignals(True)
        self.frameSlider.setMaximum(self.ser_file.count - 1)
        self.ser_frame_index = 0
        self.frameSlider.setValue(0)
        self.frameSlider.blockSignals(False)

        self.loadSERFrame(set_levels=True)

    def loadSERFrame(self, set_levels=False):
        """Load the current frame from the open SER file."""
        if self.ser_file is None:
            return

        # Save current levels before loading new frame
        current_levels = self.imageView.getHistogramWidget().getLevels()

        try:
            data = self.ser_file.load_img(self.ser_frame_index)
            self.currentImage = np.rot90(data)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load SER frame {self.ser_frame_index}:\n{e}')
            return

        filename = self.filenames[self.currentIndex]
        title = f"{os.path.basename(filename)} - Frame {self.ser_frame_index + 1}/{self.ser_file.count} ({self.currentIndex + 1}/{len(self.filenames)})"
        self.setWindowTitle(title)
        self.frameIndexLabel.setText(f"{self.ser_frame_index + 1} / {self.ser_file.count}")

        self.imageView.setImage(self.currentImage, autoHistogramRange=False)

        if set_levels:
            if self.histLevels:
                self.imageView.setLevels(*self.histLevels)
            else:
                self.setAutoScale90()
        else:
            # Restore the levels that were set before loading the new frame
            self.imageView.setLevels(*current_levels)

        self.restoreViewTransform()

    def onFrameSliderChanged(self, value):
        """Handle frame slider value change."""
        self.ser_frame_index = value
        self.loadSERFrame()

    def showSERControls(self):
        """Show SER frame navigation controls."""
        self.frameLabel.show()
        self.frameSlider.show()
        self.frameIndexLabel.show()

    def hideSERControls(self):
        """Hide SER frame navigation controls."""
        self.frameLabel.hide()
        self.frameSlider.hide()
        self.frameIndexLabel.hide()

    def restoreViewTransform(self):
        if self.prevTransform:
            self.imageView.view.getViewBox().setTransform(self.prevTransform)

    def deleteFile(self):
        if not self.filenames or self.currentIndex < 0:
            return

        filename = self.filenames[self.currentIndex]

        # Close SER file before deleting
        self.closeSer()

        try:
            os.remove(filename)
            print(f"Deleted file: {filename}")
            self.filenames.pop(self.currentIndex)

            if not self.filenames:
                self.currentIndex = -1
                self.updateImage() # Clear the view
            else:
                if self.currentIndex >= len(self.filenames):
                    self.currentIndex = len(self.filenames) - 1
                self.updateImage() # Load the next/previous image
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to delete file: {e}')

    def showFilename(self):
        if self.currentIndex >= 0 and self.currentIndex < len(self.filenames):
            filename = self.filenames[self.currentIndex]
            print(f"Current file: {filename}")

    def closeEvent(self, event):
        """Clean up when closing the window."""
        self.closeSer()
        if hasattr(self, "ipc") and self.ipc is not None:
            self.ipc.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FITS/SER image viewer with on-demand loading.")
    parser.add_argument("files", nargs='+', help="List of FITS or SER files to view")
    
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    
    if not args.files:
        print("Error: No FITS files specified.", file=sys.stderr)
        sys.exit(1)
        
    # Pass the list of filenames directly to the navigator
    navigator = ImageNavigator(args.files)
    navigator.show()

    sys.exit(app.exec_())
