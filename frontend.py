import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QPushButton,QSizePolicy,
    QSlider, QWidget, QHBoxLayout, QSizePolicy, QComboBox, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QListWidget, QMessageBox
from backend import load_rgb_image, load_hsi_image, export_histogram, load_hsi_image_from_folder, load_rgb_image_from_folder, load_canon_rgb_image
import cv2
import numpy as np
from PyQt5.QtWidgets import QInputDialog


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import csv
import os


class HSI_RGB_Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HSI and RGB Viewer")
        self.setGeometry(100, 100, 1600, 800)
        self.installEventFilter(self)

        # Initialize variables
        self.rgb_image = None
        self.hsi_image = None
        self.current_band = 0
        self.saved_pixels = []  # List of dictionaries with 'position' and 'intensity'
        self.folder_list = []  # רשימת נתיבי תיקיות
        self.current_folder_index = -1  # אינדקס התיקייה הנוכחית

        # Set up the UI
        self.setup_ui()

    def setup_ui(self):
        # Set up the main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Grid layout for images
        self.image_grid = QGridLayout()
        self.layout.addLayout(self.image_grid)

        self.folder_path_label = QLabel("Current Folder: None")
        self.folder_path_label.setAlignment(Qt.AlignCenter)
        self.folder_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.layout.addWidget(self.folder_path_label)

        # RGB Image Label
        self.rgb_label = QLabel("RGB Image (HS folder) will be displayed here")
        self.rgb_label.setAlignment(Qt.AlignCenter)
        self.rgb_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.rgb_label.setMinimumSize(500, 500)
        self.image_grid.addWidget(self.rgb_label, 0, 0)

        # HSI Image Label
        self.hsi_label = QLabel("HSI Image will be displayed here")
        self.hsi_label.setAlignment(Qt.AlignCenter)
        self.hsi_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.hsi_label.setMinimumSize(500, 500)
        self.image_grid.addWidget(self.hsi_label, 0, 1)

        # Canon RGB Image Label
        self.canon_rgb_label = QLabel("Canon RGB Image will be displayed here")
        self.canon_rgb_label.setAlignment(Qt.AlignCenter)
        self.canon_rgb_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canon_rgb_label.setMinimumSize(500, 500)
        self.image_grid.addWidget(self.canon_rgb_label, 0, 2)



        # Associate the mousePressEvent with handle_hsi_click
        self.hsi_label.mousePressEvent = self.handle_hsi_click
        self.rgb_label.mousePressEvent = self.handle_rgb_click

        # Controls
        self.controls_layout = QHBoxLayout()
        self.layout.addLayout(self.controls_layout)

        self.load_folder_button = QPushButton("Load Folder")
        self.load_folder_button.clicked.connect(self.handle_load_folder)
        self.controls_layout.addWidget(self.load_folder_button)

        self.band_slider = QSlider(Qt.Horizontal)
        self.band_slider.setMinimum(0)
        self.band_slider.setValue(0)
        self.band_slider.valueChanged.connect(self.update_hsi_band)
        self.controls_layout.addWidget(self.band_slider)

        self.save_button = QPushButton("Export Histogram")
        self.save_button.clicked.connect(self.handle_export_histogram)
        self.controls_layout.addWidget(self.save_button)

        # Add UI for saved pixels
        self.saved_pixels_list = QListWidget()
        self.saved_pixels_list.itemClicked.connect(self.display_saved_pixel_histogram)
        self.saved_pixels_list.setSelectionMode(QListWidget.MultiSelection)
        self.layout.addWidget(self.saved_pixels_list)

        # Add buttons for managing saved pixels
        self.save_pixel_button = QPushButton("Save Pixel")
        self.save_pixel_button.clicked.connect(self.save_pixel)
        self.controls_layout.addWidget(self.save_pixel_button)

        self.delete_pixels_button = QPushButton("Delete Selected Pixels")
        self.delete_pixels_button.clicked.connect(self.delete_selected_pixels)
        self.controls_layout.addWidget(self.delete_pixels_button)

        self.export_csv_button = QPushButton("Export to CSV")
        self.export_csv_button.clicked.connect(self.export_to_csv)
        self.controls_layout.addWidget(self.export_csv_button)

        # # Plot layout for intensity graph
        # self.plot_layout = QVBoxLayout()
        # self.layout.addLayout(self.plot_layout)

        # Plot layout for intensity graph
        self.plot_widget = QWidget()
        self.plot_widget.setMinimumSize(700, 500)
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.layout.addWidget(self.plot_widget)

        # Status bar
        self.status_bar = self.statusBar()

        # ComboBox for pixel type
        self.pixel_type_combo = QComboBox()
        self.pixel_type_combo.addItems(["Regular", "Crack"])
        self.controls_layout.addWidget(self.pixel_type_combo)

        # Add "Add Folder" button
        self.add_folder_button = QPushButton("Add list")
        self.add_folder_button.clicked.connect(self.handle_add_folder)
        self.controls_layout.addWidget(self.add_folder_button)

        # Add "Previous" button
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.handle_previous_folder)
        self.controls_layout.addWidget(self.prev_button)

        # Add "Next" button
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.handle_next_folder)
        self.controls_layout.addWidget(self.next_button)

    def handle_load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            try:
                # Debugging: Print the full folder path
                print(f"Selected folder path: {folder_path}")

                # Extract folder_main and folder_date
                folder_main = os.path.basename(os.path.dirname(folder_path))  # "1_03"
                folder_date = os.path.basename(folder_path)  # "18.09.24"
                print(f"Extracted folder_main: {folder_main}")
                print(f"Extracted folder_date: {folder_date}")

                # Save folder_main and folder_date
                self.folder_main = folder_main if folder_main else "Unknown"
                self.folder_date = folder_date if folder_date else "Unknown"

                # Load RGB Image (from HS folder)
                self.rgb_image = load_rgb_image_from_folder(folder_path)
                self.display_image(self.rgb_image, is_hsi=False)

                # Load HSI Image
                self.hsi_image = load_hsi_image_from_folder(folder_path)
                self.band_slider.setMaximum(self.hsi_image.shape[2] - 1)
                self.update_hsi_band()

                # Load Canon RGB Image (from RGB folder)
                self.canon_rgb_image = load_canon_rgb_image(folder_path)
                self.display_image(self.canon_rgb_image, is_hsi=False, target_label=self.canon_rgb_label)

                self.status_bar.showMessage(f"Images Loaded Successfully from Folder: {folder_path}")
            except FileNotFoundError as e:
                self.status_bar.showMessage(f"Error: {e}")
            except Exception as e:
                self.status_bar.showMessage(f"An unexpected error occurred: {e}")

    def handle_load_rgb(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load RGB Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_path:
            self.rgb_image = load_rgb_image(file_path)
            self.display_image(self.rgb_image)
            self.status_bar.showMessage("RGB Image Loaded")

    def delete_selected_pixels(self):
        """
        Delete selected pixels from the saved_pixels list and update the QListWidget.
        """
        selected_items = self.saved_pixels_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "No pixels selected for deletion.")
            return

        # Iterate over selected items and remove them from the list and memory
        for item in selected_items:
            index = self.saved_pixels_list.row(item)  # Find the index of the item in the QListWidget
            self.saved_pixels.pop(index)  # Remove from memory
            self.saved_pixels_list.takeItem(index)  # Remove from QListWidget

        self.status_bar.showMessage("Selected pixels deleted.")

    def handle_load_hsi(self):
        header_path, _ = QFileDialog.getOpenFileName(self, "Load HSI Header", "", "Header Files (*.hdr)")
        if header_path:
            try:
                self.hsi_image = load_hsi_image(header_path)
                self.current_band = 0
                self.band_slider.setMaximum(self.hsi_image.shape[2] - 1)
                self.update_hsi_band()
                self.status_bar.showMessage("HSI Image Loaded")
            except FileNotFoundError as e:
                self.status_bar.showMessage(str(e))

    def update_hsi_band(self):
        if self.hsi_image is not None:
            self.current_band = self.band_slider.value()
            band_image = self.hsi_image[:, :, self.current_band]

            # Normalize the band image
            band_image = cv2.normalize(band_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Apply rotation if necessary
            band_image = cv2.rotate(band_image, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees clockwise

            self.display_image(band_image, is_hsi=True)
            self.status_bar.showMessage(f"Displaying HSI Band {self.current_band}")

    def display_image(self, image, is_hsi=False, target_label=None):
        """
        Display the given image in the appropriate QLabel (RGB, Canon RGB, or HSI).
        """
        label = target_label if target_label else (self.hsi_label if is_hsi else self.rgb_label)

        if is_hsi:
            height, width = image.shape
            q_image = QImage(image.data, width, height, QImage.Format_Grayscale8)
        else:
            height, width, channels = image.shape
            bytes_per_line = channels * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

    def plot_intensity_density(self, intensity_values):
        """
        Plot the intensity density of a selected pixel.
        """
        # Clear previous plots
        for i in reversed(range(self.plot_layout.count())):
            self.plot_layout.itemAt(i).widget().setParent(None)

        # Create the Matplotlib figure
        figure = Figure()
        canvas = FigureCanvas(figure)
        self.plot_layout.addWidget(canvas)

        # Add the plot
        ax = figure.add_subplot(111)
        ax.plot(intensity_values, marker="o")
        ax.set_title("Intensity Density Plot")
        ax.set_xlabel("Spectral Band")
        ax.set_ylabel("Intensity")
        canvas.draw()

    def handle_export_histogram(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Histogram", "", "PNG Files (*.png)")
        if file_path:
            if self.rgb_image is not None:
                export_histogram(self.rgb_image, file_path)
            elif self.hsi_image is not None:
                export_histogram(self.hsi_image, file_path, band=self.current_band)
            self.status_bar.showMessage(f"Histogram exported to {file_path}")

    def handle_rgb_click(self, event):
        if self.rgb_image is not None and self.hsi_image is not None:
            label_width = self.rgb_label.width()
            label_height = self.rgb_label.height()
            img_height, img_width, _ = self.rgb_image.shape

            # Calculate the actual pixel in the image
            x = int(event.pos().x() * img_width / label_width)
            y = int(event.pos().y() * img_height / label_height)

            # היפוך ציר ה-x
            x = img_width - x - 1


            # Ensure the pixel coordinates are within bounds
            if 0 <= x < img_width and 0 <= y < img_height:
                # Access the pixel intensity values using the correct order [x, y]
                intensity_values = self.hsi_image[x, y, :]
                self.last_clicked_pixel = (x, y, intensity_values)  # Save the last clicked pixel
                self.plot_intensity_density(intensity_values)  # Display the intensity plot
                self.status_bar.showMessage(f"Selected pixel ({x}, {y}) from RGB, displaying HSI intensity values.")
            else:
                QMessageBox.warning(self, "Warning", "Selected pixel is out of bounds.")

    def handle_hsi_click(self, event):
        if self.hsi_image is not None:
            label_width = self.hsi_label.width()
            label_height = self.hsi_label.height()
            img_height, img_width, _ = self.hsi_image.shape

            # Calculate the actual pixel in the image
            x = int(event.pos().x() * img_width / label_width)
            y = int(event.pos().y() * img_height / label_height)

            # Flip the y-axis to match the image's coordinate system
            x = img_width - x - 1

            # Ensure the pixel coordinates are within bounds
            if 0 <= x < img_width and 0 <= y < img_height:
                # Access the pixel intensity values using the correct order [x, y]
                intensity_values = self.hsi_image[x, y, :]
                self.last_clicked_pixel = (x, y, intensity_values)  # Save the last clicked pixel
                self.plot_intensity_density(intensity_values)  # Display the intensity plot
                self.status_bar.showMessage(f"Selected pixel ({x}, {y}) from HSI image.")
            else:
                QMessageBox.warning(self, "Warning", "Selected pixel is out of bounds.")

    def save_pixel(self):
        if hasattr(self, 'last_clicked_pixel') and self.last_clicked_pixel:
            x, y, intensity = self.last_clicked_pixel
            folder_main = getattr(self, 'folder_main', "Unknown")
            folder_date = getattr(self, 'folder_date', "Unknown")
            pixel_type = self.pixel_type_combo.currentText()  # Get selected type
            self.saved_pixels.append({
                'position': (x, y),
                'intensity': intensity,
                'folder_main': folder_main,
                'folder_date': folder_date,
                'type': pixel_type
            })
            # Update the display in the QListWidget
            display_text = f"Pixel ({x}, {y}) - {pixel_type} [Folder: {folder_main}, Subfolder: {folder_date}]"
            self.saved_pixels_list.addItem(display_text)
            self.status_bar.showMessage(f"Pixel ({x}, {y}) saved as {pixel_type}")
        else:
            QMessageBox.warning(self, "Warning", "No pixel selected to save.")



    def export_to_csv(self):
        if not self.saved_pixels:
            QMessageBox.warning(self, "Warning", "No pixels to export.")
            return

        # Generate CSV file name
        # base_name = os.path.basename(self.folder_date) if hasattr(self, 'folder_date') else ""
        # csv_file_name = f"detected_pixels_{base_name}.csv"
        csv_file_name = f"detected_pixels.csv"

        # Write to CSV
        with open(csv_file_name, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([ "Folder_Main", "Folder_Date", "Type","X", "Y",] +
                            [f"Band_{i}" for i in range(len(self.saved_pixels[0]['intensity']))])
            for pixel in self.saved_pixels:
                x, y = pixel['position']
                folder_main = pixel['folder_main']
                folder_date = pixel['folder_date']
                pixel_type = pixel['type']
                row = [x, y, folder_main, folder_date, pixel_type] + list(pixel['intensity'])
                writer.writerow(row)

        QMessageBox.information(self, "Export Successful", f"Pixels exported to {csv_file_name}")

    def display_saved_pixel_histogram(self, item):
        index = self.saved_pixels_list.row(item)
        pixel_data = self.saved_pixels[index]
        intensity_values = pixel_data['intensity']
        self.plot_intensity_density(intensity_values)
        self.status_bar.showMessage(
            f"Displaying histogram for Pixel {pixel_data['position']} [Folder: {pixel_data['folder_main']}, Subfolder: {pixel_data['folder_date']}]")

    def handle_add_folder(self):
        # Open a dialog for the user to input folder paths
        text, ok = QInputDialog.getMultiLineText(self, "Add Folders",
                                                 "Paste folder paths (one per line):")
        if ok and text.strip():  # Check if user clicked OK and input is not empty
            # Split text into lines and filter out empty lines
            folder_paths = [line.strip() for line in text.split("\n") if line.strip()]

            # Add each folder to the list and update the UI
            for folder_path in folder_paths:
                if os.path.exists(folder_path):
                    self.folder_list.append(folder_path)
                    # self.saved_pixels_list.addItem(folder_path)
                    self.status_bar.showMessage(f"Added folder: {folder_path}")
                else:
                    QMessageBox.warning(self, "Warning", f"Folder not found: {folder_path}")

            # Load the first folder immediately if this is the first addition
            if len(self.folder_list) == len(folder_paths):  # First time adding folders
                self.current_folder_index = 0
                self.load_current_folder()

    def handle_previous_folder(self):
        if self.current_folder_index > 0:
            self.current_folder_index -= 1
            self.load_current_folder()
            print(self.current_folder_index)
        else:
            QMessageBox.warning(self, "Warning", "This is the first folder in the list.")

    def handle_next_folder(self):
        if self.current_folder_index < len(self.folder_list) - 1:
            self.current_folder_index += 1
            self.load_current_folder()
            print(self.current_folder_index)
        else:
            QMessageBox.warning(self, "Warning", "This is the last folder in the list.")

    def load_current_folder(self):
        folder_path = self.folder_list[self.current_folder_index]
        self.folder_path_label.setText(f"Current Folder: {folder_path}")  # Update label
        try:
            # Update folder_main and folder_date based on current folder
            self.folder_main = os.path.basename(os.path.dirname(folder_path))  # Parent folder
            self.folder_date = os.path.basename(folder_path)  # Current folder name

            # Load RGB Image (from HS folder)
            self.rgb_image = load_rgb_image_from_folder(folder_path)
            self.display_image(self.rgb_image, is_hsi=False)

            # Load HSI Image
            self.hsi_image = load_hsi_image_from_folder(folder_path)
            self.band_slider.setMaximum(self.hsi_image.shape[2] - 1)
            self.update_hsi_band()

            # Load Canon RGB Image (from RGB folder)
            self.canon_rgb_image = load_canon_rgb_image(folder_path)
            self.display_image(self.canon_rgb_image, is_hsi=False, target_label=self.canon_rgb_label)

            self.status_bar.showMessage(f"Loaded folder: {folder_path}")
        except FileNotFoundError as e:
            self.status_bar.showMessage(f"Error: {e}")
        except Exception as e:
            self.status_bar.showMessage(f"An unexpected error occurred: {e}")

    def next_folder(self):
        if self.current_folder_index < len(self.folder_list) - 1:
            self.current_folder_index += 1
            self.load_current_folder()
        else:
            self.status_bar.showMessage("You are at the last folder.")

    def prev_folder(self):
        if self.current_folder_index > 0:
            self.current_folder_index -= 1
            self.load_current_folder()
        else:
            self.status_bar.showMessage("You are at the first folder.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.save_pixel()  # Save pixel
        elif event.key() == Qt.Key_D:  # Move to the previous folder
            self.prev_folder()
        elif event.key() == Qt.Key_A:  # Move to the next folder
            self.next_folder()
        else:
            super().keyPressEvent(event)  # Handle other keys normally


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HSI_RGB_Viewer()
    viewer.show()
    sys.exit(app.exec_())
