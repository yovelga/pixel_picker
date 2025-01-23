
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import spectral as spy


def load_rgb_image_from_folder(folder_path):
    """
    Load the RGB image from the specified folder.
    The RGB image is expected to have a .png extension in the main folder.
    """
    HS_folder_path = os.path.join(folder_path, 'HS')
    rgb_files = [file for file in os.listdir(HS_folder_path) if file.endswith(".png")]
    if rgb_files:
        rgb_path = os.path.join(HS_folder_path, rgb_files[0])
        image = cv2.imread(rgb_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise FileNotFoundError("No RGB image found in the folder.")


def load_hsi_image_from_folder(folder_path):
    """
    Load the HSI image from the `results` subfolder.
    The HSI image is expected to have a .hdr file in the `results` subfolder.
    """
    HS_folder_path = os.path.join(folder_path, 'HS')
    results_path = os.path.join(HS_folder_path, "results")
    if not os.path.exists(results_path):
        raise FileNotFoundError("No `results` folder found in the specified path.")

    hdr_files = [file for file in os.listdir(results_path) if file.endswith(".hdr")]
    if hdr_files:
        header_path = os.path.join(results_path, hdr_files[0])
        data_path = header_path.replace(".hdr", ".dat")
        if os.path.exists(data_path):
            hsi_obj = spy.envi.open(header_path, data_path)
            return np.array(hsi_obj.load())
        else:
            raise FileNotFoundError("Matching data file not found for the HSI header.")
    else:
        raise FileNotFoundError("No HSI header file found in the `results` folder.")


def load_rgb_image(file_path):
    image = cv2.imread(file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_hsi_image(header_path):
    data_path = header_path.replace(".hdr", ".dat")
    if os.path.exists(data_path):
        hsi_obj = spy.envi.open(header_path, data_path)
        return np.array(hsi_obj.load())
    else:
        raise FileNotFoundError("Matching data file not found for the HSI header.")


def export_histogram(image, file_path, band=None):
    if len(image.shape) == 3 and band is None:  # RGB Image
        pixel_values = image.mean(axis=(0, 1))
        plt.bar(["R", "G", "B"], pixel_values)
    elif band is not None:  # HSI Band
        pixel_values = image[:, :, band].flatten()
        plt.hist(pixel_values, bins=50, color="gray")
    else:
        raise ValueError("Invalid image or parameters.")
    plt.title("Pixel Histogram")
    plt.savefig(file_path)

def load_canon_rgb_image(folder_path):
    """
    Load the Canon RGB image from the specified folder.
    The Canon image is expected to have a .jpg extension.
    """
    rgb_path = os.path.join(folder_path, "RGB")
    if not os.path.exists(rgb_path):
        raise FileNotFoundError("No `RGB` folder found in the specified path.")

    jpg_files = [file for file in os.listdir(rgb_path) if file.lower().endswith(".jpg")]
    if jpg_files:
        image_path = os.path.join(rgb_path, jpg_files[0])
        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512))
        if image is None:
            raise FileNotFoundError(f"Failed to load RGB image from: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise FileNotFoundError("No JPG files found in the `RGB` folder.")



