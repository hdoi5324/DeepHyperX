from utils import open_file
import numpy as np
from scipy.io import netcdf
from netCDF4 import Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

CUSTOM_DATASETS_CONFIG = {
    "DFC2018_HSI": {
        "img": "2018_IEEE_GRSS_DFC_HSI_TR.HDR",
        "gt": "2018_IEEE_GRSS_DFC_GT_TR.tif",
        "download": False,
        "loader": lambda folder: dfc2018_loader(folder),
    },
    "transect_024": {
        "img": "2018_IEEE_GRSS_DFC_HSI_TR.HDR",
        "gt": "classmap.jpg",
        "download": False,
        "loader": lambda folder: curacao_loader(folder),
    }
}
def curacao_loader(folder):
    """--model svm --folder /Users/heather/Downloads/ --dataset transect_028 --training_sample 0.3"""
    refboard_class_no = 48

    transect_data = Dataset(folder + "transect.nc", "r", format="NETCDF4")
    class_data = Dataset(folder + "classmap.nc", "r", format="NETCDF4")

    height, width, bands = transect_data["cube"].shape

    # Normalise image wrt refboard
    img = transect_data['cube'][:height,:width,:].data
    gt = class_data['labels'][:height,:width,0].data
    indices = np.nonzero(gt == refboard_class_no)
    mean_refboard = np.mean(np.array(img[indices]), axis=0)
    norm = mean_refboard[50]

    height_offset = 0
    height = 1200 # Trim size of image for now
    img = img[height_offset:height_offset+height,:,:350]/norm
    gt = gt[height_offset:height_offset+height,:]

    rgb_bands = (240, 140, 60)

    label_class_name = class_data.variables['labels'].visual_classes
    label_class_no = class_data.variables['labels'].visual_class_ids
    label_values = label_class_no.size * ['None']
    for i, val in enumerate(zip(label_class_no, label_class_name)):
        print(val)
        label_values[val[0]] = val[1]

    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette


def dfc2018_loader(folder):
    img = open_file(folder + "2018_IEEE_GRSS_DFC_HSI_TR.HDR")[:, :, :-2]
    gt = open_file(folder + "2018_IEEE_GRSS_DFC_GT_TR.tif")
    gt = gt.astype("uint8")

    rgb_bands = (47, 31, 15)

    label_values = [
        "Unclassified",
        "Healthy grass",
        "Stressed grass",
        "Artificial turf",
        "Evergreen trees",
        "Deciduous trees",
        "Bare earth",
        "Water",
        "Residential buildings",
        "Non-residential buildings",
        "Roads",
        "Sidewalks",
        "Crosswalks",
        "Major thoroughfares",
        "Highways",
        "Railways",
        "Paved parking lots",
        "Unpaved parking lots",
        "Cars",
        "Trains",
        "Stadium seats",
    ]
    ignored_labels = [0]
    palette = None
    return img, gt, rgb_bands, ignored_labels, label_values, palette
