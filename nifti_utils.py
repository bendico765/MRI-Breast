import nibabel as nib
import numpy as np
from os import listdir
from os.path import isfile, join, exists
import matplotlib.pyplot as plt
import pandas as pd
import pydicom

__all__ = (
    'get_duke_image',
    'get_ispy2_image',
    'get_ispy2_dce',
    'get_duke_mask',
    'get_ispy2_mask',
    'plot_image'
)

def get_duke_image(filepath: str) -> np.ndarray:
    img = nib.load(filepath)
    array = img.get_fdata()

    # In the following code geometrical transformations are applied
    # in order to get the same MRI image that is produced by Mango viewer

    # transpose the dimensions to re-orient the image
    # the axis order is z, y, x
    array = np.transpose(array, (2,1,0))

    # flip the axeses
    array = array[::-1, ::-1, ::-1]

    return array

def get_ispy2_image(filepath: str) -> np.ndarray:
    img = nib.load(filepath)
    array = img.get_fdata()

    # In the following code geometrical transformations are applied
    # in order to get the same MRI image that is produced by Mango viewer

    # transpose the dimensions to re-orient the image
    # the axis order is z, y, x
    array = np.transpose(array, (2,1,0))

    # flip the axeses
    array = array[:, ::-1, ::-1]

    return array

def get_duke_mask(filepath: str) -> tuple:
    mask = nib.load(filepath)
    array = mask.get_fdata()

    # In the following code geometrical transformations are applied
    # in order to get the same MRI image that is produced by Mango viewer

    # transpose the dimensions to re-orient the image
    # the axis order is z, y, x
    array = np.transpose(array, (2,1,0))

    # flip the axeses
    array = array[::-1, ::-1, ::-1]

    # get slices with roi
    x_indexes, y_indexes, z_indexes = array.nonzero()

    # eliminate duplicate entries
    x_indexes = np.unique(x_indexes)
    y_indexes = np.unique(y_indexes)
    z_indexes = np.unique(z_indexes)

    return (
        array,
        x_indexes,
        y_indexes,
        z_indexes
    )

def get_ispy2_mask(filepath: str) -> tuple:
    mask = nib.load(filepath)
    array = mask.get_fdata()

    # In the following code geometrical transformations are applied
    # in order to get the same MRI image that is produced by Mango viewer

    # transpose the dimensions to re-orient the image
    # the axis order is z, y, x
    array = np.transpose(array, (2,1,0))

    # flip the axeses
    array = array[:, ::-1, ::-1]

    # get slices with roi
    x_indexes, y_indexes, z_indexes = array.nonzero()

    # eliminate duplicate entries
    x_indexes = np.unique(x_indexes)
    y_indexes = np.unique(y_indexes)
    z_indexes = np.unique(z_indexes)

    return (
        array,
        x_indexes,
        y_indexes,
        z_indexes
    )

def plot_image(
        array: np.ndarray,
        mask: np.ndarray = None,
        title: str = None,
        filepath: str = None,
        dpi: int = None,
    ):
    plt.figure(figsize=(15, 15))

    max_intensity = np.max(array)
    plt.imshow(
        np.where(mask != 0, max_intensity, array),
        cmap = plt.cm.gray
    )

    plt.title(title)
    if filepath:
        plt.savefig(filepath, dpi=dpi)
    else:
        plt.show()

def get_ispy2_dce_paths(uncurated_dataset_path: str, patient_id: str) -> list:
    """
    Returns the list of complete paths of uncurated DCE MRI scans for a given patient.

    :param dataset_path: path where the ISPY2 uncurated informations are stored
    :param patient_id: id of the patient e.g. "ISPY2-100899"
    :return:
    """
    patient_path = f"{uncurated_dataset_path}/{patient_id}"

    # check if patient folder exists
    if not exists(patient_path):
        return []

    mr_folders = [ x for x in listdir(patient_path) if "ISPY2MRIT0" in x]
    if not mr_folders:
        return []

    mr0_folder = mr_folders[0]

    # keywords that allow to identify each phase
    keywords = [
        ".000000-ISPY2 Ph1",
        ".000000-ISPY2 Ph2",
        ".000000-ISPY2 Ph3",
        ".000000-ISPY2 Ph4",
        ".000000-ISPY2 Ph5",
    ]
    
    # for each phase, get the whole path
    dce_phases = [
        x
        for x in listdir(f"{patient_path}/{mr0_folder}") if any(map(lambda y: y in x, keywords))
    ]
    dce_phases.sort()

    return [f"{patient_path}/{mr0_folder}/{dce_phase}" for dce_phase in dce_phases]

def get_ispy2_patient_temporal_resolutions(uncurated_dataset_path: str, patient_id: str) -> tuple:
    """
    :param uncurated_dataset_path: path where the uncurated ISPY2 informations are stored
    :param patient_id: id of the patient e.g. "ISPY2-103693"
    :return:
    """
    dce_paths = get_ispy2_dce_paths(uncurated_dataset_path, patient_id)
    # check if the patient has dce images

    if not dce_paths:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    results = []
    for dce_path in dce_paths:
        filename = listdir(dce_path)[0]

        try:
            results.append(pydicom.dcmread(f"{dce_path}/{filename}").TemporalResolution)
        except AttributeError: # the dicom has no attribute Temporal Resolution
            results.append(np.nan)

    return tuple(results)

def get_ispy2_temporal_resolutions(mama_mia_dataset_path: str, ispy2_uncurated_dataset_path: str) -> pd.DataFrame:
    """
    :param mama_mia_dataset_path: path to the mama-mia dataset
    :param ispy2_uncurated_dataset_path: path to the ispy2 original uncurated dataset
    :return:
    """
    images_folder = f"{mama_mia_dataset_path}/images"

    results = []
    for patient_id in (d for d in listdir(images_folder) if not isfile(join(images_folder, d)) and "ISPY2" in d):
        # remove _ put -
        patient_id = patient_id.replace("_", "-")

        temporal_resolutions = get_ispy2_patient_temporal_resolutions(ispy2_uncurated_dataset_path, patient_id)
        results.append([patient_id, *temporal_resolutions])

    return pd.DataFrame(
        results,
        columns = [
            "Patient ID",
            "TemporalResolution 1",
            "TemporalResolution 2",
            "TemporalResolution 3",
            "TemporalResolution 4",
            "TemporalResolution 5",
            "TemporalResolution 6"
        ]
    )


def get_ispy2_patient_intensity_values(multiphase_filepaths: list[str], roi_filepath: str) -> tuple:
    max_values = []
    mean_values = []
    median_values = []
    std_values = []

    for multiphase_filepath in multiphase_filepaths:
        # reading the multiphase mri and the region of interest
        multiphase = get_ispy2_image(multiphase_filepath)
        roi, _, _, _  = get_ispy2_mask(roi_filepath)

        # keep only the region of interest voxels
        multiphase = multiphase[roi != 0]

        # compute the maximum, mean, median and std of the intensity values
        max_values.append(np.max(multiphase))
        mean_values.append(np.mean(multiphase))
        median_values.append(np.median(multiphase))
        std_values.append(np.std(multiphase))

    return max_values, mean_values, median_values, std_values

def get_ispy2_dynamic_information(lesions_df_filepath: str, temporal_resolutions_df_filepath: str):
    lesions_df = pd.read_csv(lesions_df_filepath)
    temporal_resolutions_df = pd.read_csv(temporal_resolutions_df_filepath)

    # uncurated dataset use - for the patient id (e.g. ISPY2-100899), while curated one uses ISPY2_100899
    temporal_resolutions_df["Patient ID"] = temporal_resolutions_df["Patient ID"].apply(lambda s: s.replace("-", "_"))

    # merge the two datasets
    df = lesions_df.merge(temporal_resolutions_df, on="Patient ID")

    for _, row in df.iterrows():
        result = []
        result.append(row["Patient ID"])

        # get the temporal resolutions between one pulse and the next one
        tr2 = row["TemporalResolution 2"]
        tr3 = row["TemporalResolution 3"]
        tr4 = row["TemporalResolution 4"]
        tr5 = row["TemporalResolution 5"]

        max_values, mean_values, median_values, std_values = get_ispy2_patient_intensity_values([row[f"Multiphase {i} path"] for i in range(1,6)], row["Roi mask Filepath"])

        # from the multiphase get the max, mean, median and std values for each phase
        for values in [max_values, mean_values, median_values, std_values]:
            phase1_value = values[0]
            phase2_value = values[1]
            phase3_value = values[2]
            phase4_value = values[3]
            phase5_value = values[4]

            # compute the derivative as (t_i - t_i-1)/delta_t
            phase5_derivative = (phase5_value - phase4_value) / tr5
            phase4_derivative = (phase4_value - phase3_value) / tr4
            phase3_derivative = (phase3_value - phase2_value) / tr3
            phase2_derivative = (phase2_value - phase1_value) / tr2

            result.append(phase2_derivative)
            result.append(phase3_derivative)
            result.append(phase4_derivative)
            result.append(phase5_derivative)

        yield result

def get_ispy2_dataset(folder: str):
    """

    :param folder: path to the mama-mia dataset folder
    :return:
    """
    images_folder = f"{folder}/images"

    return pd.DataFrame((
        [
            patient_id,
            f"{folder}/images/{patient_id}/{patient_id}_0000.nii.gz",
            f"{folder}/images/{patient_id}/{patient_id}_0001.nii.gz",
            f"{folder}/images/{patient_id}/{patient_id}_0002.nii.gz",
            f"{folder}/images/{patient_id}/{patient_id}_0003.nii.gz",
            f"{folder}/images/{patient_id}/{patient_id}_0004.nii.gz",
            f"{folder}/images/{patient_id}/{patient_id}_0005.nii.gz",
            f"{folder}/segmentations/expert/{patient_id}.nii.gz"
        ]
        for patient_id in (d for d in listdir(images_folder) if not isfile(join(images_folder, d)) and "ISPY2" in d)),
        columns = [
            "Patient ID",
            "Multiphase 1 path",
            "Multiphase 2 path",
            "Multiphase 3 path",
            "Multiphase 4 path",
            "Multiphase 5 path",
            "Multiphase 6 path",
            "Roi mask Filepath"
        ]
    )
