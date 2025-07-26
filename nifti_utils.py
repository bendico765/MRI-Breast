import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

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

    x_indexes = []
    y_indexes = []
    z_indexes = []

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

    x_indexes = []
    y_indexes = []
    z_indexes = []

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

def get_ispy2_dce(folder: str):
    patients = [
        "ISPY2_100899",
        "ISPY2_102011",
        "ISPY2_102212",
        "ISPY2_103693",
        "ISPY2_103939",
        "ISPY2_104268",
        "ISPY2_104384",
        "ISPY2_105286",
        "ISPY2_105513",
        "ISPY2_107700",
    ]

    results = []

    for patient in patients:
        images_path = f"{folder}/images/{patient}"
        segmentation_path = f"{folder}/segmentations/expert/{patient}.nii.gz"
        results.append((
            get_ispy2_image(f"{images_path}/{patient}_0000.nii.gz"),
            get_ispy2_image(f"{images_path}/{patient}_0001.nii.gz"),
            get_ispy2_image(f"{images_path}/{patient}_0002.nii.gz"),
            get_ispy2_image(f"{images_path}/{patient}_0003.nii.gz"),
            get_ispy2_image(f"{images_path}/{patient}_0004.nii.gz"),
            get_ispy2_mask(segmentation_path)
        ))

    return results