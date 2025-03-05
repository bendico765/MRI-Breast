import pydicom
import os
import numpy as np

def read_dicomdir(dir_path: str) -> list:
	# opening the dicom files in the dir
	dicom_files = [
		os.path.join(dir_path, f) 
		for f in os.listdir(dir_path)
	]

	# Sort the files based on slice location (for this patient is already done)
	dicom_files.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)

	# read the actual dicom files
	slices = [pydicom.dcmread(f) for f in dicom_files]
    
	return slices
    
def get_3d_shape(dicom_slices: list) -> np.ndarray:
	pixel_arrays = [s.pixel_array for s in dicom_slices]

	# stack the slices to create a 3d array
	volume_3d = np.stack(pixel_arrays, axis=0)
    
	# flip the dicom on the vertical axis
	return volume_3d[:, ::-1, :]
    
def get_multiphase_mri(dir_path: str) -> tuple:
	"""
	Given the path to the multiphase mri, return a 5 element tuple of numpy mri volumes,
	one for each phase 
	"""
	volume = get_3d_shape(read_dicomdir(dir_path))
	l = np.array_split(volume, 5, axis=0)
	return (*l,)
    
def get_n_bins(array: np.ndarray) -> int:
	"""
	Given an array of data, compute the optimal number of bins for its histogram
	using the Freedman Diaconis rule
	"""
	q75, q25 = np.percentile(array, [75, 25])
	iqr = q75 - q25

	bin_width = 2 * iqr * pow(np.size(array), -1/3)
	return int((np.max(array) - np.min(array))/bin_width)
    
def get_dice(mask1: np.array, mask2: np.array) -> float:
	"""
	Computes the DICE coefficient of the numpy masks
	"""
	mask1 = mask1.astype(bool)
	mask2 = mask2.astype(bool)
	# Compute Dice coefficient
	intersection = np.logical_and(mask1, mask2)
	return 2. * intersection.sum() / (mask1.sum() + mask2.sum())
    
def bbox_3D(a: np.array) -> tuple:
	"""
	Returns the zyx coordinates of the smallest enclosure box of the input mask.
	The returned coordinates are z_min, z_max, y_min, y_max, x_min, x_max
	"""
	x = np.any(a, axis=(0, 1))
	y = np.any(a, axis=(0, 2))
	z = np.any(a, axis=(1, 2))
	xmin, xmax = np.where(x)[0][[0, -1]]
	ymin, ymax = np.where(y)[0][[0, -1]]
	zmin, zmax = np.where(z)[0][[0, -1]]
	return zmin, zmax, ymin, ymax, xmin, xmax
    
def get_temporal_resolutions(path: str) -> tuple:
	"""
	Load the multiphase mri and return the temporal resolutions of each phase
	"""
	dicom_files = read_dicomdir(path)

	dce1, dce2, dce3, dce4, dce5 = (*np.array_split(dicom_files, 5, axis=0),)

	return (
		dce1[0]["TemporalResolution"].value if "TemporalResolution" in dce1[0] else None,
		dce2[0]["TemporalResolution"].value if "TemporalResolution" in dce2[0] else None,
		dce3[0]["TemporalResolution"].value if "TemporalResolution" in dce3[0] else None,
		dce4[0]["TemporalResolution"].value if "TemporalResolution" in dce4[0] else None,
		dce5[0]["TemporalResolution"].value if "TemporalResolution" in dce5[0] else None
	)
    
def get_patient_intensity_values(multiphase_path: str, roi_path: np.ndarray) -> tuple:
	"""
	Given a patient row from a dataframe, returns a tuple of 4 arrays of length 5. 
	Each array contains 5 elements, one for each dce phase. The 4 arrays contain the
	maximum, mean, median and std of the intensity values of the region of interest
	"""
	# reading the multiphase mri and the region of interest
	dce1, dce2, dce3, dce4, dce5 = get_multiphase_mri(multiphase_path)
	roi = np.load(roi_path)

	# keep only the region of interest voxels for each dce
	dce1 = dce1[roi != 0]
	dce2 = dce2[roi != 0]
	dce3 = dce3[roi != 0]
	dce4 = dce4[roi != 0]
	dce5 = dce5[roi != 0]

	# compute for each phase the maximum, mean, median and std of the intensity values
	max_values = np.max([dce1, dce2, dce3, dce4, dce5], axis=1)
	mean_values = np.mean([dce1, dce2, dce3, dce4, dce5], axis=1)
	median_values = np.median([dce1, dce2, dce3, dce4, dce5], axis=1)
	std_values = np.std([dce1, dce2, dce3, dce4, dce5], axis=1)

	return max_values, mean_values, median_values, std_values
