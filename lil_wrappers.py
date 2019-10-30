#!/usr/bin/env python
"""Wrappers for calling LIL executable"""

import os
import subprocess
import gen
# --------------PARAMS---------------------


dir_thisScript = os.path.dirname(__file__) + "/"
loc_LILScripts = dir_thisScript + "LIL/NoStreamlinesRequired"


# --------------METHODS---------------------


def CopyOrientation(loc_orientationToCopy, loc_imageToChange, loc_saveTo, verbose=True):
	''' Copies orientation from one image to another
	
	ITK sometimes doesn't calculate the header matrix thing precisely enough and we end up wrong by very small amounts, then LIL kicks up a fuss
	This remedies by copying header info from one image into another
	
	Args:
		loc_orientationToCopy: location of a nifti image to copy orientation from
		loc_imageToChange: location of an image to copy voxels from
		loc_saveTo: location to save combined result to

	'''
	gen.Run([loc_LILScripts, "CopyOrientation", loc_orientationToCopy, loc_imageToChange, loc_saveTo], printCommands=verbose)


def NoStreamlinesReq(loc_tck, resolution, binarisationThreshold=0.001, target_dice=0.95, confidence=0.95, loc_roi=None, verbose=True):
	'''Calculates the number of streamlines required to achieve a reliable binary tractogram'''

	command = [loc_LILScripts, "streamline_count_tckmap",
			"tck", loc_tck, 
			"res", str(resolution), 
			"bint", str(binarisationThreshold),
			"target_metric", str(target_dice),
			"confidence", str(confidence)
			]
	
	if loc_roi is not None:
		command.append("roi")
		command.append(loc_roi)

	result = gen.Run_CaptureOutput(command, printCommands=verbose)
	
	return int(result.splitlines()[-1])


def NoStreamlinesReq_Sample(loc_tck, loc_image, target_sd, verbose=True):
	'''Calculates the number of streamlines required to achieve sampling from a tractogram/image pair
	
	Args:
		loc_tck:	Location of a tck file made with mrtrix
		loc_image:	Location of the image to sample from (e.g. the FA image)
		target_sd:	The target standard deviation
	'''

	command = [loc_LILScripts, "streamline_count_sample",
			"tck", loc_tck, 
			"sample_from", loc_image, 
			"sd", str(target_sd)]

	result = gen.Run_CaptureOutput(command, printCommands=verbose)
	
	return int(result.splitlines()[-1])


def TrackEdit(argumentArray, verbose=True, skipIfResultFound=False):
	"""Calls executable for editing tractograms"""

	if skipIfResultFound and os.path.exists(argumentArray[-1]):
		return ""

	result = gen.Run_CaptureOutput([loc_LILScripts, "TrackEdit"] + argumentArray, printCommands=verbose)
	return result

	'''Moves a tractogram from one space to another
		
	Args:
		loc_input: inputTractogram
		loc_transform: the transform, either in ANTS, flirt, or mrtrix format
		transformFormat: one of "ants_binary", "ants_plaintext", "flirt" or "mrtrix"
		loc_output: where to save the result
		verbose: whether to print the command to screen
		All others: See mrtrix help	
	'''

	deleteTemp = False
	deleteMrtrixFormatFile = False
	try:
		# Convert the transform
		# -- Convert from binary to plain text if needed
		if transformFormat == "ants_binary":
			loc_tempPlainTextTransform = gen.GetTempfileName(suffix="txt")
			gen.Run(["$ANTSPATH/ConvertTransformFile", 3, loc_transform, loc_tempPlainTextTransform])
			transformFormat = "ants_plaintext"
			loc_transform = loc_tempPlainTextTransform
			deleteTemp = True

		# -- Convert to mrtrix format if needed
		if (transformFormat == "flirt") or (transformFormat == "ants_plaintext"):
			loc_mrtrix_format = gen.GetTempfileName(suffix="txt")
			mrtrix.TransformConvert(loc_transform, True, loc_mrtrix_format, force=force, verbose=verbose)
			deleteMrtrixFormatFile = False
		elif transformFormat == "mrtrix":
			loc_mrtrix_format = loc_transform
		else:
			raise Exception("Unknown transform format:" + transformFormat)

		# Apply the transform
		TrackEdit([loc_input, "transform", loc_mrtrix_format, "true", loc_output], verbose=verbose)
	
	finally:
		# Clean up
		if deleteTemp:
			gen.Delete(loc_transform)
		
		if deleteMrtrixFormatFile:
			gen.Delete(loc_mrtrix_format)
