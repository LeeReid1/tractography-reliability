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


def NoStreamlinesReq(loc_tck, resolution, binarisationThreshold=0.001, target_metric=0.95, confidence=0.95, loc_roi=None, loc_dixelsTextFileOrMifImage=None, endsOnly=False, dixels_unidirectional=False, verbose=True):
	'''Calculates the number of streamlines required to achieve a reliable binary tractogram'''

	command = [loc_LILScripts, 
			"tck", loc_tck, 
			"res", str(resolution), 
			"target_metric", str(target_metric),
			"confidence", str(confidence)
			]
	
	if loc_roi is not None:
		command.append("roi")
		command.append(loc_roi)

	if loc_dixelsTextFileOrMifImage is not None:
		# Dixel trackmap
		command.append("dixel_dir_file")
		command.append(loc_dixelsTextFileOrMifImage)

		if dixels_unidirectional:
			command.append("dixel_unidirectional")
	else:
		# Binary trackmap
		command.append("bint")
		command.append(str(binarisationThreshold))

	if endsOnly:
		command.append("ends_only")

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
