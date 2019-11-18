#!/usr/bin/env python
"""Wrappers for calling MRTrix"""

import os
import datetime
import gen
import numpy as np
import tempfile
# --------------PARAMS---------------------


loc_thisScript = os.path.dirname(__file__) + "/" 
dir_thisScript = os.path.dirname(loc_thisScript) + "/"

# Find where mrtrix is located
dir_MRTrix = os.environ.get("MRtrix3")
if dir_MRTrix is None:
	# Assume mrtrix is in the system path
	dir_MRTrix = ""
else:
	# Defined by an environment variable
	dir_MRTrix = dir_MRTrix + "/"


# --------------METHODS---------------------

def GetImageSize_Vox(loc_in):
	result = gen.Run_CaptureOutput([dir_MRTrix + "mrinfo", loc_in, "-size"], printCommands=False)
	return ParseToNumbers(result)


def GetTrackCount(loc_tck, printCommands=True):
	'''Returns the track count for a tck file'''
	output = gen.Run_CaptureOutput([dir_MRTrix + "tckinfo", "-count", loc_tck], printCommands=printCommands)
	output = str(output)
	noTracksTotal=float(output.strip().split('\n')[-1].split(' ')[-1])
	return noTracksTotal


def ParseToNumbers(result):
	numbers = [float(n) for n in result.split()]
	return numbers


def MRConvert(loc_in, loc_out, coord_axis=None, coord_coord=None, force=False, verbose=True):
	''' Runs MRConvert

	Args:
		loc_in: the path to the input file or dicom directory
		loc_out: the save location
		coord_axis: the axis to take an image from. Also define coord_coord
		coord_coord: the index to take from axis coord_axis
	
	'''
	command = [dir_MRTrix + "mrconvert"]

	if not ((coord_axis is None) == (coord_coord is None)):
		raise Exception("Both or neither coord arguments must be provided")
	
	if (coord_axis is not None):
		command.append("-coord")
		command.append(str(coord_axis))
		command.append(str(coord_coord))

	if force:
		command.append("-force")

	command.append(loc_in)
	command.append(loc_out)

	gen.Run(command, printCommands=verbose)




def MRCrop(loc_in, loc_out, axis_indexStartEnd=[], force=False, verbose=True):
	''' Runs mrcrop
	
	Args:
		axis_indexStartEnd: three numbers: the axis index, the starting index, and the ending index

	Returns:
		Nothing is returned
	'''

	command = [dir_MRTrix + "mrcrop"]
	if len(axis_indexStartEnd) == 3:
		command = command + ["-axis", str(axis_indexStartEnd[0]), str(axis_indexStartEnd[1]), str(axis_indexStartEnd[2]) ]
	elif len(axis_indexStartEnd) > 0:
		raise Exception("axis information needs three arguments")

	if force:
		command.append("-force")

	command.append(loc_in)
	command.append(loc_out)

	gen.Run(command, printCommands=verbose)


def MREdit(loc_in, loc_out, plane_axisCoordValue=[], force=False, verbose=True):
	''' Runs mredit
	
	Args:
		plane_axisCoordValue: three numbers: the axis index, the starting index, and the value to set

	Returns:
		Nothing is returned
	'''

	command = [dir_MRTrix + "mredit"]
	if len(plane_axisCoordValue) == 3:
		command = command + ["-plane", str(plane_axisCoordValue[0]), str(plane_axisCoordValue[1]), str(plane_axisCoordValue[2]) ]
	elif len(plane_axisCoordValue) > 0:
		raise Exception("axis information needs three arguments")

	if force:
		command.append("-force")

	command.append(loc_in)
	command.append(loc_out)

	gen.Run(command, printCommands=verbose)


def MRCalc(loc_in, commands, loc_out, force=False, verbose=True):
	''' Runs mrcalc

	Args:
		loc_in: the path to the input file or dicom directory
		loc_out: the save location
		scale: define the new voxel size for the output image (number)
	
	'''
	command = [dir_MRTrix + "mrcalc", loc_in] 
	
	if force:
		command.append("-force")

	command = command + commands + [loc_out]

	gen.Run(command, printCommands=verbose)


def MRResize(loc_in, loc_out, scale=None, force=False, verbose=True):
	''' Runs mrresize

	Args:
		loc_in: the path to the input file or dicom directory
		loc_out: the save location
		scale: define the new voxel size for the output image (number)
	
	'''
	command = [dir_MRTrix + "mrresize"]

	if scale is not None:
		command.append("-scale")
		command.append(str(scale))

	if force:
		command.append("-force")

	command.append(loc_in)
	command.append(loc_out)

	gen.Run(command, printCommands=verbose)


def TckSample(loc_in, sampleFrom, saveTo=None, stat_tck=None, force=False, verbose=True, quiet=False):
	'''Runs tcksample and returns the result as a list of numbers if saveTo is set to None'''

	tempFileUsed = False
	if saveTo is None:
		saveTo = gen.GetTempfileName("txt")
		tempFileUsed = True

	command = [dir_MRTrix + "tcksample", loc_in, sampleFrom, saveTo]

	if stat_tck is not None:
		command.append("-stat_tck")
		command.append(stat_tck)

	if quiet:
		command.append("-quiet")

	if force:
		command.append("-force")

	try:
		gen.Run(command, printCommands=verbose)

		if tempFileUsed:
			return np.loadtxt(saveTo)

	finally:
		gen.Delete(saveTo)



def TckGen(loc_fodImage=None, loc_saveTo=None,
loc_SeedImage=None, loc_SeedDixelImage=None, nSeeds=None, seed_unidirectional=False,
select=None, maxLength=None, minLength=None, downsample=None, 
loc_includeList=None, loc_includeOrderedList=None, loc_exclude=None,
stopOnInclude=False, 
loc_retrackImage=None, retrackROIPosition=None, retrackAttempts=0,
manualArgs=None, force=False):
	''' Runs and times tckgen
	
	Note that some arguments are only supported in CSIRO's CONSULT build of mrtrix and may not be supported by your build
	
	Args:
		loc_includeList: a list of locations of include ROIs
		loc_includeOrderedList: a list of locations of ordered include ROIs (only if using CSIRO special build)
		manualArgs: any pre-prepared arguments to add
		All others: See mrtrix help


	Returns:
		The time taken to track
	
	'''
	if loc_fodImage is not None:
		tckgenCommands = [dir_MRTrix + 'tckgen', loc_fodImage]
	else:
		tckgenCommands = []

	# Seeds
	if loc_SeedImage is not None:
		tckgenCommands = tckgenCommands + ["-seed_image", loc_SeedImage]

	if loc_SeedDixelImage is not None:
		tckgenCommands = tckgenCommands + ["-seed_dixel_image", loc_SeedDixelImage]
	

	if nSeeds is not None:
		tckgenCommands = tckgenCommands + ["-seeds", str(nSeeds)]

	if seed_unidirectional:
		tckgenCommands.append("-seed_unidirectional")

	# Retracking
	if loc_retrackImage is not None:
		tckgenCommands = tckgenCommands + ["-retrack_image", loc_retrackImage]
	
	if retrackROIPosition is not None:
		tckgenCommands = tckgenCommands + ["-retrack_roi_position", retrackROIPosition]

	if retrackAttempts is not None and retrackAttempts > 0:
		tckgenCommands = tckgenCommands + ["-retrack_attempts", str(retrackAttempts)]
		


	# Number of tracks
	if select is not None:
		tckgenCommands = tckgenCommands + ["-select", str(select)]



	# Track size
	if maxLength is not None:
		tckgenCommands = tckgenCommands + ["-maxlength", str(maxLength)]
	if minLength is not None:
		tckgenCommands = tckgenCommands + ["-minlength", str(minLength)]

	# Down sampling
	if downsample is not None:
		if downsample <= 0:
			raise Exception("downsample must be > 0")
		tckgenCommands = tckgenCommands + ["-downsample", str(downsample)]
	
	# Inclusion and exclusion
	if stopOnInclude:
		tckgenCommands.append("-stop")

	if loc_includeList is not None:
		for cur in loc_includeList:
			tckgenCommands.append("-include")
			tckgenCommands.append(cur)

	if loc_includeOrderedList is not None:
		for cur in loc_includeOrderedList:
			tckgenCommands.append("-include_ordered")
			tckgenCommands.append(cur)
	
	if loc_exclude is not None:
		tckgenCommands = tckgenCommands + ["-exclude", loc_exclude]

	# Force
	if force:
		tckgenCommands.append("-force")

	# Any other args
	if manualArgs is not None:
		tckgenCommands = tckgenCommands + manualArgs

	if loc_saveTo is not None:
		tckgenCommands.append(loc_saveTo)

	# Run and time
	start = datetime.datetime.now()
	
	gen.Run(tckgenCommands)
	
	end = datetime.datetime.now()
	diff = end-start

	return diff


def TrackMap(loc_in, saveTo, stat_tck=None, voxSize=None, template=None, precise=False, upsample=None, force=False, quiet=False, verbose=True):
	'''Runs tckmap'''

	command = [dir_MRTrix + "tckmap", loc_in, saveTo]

	if (voxSize is None) and (template is None):
		raise Exception("Voxel size or template must be provided")

	if voxSize is not None:
		command.append("-vox")
		command.append(str(voxSize))

	if template is not None:
		command.append("-template")
		command.append(template)

	if stat_tck is not None:
		command.append("-stat_tck")
		command.append(stat_tck)

	if precise:
		command.append("-precise")

	if upsample is not None:
		command.append("-upsample")
		command.append(str(upsample))

	if quiet:
		command.append("-quiet")

	if force:
		command.append("-force")

	gen.Run(command, printCommands=verbose)

def TckTransform(loc_input, loc_transform, transformFormat, loc_movingImageInTransform, loc_output, force=False, quiet=False, verbose=False):
	'''Moves a tractogram from one space to another. Also see LIL
		
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
			TransformConvert(loc_transform, True, loc_mrtrix_format, force=force, quiet=quiet, verbose=verbose)
			deleteMrtrixFormatFile = False
		elif transformFormat == "mrtrix":
			loc_mrtrix_format = loc_transform
		else:
			raise Exception("Unknown transform format:" + transformFormat)

		# -- Convert into a warp in a ridiculous series of operations
		loc_warp =  gen.GetTempfileName("mif")
		with tempfile.TemporaryDirectory() as dir_temp: 
			gen.Run([dir_MRTrix + "warpinit", loc_movingImageInTransform, dir_temp + "warp[].mif"])
			gen.Run([dir_MRTrix + "mrtransform", dir_temp + "warp[].mif -linear", loc_mrtrix_format, dir_temp + "warp_[].mif"])
			loc_warp_inv = dir_temp + "4dWarp_inv.mif"
			gen.Run([dir_MRTrix + "warpcorrect", dir_temp + "warp_[].mif", loc_warp_inv])
			gen.Run([dir_MRTrix + "warpinvert", loc_warp_inv, loc_warp])
		
		# Apply the transform
		command = [dir_MRTrix + "tcktransform"]
		AppendStdArgs(command, quiet, force)
		command = command + [loc_input, loc_warp, loc_output ]
		gen.Run(command)
	
	finally:
		gen.Delete(loc_warp)

		# Clean up
		if deleteTemp:
			gen.Delete(loc_transform)
		
		if deleteMrtrixFormatFile:
			gen.Delete(loc_mrtrix_format)


def TransformConvert(loc_input, itkTrueFSLFalse, loc_out, force=False, quiet=False, verbose=True):

	command = [dir_MRTrix + "transformconvert", loc_input, ("itk" if itkTrueFSLFalse else "flirt") + "_import", loc_out]
	
	AppendStdArgs(command, quiet, force)

	gen.Run(command, printCommands=verbose)


def AppendStdArgs(command, quiet, force):
	
	if quiet:
		command.append("-quiet")

	if force:
		command.append("-force")
