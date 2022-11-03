#!/usr/bin/env python
'''This script allows tracking to be performed until sampling from an image is likely to be within expected error bounds.'''

import argparse
import shutil
from dwi_tools import gen
import lil_wrappers
import mrtrix_wrappers as mrtrix
import os

#---------PARAMS-----------

def_minimumTrackCount = 1000 # Default minimum track count. Reducing this is not recommended
def_min_step = 1000 # Default minimum number of streamlines to collect each time tckgen is run
def_max_step = 5000 # Default maximum number of streamlines to collect each time tckgen is run

#---------PARAMS END--------
#---------METHODS--------------


def Run(save_to_tck, track_function, assess_function, min_step=def_min_step, max_step=def_max_step, minimum_trackCount=def_minimumTrackCount, append_if_file_exists=False, verbose=True):
	'''Generates a track file using a provided function, stopping when convergence criteria are met
	
	Arguments:
		save_to_tck:		Where to save the final tractogram to
		track_function: 	A function that calls tckgen in mrtrix. Must accept only the location to save in and how many steamlines to generate
		assess_function:	A function that assesses how many streamlines are required. Must accept only the location of the track file as input
		min_step:			Minimum number of streamlines to collect each time tckgen is run
		max_step:			Maximum number of streamlines to collect each time tckgen is run
		minimum_trackCount:	Minimum number of streamlines to collect
		append_if_file_exists:	If the file exists already, this appends streamlines to that file. If false, that file is overwritten. 
	
	Returns:	The estimated number of streamlines required, and the number of streamlines actually generated
	'''

	if append_if_file_exists and os.path.exists(save_to_tck):
		# The tck file already exists. We will continue adding streamlines to this file
		if verbose:
			print("tck file found. Any additional streamlines will be appended to this file")
		noTracksSoFar = mrtrix.GetTrackCount(save_to_tck, printCommands=False)
		noTracksRequired = max(minimum_trackCount, assess_function(save_to_tck)) if noTracksSoFar > 1 else minimum_trackCount #else statement avoids mrtrix crashing bug
	else:
		noTracksSoFar = 0
		noTracksRequired = minimum_trackCount

	tempFileLoc = gen.GetTempfileName(suffix="tck")

	try:
		while True:
			# Generate more tracks
			if noTracksSoFar >= noTracksRequired:
				# We have reached convergence
				break
			elif noTracksSoFar == 0:
				step = minimum_trackCount
			else:
				# Always collect at least the minimum step size
				# Failure to do so when the minimum steamline count has not been met results
				# in progressively smaller steps as the seed count reduces, which is extremely inefficient
				# and can effectively stop the minimum from ever being reached
				step = max(noTracksRequired - noTracksSoFar, min_step)
			
			if noTracksSoFar >= minimum_trackCount:
				step = min(max_step, max(step,min_step))
			
			if verbose:
				print("Step: " + str(step))
			track_function(tempFileLoc, int(step))

			# Append to the existing result
			if noTracksSoFar == 0:
				gen.Delete(save_to_tck)
				shutil.move(tempFileLoc, save_to_tck)
			else:
				lil_wrappers.TrackEdit([save_to_tck, "AppendStart", tempFileLoc, save_to_tck], verbose=False)
				gen.Delete(tempFileLoc)
			noTracksSoFar = mrtrix.GetTrackCount(save_to_tck, printCommands=False) # Don't assume that 'step' streamlines were actually generated

			# Update how many tracks we think are needed
			if noTracksSoFar < minimum_trackCount:
				# Haven't hit the minimum yet
				# Continue collecting. 
				noTracksRequired = minimum_trackCount
			else:
				# Calculate the minimum
				noTracksRequired = assess_function(save_to_tck)
			if verbose:
				print("Cur Tracks: " + str(noTracksSoFar))
				print("Predicted Tracks: " + str(noTracksRequired))

			
			# Check that tractography is actually generating streamlines
			if noTracksSoFar == 0:
				# Prevent an infinite loop
				message = "Streamline generation does not appear to be working. Check ROIs, unset -seeds if it is set"
				if minimum_trackCount < 1000:
					# if this is very low, we can generate zero streamlines simply by chance
					message = message + " and set minimum_trackCount to at least 1000"
				raise Exception(message)
			
	finally:
		gen.Delete(tempFileLoc)

	return noTracksRequired, noTracksSoFar
