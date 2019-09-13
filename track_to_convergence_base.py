'''This script allows tracking to be performed until sampling from an image is likely to be within expected error bounds.'''

import argparse
import shutil
import gen
import LILWrappers as LIL


#---------PARAMS-----------

def_minimumTrackCount = 1000 # Default minimum track count. Reducing this is not recommended
def_min_step = 1000 # Default minimum number of streamlines to collect each time tckgen is run
def_max_step = 5000 # Default maximum number of streamlines to collect each time tckgen is run

#---------PARAMS END--------
#---------METHODS--------------


def Run(save_to_tck, track_function, assess_function, min_step=def_min_step, max_step=def_max_step, minimum_trackCount=def_minimumTrackCount, verbose=True):
	'''Generates a track file using a provided function, stopping when convergence criteria are met
	
	Arguments:
		save_to_tck:		Where to save the final tractogram to
		track_function: 	A function that calls tckgen in mrtrix. Must accept only the location to save in and how many steamlines to generate
		assess_function:	A function that assesses how many streamlines are required. Must accept only the location of the track file as input
		min_step:			Minimum number of streamlines to collect each time tckgen is run
		max_step:			Maximum number of streamlines to collect each time tckgen is run
		minimum_trackCount:	Minimum number of streamlines to collect
	
	'''

	noTracksSoFar = 0

	tempFileLoc = gen.GetTempfileName(suffix="tck")
	noTracksRequired = minimum_trackCount

	try:
		while True:
			# Generate more tracks
			if noTracksSoFar >= noTracksRequired:
				# We have reached convergence
				break
			elif noTracksSoFar == 0:
				step = minimum_trackCount
			else:
				step = noTracksRequired - noTracksSoFar
			
			step = min(max_step, max(step,min_step))
			
			if verbose:
				print("Step: " + str(step))
			track_function(tempFileLoc, step)

			# Append to the existing result
			if noTracksSoFar == 0:
				gen.Delete(save_to_tck)
				shutil.move(tempFileLoc, save_to_tck)
			else:
				LIL.TrackEdit([save_to_tck, "AppendStart", tempFileLoc, save_to_tck], verbose=False)
			noTracksSoFar = noTracksSoFar + step

			# Update how many tracks we think are needed
			noTracksRequired = assess_function(save_to_tck)
			if verbose:
				print("Cur Tracks: " + str(noTracksSoFar))
				print("Predicted Tracks: " + str(noTracksRequired))
	finally:
		gen.Delete(tempFileLoc)

	return noTracksSoFar
