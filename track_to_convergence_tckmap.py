'''This script allows tracking to be performed until trackmap convergence criteria are met.'''

import os
import argparse
import shutil
import gen
import lil_wrappers
import track_to_convergence_base

#---------PARAMS-----------

def_bint = 0.001 # Default binary threshold
def_target_dice = 0.95 # Default target dice
def_target_reliability = 0.95 # Default target reliability
def_minimumTrackCount = 1000 # Default minimum track count. Reducing this is not recommended
def_min_step = 1000 # Default minimum number of streamlines to collect each time tckgen is run
def_max_step = 5000 # Default maximum number of streamlines to collect each time tckgen is run

#---------PARAMS END--------
#---------METHODS--------------



def Run(save_to_tck, track_function, resolution, bint=def_bint, target_dice=def_target_dice, target_confidence=def_target_reliability, min_step=def_min_step, max_step=def_max_step, minimum_trackCount=def_minimumTrackCount, loc_roi=None, verbose=True):
	'''Generates a track file using a provided function, stopping when convergence criteria are met
	
	Arguments:
		save_to_tck:		Where to save the final tractogram to
		track_function: 	A function that calls tckgen in mrtrix
		resolution:			Trackmap resolution in mm
		bint:				Binarisation threshold; see paper for details
		target_dice:		Target dice score; see paper for details
		target_confidence:	Target reliability; Set at 95% in the paper
		min_step:			Minimum number of streamlines to collect each time tckgen is run
		max_step:			Maximum number of streamlines to collect each time tckgen is run
		minimum_trackCount:	Minimum number of streamlines to collect
		loc_roi:			Full path of a binary ROI in which to measure the other metrics. Streamline vertices outside this region are ignored.
		verbose:			Whether to print informational messages
		
	Returns:				The number of streamlines generated
	'''
	
	
	
	def AssessStreamlinesReq(loc_save_to):
		return lil_wrappers.NoStreamlinesReq(loc_save_to, resolution, binarisationThreshold=bint, target_dice=target_dice, confidence=target_confidence, loc_roi=loc_roi, verbose=verbose)
		 
	
	return track_to_convergence_base.Run(save_to_tck, track_function, assess_function=AssessStreamlinesReq, min_step=min_step, max_step=max_step, minimum_trackCount=minimum_trackCount, verbose=verbose)
	


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_to_tck", help="Where to save the track to", required=True, action="store")
	parser.add_argument("--track_command", help="Command to generate tracks (do not include any -select or save location in the command)", required=True, action="store")
	parser.add_argument("--roi", help="The region in which to assess reliability", required=False, default=None, action="store")
	parser.add_argument("--resolution", type=float, help="The trackmap resolution in mm", required=True, action="store")
	parser.add_argument("--bint", default=def_bint, type=float, help="The binarisation threshold", action="store")
	parser.add_argument("--target_dice", default=def_target_dice, type=float, help="The target dice score", action="store")
	parser.add_argument("--target_confidence", default=def_target_reliability, type=float, help="The target confidence (e.g. 0.95)", action="store")
	parser.add_argument("--min_step", default=def_min_step, type=int, help="The minimum number of streamlines to acquire per step", action="store")
	parser.add_argument("--max_step", default=def_max_step, type=int, help="The maximum number of streamlines to acquire per step", action="store")
	parser.add_argument("--minimum_track_count", default=def_minimumTrackCount, type=int, help="The minimum number of streamlines to acquire", action="store")
	parser.add_argument("--quiet", help="Hide informational messages", action="store_true")

	args = parser.parse_args()
 

	def Track(loc_to, noTracks):
		gen.Run([args.track_command, "-select", str(noTracks), "-force", loc_to], printCommands=(not args.quiet))

	Run(save_to_tck=args.save_to_tck, resolution=args.resolution, track_function=Track, bint=args.bint, target_dice=args.target_dice, target_confidence=args.target_confidence, min_step=args.min_step, max_step=args.max_step, minimum_trackCount=args.minimum_track_count, loc_roi=args.roi, verbose=(not args.quiet))
