'''This script allows tracking to be performed until trackmap convergence criteria are met.'''

import os
import argparse
import shutil
import gen
import track_to_convergence_base
import mrtrix_wrappers as mrtrix
import numpy as np

#---------PARAMS-----------

def_minimumTrackCount = 1000 # Default minimum track count. Reducing this is not recommended
def_min_step = 1000 # Default minimum number of streamlines to collect each time tckgen is run
def_max_step = 5000 # Default maximum number of streamlines to collect each time tckgen is run

#---------PARAMS END--------
#---------METHODS--------------


def Run(loc_save_to_tck, track_function, loc_image, target_sd, min_step=def_min_step, max_step=def_max_step, minimum_trackCount=def_minimumTrackCount, verbose=True):
	'''Generates a track file using a provided function, stopping when convergence criteria are met
	
	Arguments:
		loc_save_to_tck:	Where to save the final tractogram to
		track_function: 	A function that calls tckgen in mrtrix
		loc_image:			The location of the image to sample from
		target_sd:			The target standard deviation
		min_step:			Minimum number of streamlines to collect each time tckgen is run
		max_step:			Maximum number of streamlines to collect each time tckgen is run
		minimum_trackCount:	Minimum number of streamlines to collect
		verbose:			Whether to print informational messages
		
	Returns:	A tuple containing the number_of_streamlines_required, number_of_streamlines_generated
	'''

	def AssessStreamlinesReq(loc_save_to):
		# Measure the mean microstructural value for each streamline
		measurements = mrtrix.TckSample(loc_save_to, loc_image, saveTo=None, stat_tck="mean", verbose=False, quiet=True)
		
		# Calculate the sample variance of these values
		variance = np.var(measurements, ddof=1) # sample variance
		
		# Calculate approximately how many streamlines will be required, using the equation:
		# n = (1.96*2)^2 * sample_variance / W ^ 2 
		# where W is the desired width of the 95% confidence interval
		width_of_95CI = target_sd * 1.96 * 2 # W
		WSq = width_of_95CI * width_of_95CI # W^2
		est = int(np.ceil(15.3664 * variance / WSq)) # 15.3664 is (2*1.96)^2
		return est
		
	
	return track_to_convergence_base.Run(loc_save_to_tck, track_function, assess_function=AssessStreamlinesReq, min_step=min_step, max_step=max_step, minimum_trackCount=minimum_trackCount, verbose=verbose)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_to_tck", help="Where to save the track to", required=True, action="store")
	parser.add_argument("--im", help="Full path to the image from which to sample from (e.g. the FA image)", required=True, action="store")
	parser.add_argument("--track_command", help="Command to generate tracks (do not include any -select or save location in the command)", required=True, action="store")
	parser.add_argument("--sd", help="The target standard deviation", required=True, type=float, action="store")
	parser.add_argument("--min_step", default=def_min_step, type=int, help="The minimum number of streamlines to acquire per step", action="store")
	parser.add_argument("--max_step", default=def_max_step, type=int, help="The maximum number of streamlines to acquire per step", action="store")
	parser.add_argument("--minimum_track_count", default=def_minimumTrackCount, type=int, help="The minimum number of streamlines to acquire", action="store")
	parser.add_argument("--quiet", help="Hide informational messages", action="store_true")

	args = parser.parse_args()
 

	def Track(loc_to, noTracks):
		gen.Run([args.track_command, "-select", str(noTracks), "-force", loc_to], printCommands=(not args.quiet))

	Run(loc_save_to_tck=args.save_to_tck, track_function=Track, loc_image=args.im, target_sd=args.sd, min_step=args.min_step, max_step=args.max_step, minimum_trackCount=args.minimum_track_count, verbose=(not args.quiet))
