'''This script allows tracking to be performed until trackmap convergence criteria are met.'''

import os
import argparse
import shutil
import track_to_convergence_base
from dwi_tools import gen
from dwi_tools import mrtrix_wrappers as mrtrix
import numpy as np

#---------PARAMS-----------

def_minimumTrackCount = 1000 # Default minimum track count. Reducing this is not recommended
def_min_step = 1000 # Default minimum number of streamlines to collect each time tckgen is run
def_max_step = 5000 # Default maximum number of streamlines to collect each time tckgen is run

#---------PARAMS END--------
#---------METHODS--------------


def Run(loc_save_to_tck, track_function, loc_image, target_sd, min_step=def_min_step, max_step=def_max_step, minimum_trackCount=def_minimumTrackCount, append_if_file_exists=False, verbose=True):
	'''Generates a track file using a provided function, stopping when convergence criteria are met
	
	Arguments:
		loc_save_to_tck:		Where to save the final tractogram to
		track_function: 		A function that calls tckgen in mrtrix
		loc_image:				The location of the image to sample from, or a list of such locations
		target_sd:				The target standard deviation, or a list of these if supplying a list for loc_image
		min_step:				Minimum number of streamlines to collect each time tckgen is run
		max_step:				Maximum number of streamlines to collect each time tckgen is run
		minimum_trackCount:		Minimum number of streamlines to collect
		append_if_file_exists:	If the file exists already, this appends streamlines to that file. If false, that file is overwritten.
		verbose:				Whether to print informational messages
		
	Returns:	A tuple containing the number_of_streamlines_required, number_of_streamlines_generated
	'''

	try:
		iterator = iter(loc_image)
	except TypeError:
		# not iterable
		# input is just a string
		loc_image = [loc_image]

	try:
		iterator = iter(target_sd)
	except TypeError:
		# not iterable
		# input is just a string
		target_sd = [target_sd]

	if len(target_sd) != len(loc_image):
		raise Exception("number of target_sd must match number of loc_image")

	
	if verbose:
		print("Targets:")
		for i in range(len(loc_image)):
			print(str(loc_image[i]) + ":\t" + str(target_sd[i]))

	def AssessStreamlinesReq(loc_save_to):

		def AssessForOneImage(loc_microstructuralImage, target_standardDev):
			# Measure the mean microstructural value for each streamline
			measurements = mrtrix.TckSample(loc_save_to, loc_microstructuralImage, saveTo=None, stat_tck="mean", verbose=False)
			
			# Remove any NaNs
			origCount = len(measurements)
			measurements = measurements[~np.isnan(measurements)]
			if len(measurements) != origCount:
				raise Exception("Image contains NANs. Remove NaNs from image.")

			# Calculate the sample variance of these values
			variance = np.var(measurements, ddof=1) # sample variance
			
			# Calculate approximately how many streamlines will be required, using the equation:
			# n = (1.96*2)^2 * sample_variance / W ^ 2 
			# where W is the desired width of the 95% confidence interval
			width_of_95CI = target_standardDev * 1.96 * 2 # W
			WSq = width_of_95CI * width_of_95CI # W^2
			est = int(np.ceil(15.3664 * variance / WSq)) # 15.3664 is (2*1.96)^2
			return est
		
		est = 0
		for i in range(len(loc_image)):
			estReq = AssessForOneImage(loc_image[i], target_sd[i])
			est = max(est, estReq)

			if verbose:
				print("No Predicted (" + loc_image[i] + "):\t" + str(estReq))
		
		return est
		
	
	return track_to_convergence_base.Run(loc_save_to_tck, track_function, assess_function=AssessStreamlinesReq, min_step=min_step, max_step=max_step, minimum_trackCount=minimum_trackCount, append_if_file_exists=append_if_file_exists, verbose=verbose)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--save_to_tck", help="Where to save the track to", required=True, action="store")
	parser.add_argument("--im", help="Full path to the image from which to sample from (e.g. /home/me/fa.nii). This flag can be provided multiple times for multiple images.", required=True, action='append')
	parser.add_argument("--track_command", help="Command to generate tracks (do not include any -select or save location in the command)", required=True, action="store")
	parser.add_argument("--sd", help="The target standard deviation. One --sd flag must be provided for every image", required=True, type=float, action='append')
	parser.add_argument("--min_step", default=def_min_step, type=int, help="The minimum number of streamlines to acquire per step", action="store")
	parser.add_argument("--max_step", default=def_max_step, type=int, help="The maximum number of streamlines to acquire per step", action="store")
	parser.add_argument("--minimum_track_count", default=def_minimumTrackCount, type=int, help="The minimum number of streamlines to acquire", action="store")
    parser.add_argument("--append", help="Append to the existing file, rather than overwriting it", action="store_true")
	parser.add_argument("--quiet", help="Hide informational messages", action="store_true")

	args = parser.parse_args()

	if len(args.im) != len(args.sd):
		raise Exception("The number of --im and --sd flags do not match")
 

	def Track(loc_to, noTracks):
		gen.Run([args.track_command, "-select", str(noTracks), "-force", loc_to], printCommands=(not args.quiet))

	Run(loc_save_to_tck=args.save_to_tck, track_function=Track, loc_image=args.im, target_sd=args.sd, min_step=args.min_step, max_step=args.max_step, minimum_trackCount=args.minimum_track_count, append_if_file_exists=args.append, verbose=(not args.quiet))
