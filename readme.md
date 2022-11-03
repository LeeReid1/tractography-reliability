Tractography Reliability
====================

These tools provide a means of generating reliable probabilistic tractography via an existing MRtrix installation. Specifically, these tools automatically perform probabilistic tractography, using MRtrix, until a tractogram becomes stable. This removes the need for researchers to choose the number of streamlines before tractography takes place, and provides confidence that tractograms have enough streamlines to be replicable.


**Please read the installation notes before cloning this repository.**


All instructions are supplied for a Linux environment. For help with Windows environments, please contact us.

Installation
------------


First install:

*  [Python 3](https://www.python.org/downloads/) or later, and place it in the system path.
*  [Numpy](https://numpy.org/) for Python 3
*  [MRtrix3](http://www.mrtrix.org/), whose bin directory must be in the system path. Alternatively, you can set an environment variable 'MRtrix3' pointing to its bin directory
*  [git lfs](https://www.atlassian.com/git/tutorials/git-lfs#installing-git-lfs) ***BEFORE*** cloning this directory


Then, after these have been installed:

```
cd ~/
git clone --recursive https://github.com/LeeReid1/tractography-reliability.git
```

This installs to your home directory, but any destination directory will work.

If you fail to supply the --recursive flag when cloning, you must initialise the sub-repository like so:

```
cd ~/tractography-reliability # where you cloned this repository
git submodule update --init --recursive
```



Example Usage
--------------

To use the provided scripts, provide them with your track command in double quotes. Do not include `-select` or save-to locations in this command.

Remember to think about your data before choosing reliability parameters. **Do not copy and paste parameters from the examples below without consideration** as they may not be suitable for your research question or data.

### Tractogram Bootstrapping:

To generate a tractogram suitable for conversion into a binary mask.
 
```
# Generate tractography
python ~/tractography-reliability/track_to_convergence_tckmap.py --resolution 2 --bint 0.001 --save_to_tck tractogram.tck  --target_dice 0.95 --track_command "tckgen fod_wm.mif -seed_image seed.nii.gz -include include.nii.gz -stop"

# The following steps convert the tractogram to an image and are optional. They may need to be modified to reflect your parameters and environment

# -- Get the streamline count
streamline_count=$(echo $(tckinfo tractogram.tck -quiet -count) | rev | cut -d' ' -f 1 | rev)

# -- Calculate binarisation threshold (bint * nstreamlines)
binarisation_threshold=$(bc -l <<< $streamline_count*0.001)

# -- Create binary mask
tckmap tractogram.tck -vox 2 - | mrcalc - $binarisation_threshold -gt trackmap.nii.gz
```

### Microstructural Sample Reliability

You should include every image that you intend to sample from in the tractography command, because some diffusion metrics require substantially more streamlines than others.

To generate a tractogram suitable for sampling from a single image:

```
 python ~/tractography-reliability/track_to_convergence_sample.py --save_to_tck tractogram.tck --im fa.nii.gz --sd 0.001 --track_command "tckgen fod_wm.mif -seed_image seed.nii.gz -include include.nii.gz -stop" 
```

To generate a tractogram suitable for sampling from multiple images:

```
 python ~/tractography-reliability/track_to_convergence_sample.py --save_to_tck tractogram.tck --im fa.nii.gz --sd 0.001 --im md.nii.gz --sd 0.000001 --track_command "tckgen fod_wm.mif -seed_image seed.nii.gz -include include.nii.gz -stop" 
```

Tips
--------------
1. The provided scripts contain help and commenting. Type `--help` with either to read the help documentation
2. These scripts are suitable only for probabilistic tractography generated with MRtrix
3. Some flags exist that are not part of the original paper and may help in niche situations
4. Lowering the reliability criteria (e.g. low dice scores or high standard deviations) to achieve faster tractography is lowering the reliability of your results. Think about what your criteria mean and always report them within your methods.
5. Trackmap Bootstrapping criteria with Dice Coefficients >0.95 or with resolutions higher than your FOD image resolution will drastically increase the number of streamlines you require. 
6. Do not include `-select` or save-to locations in your commands.
7. Executables in the LIL directory are not intended to be used directly: these contain some options that are not formally tested, documented, or supported - please use the supplied python scripts.

How to report your use and cite
-------------------------------

### Citation:
Please cite:
Lee B. Reid, Marcela I. Cespedes, Kerstin Pannek (2020). How many streamlines are required for reliable probabilistic tractography? Solutions for microstructural measurements and neurosurgical planning. _NeuroImage_ (211). doi: https://doi.org/10.1016/j.neuroimage.2020.116646

### Tractogram Bootstrapping:
Suggested example:

>Streamline generation was performed until Tractogram Bootstrapping [cite] stability criteria were met (min Dice, 0.95; reliability, 0.95; 1mm isotropic; b<sub>int</sub>, 0.001×n).

Alternative:

> Tractogram Bootstrapping [cite] stability criteria were set at 0.95, producing a 1mm isotropic track density image that was thresholded at 0.001 × streamline count and binarised.

### Microstructural Sample Reliability
Suggested example:

>Streamline generation was performed until Microstructural Sample Reliability [cite] stability criteria were met (SD 0.001).
