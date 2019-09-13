Tracking Reliability
====================

These tools provide a means of generating probabilistic tractography via an existing MRtrix installation. 


**Please read the installation notes before cloning this repository.**


All instructions are supplied for a Linux environment. For help with Windows environments, please contact us.

Installation
------------


First install:

*  [Python 3](https://www.python.org/downloads/) or later, and place it in the system path.
*  [MRtrix3](http://www.mrtrix.org/)
*  [git lfs](https://www.atlassian.com/git/tutorials/git-lfs#installing-git-lfs) ***BEFORE*** cloning this directory

Once these have been installed:

```
cd ~/any_directory/
git clone https://rei19q@bitbucket.csiro.au/scm/~rei19q/tractography-reliability.git
```


Example Usage
--------------

To use the provided scripts, provide them with your `track_command` in double quotes. Do not include `-select` or save-to locations in this command.

Remember to think about your data before choosing reliability parameters. **Do not copy and paste parameters from the examples below** as they may not be suitable for your research question or data.

### Tractogram Bootstrapping:

To generate a tractogram suitable for conversion into a binary mask.
 
```
# Generate tractography
python ~/any_directory/track_to_convergence_tckmap.py --resolution 2 --bint 0.001 --save_to_tck tractogram.tck  --target_dice 0.95 --track_command "tckgen fod_wm.mif -seed_image seed.nii.gz -include include.nii.gz -stop"

# The following steps convert the tractogram to an image and are optional. They may need to be modified to reflect your parameters and environment

# -- Get the streamline count
streamline_count=$(echo $(tckinfo tractogram.tck -quiet -count) | rev | cut -d' ' -f 1 | rev)

# -- Calculate binarisation threshold (bint * nstreamlines)
binarisation_threshold=$(bc -l <<< $streamline_count*0.001)

# -- Create binary mask
tckmap tractogram.tck -vox 2 - | mrcalc - $binarisation_threshold -gt trackmap.nii.gz
```

### Microstructural Sample Reliability

To generate a tractogram suitable for sampling from an image.

```
 python ~/any_directory/track_to_convergence_sample.py --save_to_tck tractogram.tck --im fa.nii.gz --sd 0.001 --track_command "tckgen fod_wm.mif -seed_image seed.nii.gz -include include.nii.gz -stop" 
```

Tips
--------------
1. The provided scripts contain help and commenting. Type `--help` with either to read the help documentation
2. These scripts are suitable only for probabilistic tractography generated with MRtrix
3. Some flags exist that are not part of the original paper and may help in niche situations
4. Lowering the reliability criteria (e.g. low dice scores or high standard deviations) to achieve faster tractography is lowering the reliability of your results. Think about what your criteria mean and always report them within your methods.
5. Trackmap Bootstrapping criteria beyond with Dice Coefficients >0.95 or with resolutions higher than your FOD image resolution will drastically increase the number of streamlines you require. 
6. Do not include `-select` or save-to locations in your commands.

How to report your use and cite
-------------------------------

_This work is currently undergoing peer-review._

### Tractogram Bootstrapping:
Suggested example:

>Streamline generation was performed until Tractogram Bootstrapping [cite] stability criteria were met (min Dice, 0.95; reliability, 0.95; 1mm isotropic; b<sub>int</sub>, 0.001×n).

Alternative:

> Tractogram Bootstrapping [cite] stability criteria were set at 0.95, producing a 1mm isotropic track density image that was thresholded at 0.001 × streamline count and binarised.

### Microstructural Sample Reliability
Suggested example:

>Streamline generation was performed until Microstructural Sample Reliability [cite] stability criteria were met (SD 0.001).
