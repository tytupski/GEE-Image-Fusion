## Overview
There are a total of four scripts here that can be used to automate large image fusion tasks in GEE. The 3 scripts in the GEE_ImageFusion folder execute different parts of the GEE image fusion process. If using anaconda as your package manager, you can create a virtual environment for gee and drop the GEE_ImageFusion folder into the site-packages directory for that environment (example environment: test_fusion.yml). From there, the module can be imported into a script as any other module in that virtual environment would be. Thefour scripts within the module handle different aspects of the image fusion process. It is reccommended that you examine the Predict_L8 script first to get an idea of a possible workflow on how to use each of the functions in the module. If only interested in using functions from one of the submodules, these can be individually imported (e.g., from GEE_ImageFusion import core_functions). Each script is described below. 

## File descriptions
1. **Predict_L8.py**- this script shows an example workflow of using the functions in the GEE_ImageFusion module to automate a large image fusion task.

2. **get_paired_collections.py**- this script contains functions to retrieve, filter, mask, sort, and organize the Landsat and MODIS data.

3. **prep_functions.py**- this script contains functions to preprocess the Landsat and MODIS imagery. It has functions to perform a co-registration step, determine and mask similar pixels, and convert images to 'neighborhood' images with bands that are sorted in the necessary order for the core functions.

4. **core_functions.py**- this script contains the main functions needed to perform image fusion. If all images have been preprocessed and formatted correctly then these functions can be run to predict new images at times when only MODIS is available. Functions include spectral distance to similar pixels, spatial distance, weight calculation, conversion coefficient calculation, and prediction.

## Example time series MODIS vs. GEE image fusion 
[![MODIS vs. GEE image fusion](https://img.youtube.com/vi/v9F71tuqozY/maxresdefault.jpg)](https://www.youtube.com/watch?v=v9F71tuqozY)

## License
MIT
