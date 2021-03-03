# -*- coding: utf-8 -*-
"""
Author: Ty Nietupski (ty.nietupski@oregonstate.edu)

Functions to perform some additional preprocessing and image prep:
    - co-registration (registerImages)
    - threshold calculation and threshold based masking (Thresh, ThreshMask)
    - landsat and MODIS format preparation (prepLandsat, prepMODIS)
    - some functions used within these functions

These functions should be run after those found in get_paired_collections.py
and before core_functions.py. An example of the use of these functions can
be found in Predict_l8.py.


The MIT License

Copyright © 2021 Ty Nietupski

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import ee


def registerImages(landsat_t01, modis_t01, modis_tp):
    """
    Register each image to the earliest (t0) landsat image in the set of pairs.

    Parameters
    ----------
    landsat_t01 : ee_list.List
        Landsat images, earliest (t0) and latest (t1).
    modis_t01 : ee_list.List
        MODIS images, earliest (t0) and latest (t1).
    modis_tp : ee_list.List
        MODIS images, evey image between t0 and t1.

    Returns
    -------
    landsat_t01 : ee_list.List
        Resampled version of input.
    modis_t01 : ee_list.List
        Resampled, registered version of input (registered to Landsat t0).
    modis_tp : ee_list.List
        Resampled, registered version of input (registered to Landsat t0).

    """
    # resample
    landsat_t01 = landsat_t01.map(lambda image:
                                  ee.Image(image).resample('bicubic'))

    modis_t01 = modis_t01.map(lambda image:
                              ee.Image(image).resample('bicubic'))

    modis_tp = modis_tp.map(lambda image:
                            ee.Image(image).resample('bicubic'))

    # register MODIS to landsat t0
    modis_t01 = modis_t01.map(lambda image:
                              ee.Image(image)\
                              .register(referenceImage=
                                        ee.Image(landsat_t01.get(0)),
                              maxOffset=ee.Number(150.0),
                              stiffness=ee.Number(7.0)))

    modis_tp = modis_tp.map(lambda image:
                            ee.Image(image)\
                            .register(referenceImage=
                                      ee.Image(landsat_t01.get(0)),
                            maxOffset=ee.Number(150.0),
                            stiffness=ee.Number(7.0)))

    return landsat_t01, modis_t01, modis_tp


def threshold(landsat, coverClasses):
    """
    Determine similarity threshold for each landsat image based on the number\
    of cover classes.

    Parameters
    ----------
    landsat : ee_list.List
        List of landsat images to determine threshold.
    coverClasses : int
        Number of cover classes in region.

    Returns
    -------
    ee_list.List
        ee_list.List


    """
    def getThresh(image):
        """
        Local function to determine thresholds for each band of the landsat\
        image.

        Parameters
        ----------
        image : image.Image
            Landsat image.

        Returns
        -------
        thresh : image.Image
            Image with constant bands for the threshold associated with each\
            band.

        """
        # calculate the standard deviation for each band within image
        stddev = ee.Image(image).reduceRegion(reducer=ee.Reducer.stdDev(),
                                              bestEffort=True,
                                              maxPixels=ee.Number(1e6))
        # convert stddev dictionary to multiband image
        stddev = stddev.toImage()

        # get our band names from the image and rename for threshold
        names = stddev.bandNames() \
            .map(lambda bn: ee.String(bn).cat('_thresh'))

        # calculate the threshold from stddev and number of landcover classes
        thresh = stddev.multiply(ee.Image.constant(ee.Number(2)) \
                                 .divide(coverClasses))
        thresh = thresh.rename(names)

        return thresh

    threshs = ee.List(landsat).map(getThresh)

    return threshs


def threshMask(neighLandsat_t01, thresh, commonBandNames):
    """
    Use thresholds calculated with the threshold function to mask pixels in\
    the neighborhood images that are dissimilar.

    Parameters
    ----------
    neighLandsat_t01 : ee_list.List
        List format of the neighborhood landsat images at t0 and t1.
    thresh : ee_list.List
        List of the threshold images for t0 and t1.
    commonBandNames : ee_list.List
        List of the band names to use in selecting bands.

    Returns
    -------
    masks : ee_list.List
        Landsat neighborhood images with dissimilar pixels masked.

    """
    masks = ee.List([0, 1]) \
        .map(lambda i:
             commonBandNames \
             .map(lambda name:
                  # get t0 or t1 neighborhood image and calc distance from
                  # central pixel, if over threshold, mask the pixel
                  ee.Image(neighLandsat_t01.get(i)) \
                      .select([ee.String(name).cat('_(.+)')]) \
                      .select([ee.String(name).cat('_0_0')]) \
                  .subtract(ee.Image(neighLandsat_t01.get(i)) \
                                .select([ee.String(name).cat('_(.+)')])) \
                  .abs() \
                  .lte(ee.Image(thresh.get(i)) \
                       .select([ee.String(name).cat('_(.+)')]))))

    return masks


def prepMODIS(modis_t01, modis_tp, kernel,
              numPixels, commonBandNames, pixelBandNames):
    """
    Convert MODIS images to neighborhood images and reorganize so that they\
    are in a format that will work with the 'core functions'

    Parameters
    ----------
    modis_t01 : ee_list.List
        MODIS images at time 0 (t0) and time 1 (t1).
    modis_tp : ee_list.List
        MODIS images between earliest and latest pairs.
    kernel : Kernel
        Kernel object used to create neighborhood image (window).
    numPixels : ee_number.Number
        Total number of pixels in the kernel.
    commonBandNames : ee_list.List
        Names of bands to use in fusion.
    pixelBandNames : ee_list.List
        Names of bands that will be generated when the images are converted\
        to neighborhood images.

    Returns
    -------
    modSorted_t01 : ee_list.List
        The converted, reorganized neighborhood images for t0 and t1.
    modSorted_tp : ee_list.List
        The converted, reorganized neighborhood images between t0 and t1.

    """
    # convert images to neighborhood images
    neighMod_t01 = modis_t01 \
        .map(lambda image: ee.Image(image).neighborhoodToBands(kernel))
    neighMod_tp = modis_tp \
        .map(lambda image: ee.Image(image).neighborhoodToBands(kernel))

    # convert into an array image
    modArr = ee.Image(neighMod_t01.get(0)).toArray() \
        .arrayCat(ee.Image(neighMod_t01.get(1)).toArray(), 0)

    # create list of arrays sliced by pixel position
    modPixArrays_t01 = ee.List.sequence(0, numPixels.subtract(1)) \
        .map(lambda i:
             modArr.arraySlice(0,
                               ee.Number(i).int(),
                               numPixels.multiply(commonBandNames.length()\
                                                  .multiply(2)).int(),
                               numPixels))

    modPixArrays_tp = neighMod_tp \
        .map(lambda image:
             ee.List.sequence(0, numPixels.subtract(1)) \
                 .map(lambda i:
                      ee.Image(image) \
                      .toArray() \
                      .arraySlice(0,
                                  ee.Number(i).int(),
                                  numPixels.multiply(commonBandNames\
                                                     .length()).int(),
                                  numPixels)))

    # flatten arrays and name based on doy, band, and pixel position
    modSorted_t01 = ee.List.sequence(0, numPixels.subtract(1)) \
        .map(lambda i:
             ee.Image(modPixArrays_t01.get(i)) \
             .arrayFlatten([pixelBandNames.get(i)]))

    modSorted_tp = ee.List.sequence(0, modPixArrays_tp.length().subtract(1)) \
        .map(lambda i:
             ee.List.sequence(0, numPixels.subtract(1)) \
             .map(lambda x:
                  ee.Image(ee.List(modPixArrays_tp.get(i)).get(x)) \
                  .arrayFlatten([commonBandNames]) \
                  .set('DOY', ee.Image(modis_tp.get(i)).get('DOY'))))

    return modSorted_t01, modSorted_tp


def prepLandsat(landsat_t01, kernel,
                numPixels, commonBandNames,
                doys, coverClasses):
    """
    Convert Landsat images to neighborhood images, mask dissimilar pixels,\
    and reorganize so that they are in a format that will work with the\
    'core functions'.

    Parameters
    ----------
    landsat_t01 : ee_list.List
        Landsat images at time 0 (t0) and time 1 (t1).
    kernel : Kernel
        Kernel object used to create neighborhood image (window)..
    numPixels : ee_number.Number
        Total number of pixels in the kernel.
    commonBandNames : ee_list.List
        Names of bands to use in fusion.
    doys : ee_list.List
        List of day of year associated with t0 and t1.
    coverClasses : int
        Number of cover classes in region.

    Returns
    -------
    maskedLandsat : ee_list.List
        The converted, masked, reorganized neighborhood images for t0 and t1.
    pixPositions : ee_list.List
        Names for each index in the window (e.g., "_0_0", "_0_1" ...).
    pixelBandNames : ee_list.List
        Names of bands that will be generated when the images are converted\
        to neighborhood images.

    """
    # convert images to neighborhood images
    neighLandsat_t01 = landsat_t01 \
        .map(lambda image: ee.Image(image).neighborhoodToBands(kernel))

    # create list of pixel postions
    pixPositions = ee.Image(neighLandsat_t01.get(0)).bandNames() \
        .map(lambda bn: ee.String(bn).replace('[a-z]+_', '_')) \
        .slice(0, numPixels)

    # create list of band names to rename output arrays
    pixelBandNames = pixPositions \
        .map(lambda position:
             doys.map(lambda doy:
                      commonBandNames.map(lambda bn:
                                          ee.String(doy).cat('_') \
                                          .cat(ee.String(bn)) \
                                          .cat(ee.String(position))))) \
        .map(lambda l: ee.List(l).flatten())

    # convert to array and sort
    # essentially we would have the array values stacked with
    # time 0 on top of time 1
    # ie 1 column of values for pix positions at both times
    lanArr_t01 = ee.Image(neighLandsat_t01.get(0)).toArray() \
        .arrayCat(ee.Image(neighLandsat_t01.get(1)).toArray(), 0)

    pixArrays = ee.List([])
    pixArrays = ee.List.sequence(0, numPixels.subtract(1)) \
        .map(lambda i: lanArr_t01 \
             .arraySlice(0,
                         ee.Number(i).int(),
                         numPixels.multiply(commonBandNames \
                                            .length() \
                                            .multiply(2)).int(),
                         numPixels))

    # flatten arrays and name based on doy, band, and pixel position
    lanSorted = ee.List.sequence(0, numPixels.subtract(1))\
        .map(lambda i:
             ee.Image(pixArrays.get(i)).arrayFlatten([pixelBandNames.get(i)]))

    # determine threshold for images
    thresh = threshold(landsat_t01, coverClasses)

    # mask window images with thresholds
    mask_t01 = threshMask(neighLandsat_t01, thresh, commonBandNames)

    # convert list of masks of lists of masks to image and then to array
    maskArr_t01 = ee.ImageCollection(mask_t01.flatten()) \
        .toBands() \
        .toArray()

    maskArrays = ee.List.sequence(0, numPixels.subtract(1))\
        .map(lambda i:
             maskArr_t01 \
             .arraySlice(0, ee.Number(i).int(),
                         numPixels.multiply(commonBandNames.length()\
                                            .multiply(2)).int(),
                         numPixels))

    # flatten mask arrays and name based on doy, band, and pixel position
    masksSorted = ee.List.sequence(0, numPixels.subtract(1))\
        .map(lambda i:
             ee.Image(maskArrays.get(i)).arrayFlatten([pixelBandNames.get(i)]))

    # mask landsat images
    maskedLandsat = ee.List.sequence(0, numPixels.subtract(1))\
        .map(lambda index:
             ee.Image(lanSorted.get(index)) \
             .updateMask(ee.Image(masksSorted.get(index))))

    return maskedLandsat, pixPositions, pixelBandNames
