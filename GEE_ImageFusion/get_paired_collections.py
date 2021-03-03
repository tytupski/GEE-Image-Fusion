# -*- coding: utf-8 -*-
"""
Author: Ty Nietupski (ty.nietupski@oregonstate.edu)

Functions for image preprocessing and data organization:
    - mask landsat
    - mask modis
    - addNDVI
    - etmToOli
    - get paired image collections (getPaired)
    - reorganize paired collection to units for prediction (makeSubCollections)

This script contains the functions used to acquire, preprocess, and organize
all of the Landsat and MODIS images over a given period of time. These
functions should be run first, after defining some global variables. Functions
in prep_functions.py and core_functions.py follow these. An example of
the use of these functions can be found in Predict_L8.py.


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

##############################################################################
# MASKING, INDEX CALCULATION, L5 & L7 TO L8 HARMONIZATION
##############################################################################


def maskLandsat(image):
    """
    Mask cloud, shadow, and snow with fmask and append the percent of pixels \
    masked as new image property.

    Parameters
    ----------
    image : image.Image
        Landsat image with qa band.

    Returns
    -------
    image.image
        Masked landsat image with CloudSnowMaskedPercent property.

    """
    # Bits 3 and 5 are cloud shadow and cloud, respectively. 4 is snow
    cloudShadowBitMask = 1 << 3
    cloudsBitMask = 1 << 5
    snowBitMask = 1 << 4

    # Get the pixel QA band.
    qa = image.select('pixel_qa')

    # make mask
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) \
        .And(qa.bitwiseAnd(cloudsBitMask).eq(0)) \
        .And(qa.bitwiseAnd(snowBitMask).eq(0))

    # mask the mask with the mask...
    maskedMask = mask.updateMask(mask)

    # count the number of nonMasked pixels
    maskedCount = maskedMask.select(['pixel_qa']) \
        .reduceRegion(reducer=ee.Reducer.count(),
                      geometry=image.geometry(),
                      scale=ee.Number(30),
                      maxPixels=ee.Number(4e10))

    # count the total number of pixels
    origCount = image.select(['blue']) \
        .reduceRegion(reducer=ee.Reducer.count(),
                      geometry=image.geometry(),
                      scale=ee.Number(30),
                      maxPixels=ee.Number(4e10))

    # calculate the percent of masked pixels
    percent = ee.Number(origCount.get('blue')) \
        .subtract(maskedCount.get('pixel_qa')) \
        .divide(origCount.get('blue')) \
        .multiply(100) \
        .round()

    # Return the masked image with new property and time stamp
    return image.updateMask(mask) \
        .set('CloudSnowMaskedPercent', percent) \
        .copyProperties(image, ["system:time_start"])


def maskMODIS(image):
    """
    Mask snow covered and extremely high albedo areas from the modis images.

    Parameters
    ----------
    image : image.Image
        MODIS image.

    Returns
    -------
    image.image
        Masked MODIS image.

    """
    # calculate snow water index for the image
    swi = image.expression(
        '(green * (nir - swir1)) / ((green + nir) * (nir + swir1))',
        {'green': image.select(['green']),
         'nir': image.select(['nir']),
         'swir1': image.select(['swir1'])
         }).rename('swi')

    # mask out values of swi above 0.1
    mask = swi.lt(0.1)

    return image \
        .updateMask(mask) \
        .copyProperties(image, ['system:time_start', 'system:id'])


def addNDVI(image):
    """
    Mask snow covered and extremely high albedo areas from the modis images.

    Parameters
    ----------
    image : image.Image
        Landsat or MODIS image with bands named 'nir' and 'red'.

    Returns
    -------
    image.image
        Image with additional NDVI band.
    """
    # calculate NDVI
    ndvi = image.normalizedDifference(['nir', 'red']).select(['nd'], ['ndvi'])

    return image.addBands(ndvi)


def etmToOli(img):
    """
    Calibrate the NDVI values so that they are more similar to OLI NDVI.

    Parameters
    ----------
    img : image.Image
        Landsat 5 or 7 image.

    Returns
    -------
    image.image
        Adjusted Landsat image.

    """
    # coefficients from Roy et al. 2016
    coefficients = {'beta_0': ee.Image.constant([0.0235]),
                    'beta_1': ee.Image.constant([0.9723])}

    return img \
        .multiply(coefficients['beta_1']) \
        .add(coefficients['beta_0']) \
        .toFloat() \
        .copyProperties(img, ["system:time_start", 'system:id', 'DOY'])


##############################################################################
# FILTER AND PAIR IMAGES
##############################################################################


def getPaired(startDate, endDate,
              landsatCollection, landsatBands, bandNamesLandsat,
              modisCollection, modisBands, bandNamesModis,
              commonBandNames,
              region):
    """
    Create a list of image collections. Landsat and MODIS with low cloud cover\
    from the same date and the MODIS images between these pairs.

    Parameters
    ----------
    startDate: str
        Start date of fusion timeframe.
    endDate: str
        End date of the fusion timeframe.
    landsatCollection: str
        Landsat collection https://developers.google.com/earth-engine/datasets
    landsatBands: ee_list.List
        List of integers corresponding to Landsat bands.
    bandNamesLandsat: ee_list.List
        List of strings used to rename bands.
    modisCollection: str
        MODIS collection https://developers.google.com/earth-engine/datasets
    modisBands: ee_list.List
        List of integers corresponding to MODIS bands in same order as Landsat.
    bandNamesModis: ee_list.List
        List of strings used to rename bands.
    commonBandNames: ee_list.List
        List of bands to use in fusion. Common to both Landsat and MODIS.
    region: geometry.Geometry
        Location to use in filtering collections. Must not be in scene overlap.

    Returns
    -------
    python list obejct
        Each element in this list is an ee.ImageCollection. The first and \
        second elements are the Landsat occuring on the same date and \
        the last element is the MODIS images between each of the pair \
        dates.

    """
    if landsatCollection == 'LANDSAT/LC08/C01/T1_SR':
        # get landsat images
        landsat = ee.ImageCollection(landsatCollection) \
                    .filterDate(startDate, endDate) \
                    .filterBounds(region) \
                    .filterMetadata('CLOUD_COVER', 'less_than', 5) \
                    .select(landsatBands, bandNamesLandsat) \
                    .map(addNDVI) \
                    .map(maskLandsat) \
                    .filterMetadata('CloudSnowMaskedPercent', 'less_than', 50)\
                    .map(lambda image: image \
                         .setMulti({
                             'system:time_start':
                                 ee.Date(image.date().format('y-M-d')) \
                                 .millis(),
                             'DOY': image.date().format('D')
                             })) \
                    .select(commonBandNames)
    else:
        # get landsat images
        landsat = ee.ImageCollection(landsatCollection) \
                    .filterDate(startDate, endDate) \
                    .filterBounds(region) \
                    .filterMetadata('CLOUD_COVER', 'less_than', 5) \
                    .select(landsatBands, bandNamesLandsat) \
                    .map(addNDVI) \
                    .map(maskLandsat) \
                    .filterMetadata('CloudSnowMaskedPercent', 'less_than', 50)\
                    .map(lambda image: image \
                         .setMulti({
                             'system:time_start':
                                 ee.Date(image.date().format('y-M-d')) \
                                 .millis(),
                             'DOY': image.date().format('D')
                             })) \
                    .select(commonBandNames) \
                    .map(etmToOli)

    # get modis images
    modis = ee.ImageCollection(modisCollection) \
              .filterDate(startDate, endDate) \
              .select(modisBands, bandNamesModis) \
              .map(addNDVI) \
              .map(maskMODIS) \
              .map(lambda image: image.set('DOY', image.date().format('D'))) \
              .select(commonBandNames)

    # filter the two collections by the date property
    dayfilter = ee.Filter.equals(leftField='system:time_start',
                                 rightField='system:time_start')

    # define simple join
    pairedJoin = ee.Join.simple()
    # define inverted join to find modis images without landsat pair
    invertedJoin = ee.Join.inverted()

    # create collections of paired landsat and modis images
    landsatPaired = pairedJoin.apply(landsat, modis, dayfilter)
    modisPaired = pairedJoin.apply(modis, landsat, dayfilter)
    modisUnpaired = invertedJoin.apply(modis, landsat, dayfilter)

    return [landsatPaired, modisPaired, modisUnpaired]


##############################################################################
# CREATE SUBCOLLECTIONS FOR EACH SET OF LANDSAT/MODIS PAIRS
##############################################################################


def getDates(image, empty_list):
    """
    Get date from image and append to list.

    Parameters
    ----------
    image : image.Image
        Any earth engine image.
    empty_list : ee_list.List
        Earth engine list object to append date to.

    Returns
    -------
    updatelist : ee_list.List
        List with date appended to the end.

    """
    # get date and update format
    date = ee.Image(image).date().format('yyyy-MM-dd')

    # add date to 'empty list'
    updatelist = ee.List(empty_list).add(date)

    return updatelist


def makeSubcollections(paired):
    """
    Reorganize the list of collections into a list of lists of lists. Each\
    list within the list will contain 3 lists. The first of these three will\
    have the earliest and latest Landsat images. The second list will have the\
    earliest and latest MODIS images. The third list will have all the MODIS\
    images between the earliest and latest pairs.\
    (e.g. L8 on 05/22/2017 & 06/23/2017, MOD 05/23/2017 & 06/23/2017,\
     MOD 05/23/2017 through 06/22/2017).

    Parameters
    ----------
    paired : python List
        List of image collections. 1. Landsat pairs, 2. MODIS pairs, and\
        3. MODIS between each of the pairs.

    Returns
    -------
    ee_list.List
        List of lists of lists.

    """
    def getSub(ind):
        """
        Local function to create individual subcollection.

        Parameters
        ----------
        ind : int
            Element of the list to grab.

        Returns
        -------
        ee_list.List
            List of pairs lists for prediction 2 pairs and images between.

        """
        # get landsat images
        lan_01 = paired[0] \
            .filterDate(ee.List(dateList).get(ind),
                        ee.Date(ee.List(dateList).get(ee.Number(ind).add(1)))\
                            .advance(1, 'day')) \
            .toList(2)
        # get modis paired images
        mod_01 = paired[1] \
            .filterDate(ee.List(dateList).get(ind),
                        ee.Date(ee.List(dateList).get(ee.Number(ind).add(1)))\
                            .advance(1, 'day')) \
            .toList(2)
        # get modis images between these two dates
        mod_p = paired[2] \
            .filterDate(ee.List(dateList).get(ind),
                        ee.Date(ee.List(dateList).get(ee.Number(ind).add(1)))\
                            .advance(1, 'day'))

        mod_p = mod_p.toList(mod_p.size())

        # combine collections to one object
        subcollection = ee.List([lan_01, mod_01, mod_p])

        return subcollection

    # empty list to store dates
    empty_list = ee.List([])

    # fill empty list with dates
    dateList = paired[0].iterate(getDates, empty_list)

    # filter out sub collections from paired and unpaired collections
    subcols = ee.List.sequence(0, ee.List(dateList).length().subtract(2))\
        .map(getSub)

    return subcols
