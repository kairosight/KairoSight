import os
from imageio import volread, get_reader
# import numpy as np


def open_single(source, meta=None):
    """Open images from a file containing one signal source or channel.

       Parameters
       ----------
       source : str
            The full path to the file
       meta : str, optional
            The full path to a file containing metadata

       Returns
       -------
       image : ndarray
            An array of normalized fluorescence data
       meta : dict
            A dict of metadata
       """
    # Check parameter types
    if type(source) not in [str]:
        raise TypeError('Required "source" ' + source + ' parameter must be a string')
    if meta and (type(meta) not in [str]):
        raise TypeError('Optional "meta" ' + meta + ' parameter must be a string')

    # Check validity
    # Make sure the source is an existing file
    if not os.path.isfile(source):
        raise FileNotFoundError('Required "source" ' + source + ' is not a file or does not exist.')
    if meta and not os.path.isfile(meta):
        raise FileNotFoundError('Optional "meta" ' + meta + ' is not a file or does not exist.')

    # Open the file
    # file_source = open(source, 'rb')
    # tags = exifread.process_file(file)  # Read EXIF data
    stack = volread(source)  # Read image data, keep this second because it closes the file after reading
    stack_meta = get_reader(source).get_meta_data()
    # Open the metadata, if provided
    if meta:
        file_meta = open(meta)
        meta = file_meta.read()
        file_meta.close()
    else:
        meta = stack_meta
    return stack, meta


def crop(stack_in,  d_x, d_y):
    """Crop a stack (3-D array, TYX) of grayscale optical data.

       Parameters
       ----------
       stack_in : ndarray
            A 3-D array (T, Y, X) of optical data
       d_x : int
            Amount of pixels to remove from the input's width.
            If < 0, crop from the maximal width
       d_y : int
            Amount of pixels to remove from the input's height.
            If < 0, crop from the maximal height

       Returns
       -------
       stack_out : ndarray
            A cropped 3-D array (T, Y, X) of optical data
       """
    pass


def mask_generate(stack_in, thresh_type='Otsu_global'):
    """Generate a mask for a stack (3-D array, TYX) of grayscale optical data
    using a binary threshold algorithm (histogram-based or local).

       Parameters
       ----------
       stack_in : ndarray
            A 3-D array (T, Y, X) of optical data
       thresh_type : str
            The type of thresholding algorithm to use

       Returns
       -------
       mask : ndarray
            A binary 2D array generated from the threshold algorithm
       """
    pass


def mask_apply(stack_in, mask):
    """Apply a binary mask to segment a stack (3-D array, TYX) of grayscale optical data.

       Parameters
       ----------
       stack_in : ndarray
            A 3-D array (T, Y, X) of optical data
       mask : ndarray
            A binary 2D array

       Returns
       -------
       stack_out : ndarray
            A masked 3-D array (T, Y, X) of optical data
       """
    pass
