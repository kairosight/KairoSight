import os
from math import floor
from imageio import volread, get_reader
import numpy as np


def open_signal(source, fps=500):
    """Open an array of optical data from a text file (.csv)
    as a calculated time array and an array of 16-bit arbitrary fluorescent data

        Parameters
        ----------
        source : str
            The full path to the file
        fps : int
            The framerate of the recorded captured

        Returns
        -------
        signal_time : ndarray
            An array of timestamps (ms) corresponding to the model_data
        signal_data : ndarray
            An array of normalized fluorescence data
    """
    # Check parameter types
    if type(source) not in [str]:
        raise TypeError('Required "source" ' + source + ' parameter must be a string')
    # Check parameter validity
    # Make sure the source is an existing file
    if not os.path.isfile(source):
        raise FileNotFoundError('Required "source" ' + source + ' is not a file or does not exist.')

    # Load the text file
    signal_text = np.genfromtxt(source, delimiter=',')

    # Calculate important constants
    FPMS = fps / 1000
    FRAME_T = 1 / FPMS

    if len(signal_text.shape) > 1:
        # Multiple columns
        data_x = signal_text[1:, 0]     # rows of the first column (skip X,Y header row)
        data_y_counts = signal_text[1:, 1].astype(np.uint16)  # rows of the first column (skip X,Y header row)
        FRAMES = len(data_x)
        FINAL_T = floor(FPMS * FRAMES)
        signal_data = data_y_counts
    else:
        signal_data = signal_text[0]
    # Generate array of timestamps
    signal_time = np.linspace(start=0, stop=FINAL_T, num=FRAMES)

    return signal_time, signal_data


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
