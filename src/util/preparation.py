import os
from math import floor
import numpy as np
from imageio import volread, get_reader
from skimage.util import img_as_uint


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
    else:
        # Single column, data only
        data_y_counts = signal_text[0]
        FRAMES = len(data_y_counts)
        FINAL_T = floor(FPMS * FRAMES)

    signal_data = data_y_counts
    # Generate array of timestamps
    signal_time = np.linspace(start=0, stop=FINAL_T, num=FRAMES)

    return signal_time, signal_data


def open_stack(source, meta=None):
    """Open a stack of images from a file containing one signal source or channel.

       Parameters
       ----------
       source : str
            The full path to the file
       meta : str, optional
            The full path to a file containing metadata

       Returns
       -------
       stack : ndarray
            A 3-D array (T, Y, X) of optical data, 16-bit
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

    # Open the metadata, if provided
    stack_meta = get_reader(source).get_meta_data()
    # Open the file
    # file_source = open(source, 'rb')
    # tags = exifread.process_file(file)  # Read EXIF data
    stack_data = np.array(volread(source))  # Read image data, closes the file after reading
    stack = img_as_uint(stack_data)  # Read image data, closes the file after reading
    if meta:
        file_meta = open(meta)
        meta = file_meta.read()
        file_meta.close()
    else:
        meta = stack_meta
    return stack, meta


def crop(stack_in,  d_x, d_y):
    """Crop a stack (3-D array, TYX) of optical data.

       Parameters
       ----------
       stack_in : ndarray
            A 3-D array (T, Y, X) of optical data
       d_x : int
            Amount of pixels to remove from the input's width.
            < 0 to crop from the left, > 0 to crop from the right
       d_y : int
            Amount of pixels to remove from the input's height.
            < 0 to crop from the top, > 0 to crop from the bottom

       Returns
       -------
       stack_out : ndarray
            A cropped 3-D array (T, Y, X) of optical data, dtype : stack_in.dtype
       """
    # Check parameters
    if type(stack_in) is not np.ndarray:
        raise TypeError('Stack type must be an "ndarray"')
    if len(stack_in.shape) is not 3:
        raise TypeError('Stack must be a 3-D ndarray (T, X, Y)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')
    if type(d_x) is not int:
        raise TypeError('X pixels to crop must be an "int"')
    if type(d_y) is not int:
        raise TypeError('Y pixels to crop must be an "int"')

    stack_out = np.empty_like(stack_in)

    if (d_x > 0) and (d_y > 0): 
        stack_out = stack_in[:, 0:-d_y, 0:-d_x]
    else:
        if d_y > 0:
            stack_out = stack_in[:, 0:-d_y:, -d_x:]
        if d_x > 0:
            stack_out = stack_in[:, -d_y:, 0:-d_x]

    return stack_out


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
