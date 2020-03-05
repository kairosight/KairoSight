import os
import time
from memory_profiler import profile
from math import floor
import numpy as np
from pathlib import Path, PurePath
from imageio import volread, volwrite, get_reader
from scipy import signal
from skimage.util import img_as_uint, img_as_float
from skimage.filters import sobel, rank, threshold_otsu, threshold_mean, threshold_minimum
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, random_walker
from skimage.segmentation import mark_boundaries
from skimage.transform import rescale
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops
from skimage.morphology import disk
from skimage import measure
import cv2

# Constants
FL_16BIT_MAX = 2 ** 16 - 1  # Maximum intensity value of a 16-bit pixel: 65535
MASK_TYPES = ['Otsu_global', 'Mean', 'Random_walk', 'best_ever']


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
    # Generate array of timestamps
    FPMS = fps / 1000

    if len(signal_text.shape) > 1:
        # Multiple columns
        data_x = signal_text[1:, 0]  # rows of the first column (skip X,Y header row)
        data_y_counts = signal_text[1:, 1].astype(np.uint16)  # rows of the first column (skip X,Y header row)
        FRAMES = len(data_x)
        FINAL_T = floor(FRAMES / FPMS)
    else:
        # Single column, data only
        data_y_counts = signal_text[0]
        FRAMES = len(data_y_counts)
        FINAL_T = floor(FRAMES / FPMS)

    signal_data = data_y_counts
    # Generate array of timestamps
    signal_time = np.linspace(start=0, stop=FINAL_T, num=FRAMES)

    return signal_time, signal_data


def open_stack(source, meta=None):
    """Open a stack of images (.tif, .tiff, .pcoraw) from a file.

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

        Notes
        -----
            Files with .pcoraw extension are converted and saved as .tif.
            Expecting volume dimension order XYCZT
       """
    # Check parameter types
    if type(source) not in [str]:
        raise TypeError('Required "source" ' + source + ' parameter must be a string')
    if meta and (type(meta) not in [str]):
        raise TypeError('Optional "meta" ' + meta + ' parameter must be a string')

    # Check validity
    # Make sure the directory, source file, and optional meta file exists

    if not os.path.isdir(os.path.split(source)[0]):
        raise FileNotFoundError('Required directory ' + os.path.split(source)[0]
                                + ' is not a directory or does not exist.')
    if not os.path.isfile(source):
        raise FileNotFoundError('Required "source" ' + source + ' is not a file or does not exist.')
    if meta and not os.path.isfile(meta):
        raise FileNotFoundError('Optional "meta" ' + meta + ' is not a file or does not exist.')

    # If a .pcoraw file, convert to .tiff
    f_purepath = PurePath(source)
    f_extension = f_purepath.suffix
    if f_extension == '.pcoraw':
        p = Path(source)
        p.rename(p.with_suffix('.tif'))
        source = os.path.splitext(source)[0] + '.tif'
        print('* .pcoraw covnerted to a .tif')

    # Open the metadata, if provided
    stack_meta = get_reader(source, mode='v').get_meta_data()

    # Open the file
    # file_source = open(source, 'rb')
    # tags = exifread.process_file(file)  # Read EXIF data
    stack = img_as_uint((volread(source)))  # Read image data, closes the file after reading

    if meta:
        file_meta = open(meta)
        meta = file_meta.read()
        file_meta.close()
    else:
        meta = stack_meta
    return stack, meta


def crop_frame(frame_in, d_x, d_y):
    frame_out = frame_in.copy()

    if (d_x > 0) and (d_y > 0):
        frame_out = frame_out[0:-d_y, 0:-d_x]
    else:
        if d_x < 0:
            frame_out = frame_out[:, -d_x:]
        if d_y < 0:
            frame_out = frame_out[-d_y:, :]

    return frame_out


def crop_stack(stack_in, d_x=False, d_y=False):
    """Crop a stack (3-D array, TYX) of optical data,
    by default removes from right and bottom.

       Parameters
       ----------
       stack_in : ndarray
            A 3-D array (T, Y, X) of optical data, dtype : uint16 or float
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
        raise TypeError('Stack must be a 3-D ndarray (T, Y, X)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')
    # if type(d_x) is not int:
    #     raise TypeError('X pixels to crop must be an "int"')
    # if type(d_y) is not int:
    #     raise TypeError('Y pixels to crop must be an "int"')

    # stack_out = stack_in.copy()
    # if either X or Y crop is unused, set to 0
    if d_x is False:
        d_x = 0
    if d_y is False:
        d_y = 0

    if (d_x > 0) and (d_y > 0):
        stack_out = stack_in[:, 0:-d_y, 0:-d_x]
    else:
        if d_x < 0:
            stack_out = stack_in[:, :, -d_x:]
            stack_in = stack_out
        elif d_x > 0:
            stack_out = stack_in[:, :, 0:-d_x:]
            stack_in = stack_out

        if d_y < 0:
            stack_out = stack_in[:, -d_y:, :]
        elif d_y > 0:
            stack_out = stack_in[:, 0:-d_y:, :]

    return stack_out


def mask_generate(frame_in, mask_type='Otsu_global'):
    """Generate a mask for a frame 2-D array (Y, X) of grayscale optical data
    using binary threshold (histogram-based or local) or segmentation algorithms.

       Parameters
       ----------
       frame_in : ndarray
            A 2-D array (Y, X) of optical data, dtype : uint16 or float
       mask_type : str  # TODO add masking via SNR
            The type of masking thresholding algorithm to use, default : Otsu_global

       Returns
       -------
       frame_out : ndarray
            A 2-D array (Y, X) of masked optical data,  dtype : frame_in.dtype
       mask : ndarray
            A binary 2-D array generated from the threshold algorithm, dtype : np.bool_
       """
    # Check parameters
    if type(frame_in) is not np.ndarray:
        raise TypeError('Frame type must be an "ndarray"')
    if len(frame_in.shape) is not 2:
        raise TypeError('Frame must be a 2-D ndarray (Y, X)')
    if frame_in.dtype not in [np.uint16, float]:
        raise TypeError('Frame values must either be "np.uint16" or "float"')
    if type(mask_type) is not str:
        raise TypeError('Filter type must be a "str"')

    if mask_type not in MASK_TYPES:
        raise ValueError('Filter type must be one of the following: {}'.format(MASK_TYPES))

    frame_out = frame_in.copy()
    mask = frame_in.copy()
    frame_in_gradient = sobel(frame_in)

    if mask_type is 'Otsu_global':
        # Good for ___, but ___
        global_otsu = threshold_otsu(frame_in)
        binary_global = frame_in <= global_otsu
        mask = binary_global
        frame_out[mask] = 0

    elif mask_type is 'Mean':
        # Good for ___, but __
        thresh = threshold_mean(frame_in)
        binary_global = frame_in <= thresh
        mask = binary_global
        frame_out[mask] = 0

    elif mask_type is 'Random_walk':
        # The range of the binary image spans over (-1, 1).
        # We choose the hottest and the coldest pixels as markers.
        frame_in_float = img_as_float(frame_in)
        # TODO "smooth" before marking for mask
        # reduction_factor = 1 / 5
        # frame_in_rescale_space = rescale(frame_in_float, reduction_factor)

        frame_in_rescale = rescale_intensity(frame_in_float,
                                             in_range=(frame_in_float.min(), frame_in_float.max()),
                                             out_range=(-1, 1))
        markers = np.zeros(frame_in_rescale.shape)
        # Darkish half and lightish half
        # TODO calculate these bounds
        otsu = threshold_otsu(frame_in_rescale, nbins=256 * 2)
        # adjusted_otsu = otsu

        darker_otsu = np.mean([-1, otsu])
        dark_otsu = np.mean([darker_otsu, otsu])
        lightest_otsu = np.mean([dark_otsu, otsu])
        light_otsu = np.mean([dark_otsu, lightest_otsu])
        lighter_otsu = np.mean([light_otsu, lightest_otsu])

        adjusted_otsu = lighter_otsu
        # # adjusted_otsu = global_otsu - (abs((-1 - global_otsu)/2))

        print('* Masking with Otsu value: {}'.format(adjusted_otsu))
        markers[frame_in_rescale < adjusted_otsu] = 1
        markers[frame_in_rescale >= adjusted_otsu] = 2

        # Run random walker algorithm
        binary_random_walk = random_walker(frame_in_rescale, markers, beta=2, mode='bf')
        # Keep the largest region, as a np.uint16
        labeled_mask = label(binary_random_walk)
        largest_mask = np.empty_like(labeled_mask, dtype=np.bool_)
        largest_region_area = 0
        for idx, region_prop in enumerate(regionprops(labeled_mask)):
            # Use the biggest bright region

            # for prop in region_prop:
            #     print(prop, region_prop[prop])
            # use the second-largest region
            if region_prop.area < 2:
                pass
            if region_prop.area > largest_region_area and region_prop.label > 1:
                print('* Region #{}\t:\tint: _\tarea: {}'
                      .format(idx + 1, region_prop.area))
                largest_region_area = region_prop.area
                largest_mask[labeled_mask == region_prop.label] = False
                largest_mask[labeled_mask != region_prop.label] = True
                print('* Using #{} area: {}'
                      .format(idx+1, region_prop.area))

        frame_out[largest_mask] = 0
        # mask = markers
        mask = largest_mask
    else:
        raise NotImplementedError('Mask type "{}" not implemented'.format(mask_type))

    return frame_out, mask


def mask_apply(stack_in, mask):
    """Apply a binary mask to segment a stack (3-D array, TYX) of grayscale optical data.

       Parameters
       ----------
       stack_in : ndarray
            A 3-D array (T, Y, X) of optical data, dtype : uint16 or float
       mask : ndarray
            A binary 2-D array (Y, X) to mask optical data, dtype : np.bool_

       Returns
       -------
       stack_out : ndarray
            A masked 3-D array (T, Y, X) of optical data, dtype : stack_in.dtype
            Masked values are FL_16BIT_MAX (aka 65535)
       """
    # Check parameters
    if type(stack_in) is not np.ndarray:
        raise TypeError('Stack type must be an "ndarray"')
    if len(stack_in.shape) is not 3:
        raise TypeError('Stack must be a 3-D ndarray (T, Y, X)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')

    if type(mask) is not np.ndarray:
        raise TypeError('Mask type must be an "ndarray"')
    if mask.dtype not in [np.int64, bool]:
        raise TypeError('Stack values must either be "np.bool_"')
    if len(mask.shape) is not 2:
        raise TypeError('Mask must be a 2-D ndarray (Y, X)')

    frame_0 = stack_in[0]

    # if (mask.shape[0] is not frame_0.shape[0]) or (mask.shape[1] is not frame_0.shape[1]):
    if mask.shape != frame_0.shape:
        raise ValueError('Mask shape must be the same as the stack frames:'
                         '\nMask:\t{}\nFrame:\t{}'.format(mask.shape, frame_0.shape))

    stack_out = np.empty_like(stack_in)

    for i_frame, frame in enumerate(stack_in):
        frame_out = frame.copy()
        frame_out[mask] = 0
        stack_out[i_frame] = frame_out

    return stack_out


def get_gradient(im):
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def align_signals(signal1, signal2):
    """Aligns two signal arrays using signal.correlate.
    https://stackoverflow.com/questions/19642443/use-of-pandas-shift-to-align-datasets-based-on-scipy-signal-correlate

        Parameters
        ----------
        signal1 : ndarray, dtype : uint16 or float
            Signal array
        signal2 : ndarray, dtype : uint16 or float
            Signal array, will be aligned to signal1

        Returns
        -------
        signal2_aligned : ndarray
            Aligned version of signal2

        Notes
        -----
            Signal arrays must be the same length?
            Should not be applied to signal data containing at least one transient.
            Fills empty values with np.NaN
    """
    # Set signal datatype as float32
    sig1 = np.float32(signal1)
    sig2 = np.float32(signal2)

    # Find the length of the signal
    # sig_length = len(sig1)
    # print('sig1 min, max: ', np.nanmin(sig1), ' , ', np.nanmax(sig1))
    # print('sig2 min, max: ', np.nanmin(sig2), ' , ', np.nanmax(sig2))

    # dx = np.mean(np.diff(sig1.x.values))
    shift = (np.argmax(signal.correlate(sig1, sig2)) - len(sig2))

    signal2_aligned = np.roll(sig2, shift=shift+1)
    if shift > 0:
        signal2_aligned[:shift] = np.nan
        # signal2_aligned = signal2_aligned[shift:]
    else:
        signal2_aligned[shift:] = np.nan
        # signal2_aligned = signal2_aligned[:-shift]

    return signal2_aligned


def align_stacks(stack1, stack2):
    """Aligns two stacks of images using the gradient representation of the image
    and a similarity measure called Enhanced Correlation Coefficient (ECC).
    TODO try Feature-Based approach https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/, https://github.com/spmallick/learnopencv/blob/c8e3ae2d2b0423f5c6d21c6189ee8ff3192c0555/ImageAlignment-FeatureBased/align.py

        Parameters
        ----------
        stack1 : ndarray, dtype : uint16
            Image stack with shape (x, y, t)
        stack2 : ndarray, dtype : uint16
            Image stack with shape (x, y, t), will be aligned to stack1

        Returns
        -------
        stack2_aligned : ndarray
            Aligned version of stack2


        Notes
        -----
        # Based on examples by Satya Mallick (https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/)
    """
    # Read uint16 grayscale images from the image stacks    # TODO actually use uint16
    im1 = np.float32(stack1[..., 0])
    im2 = np.float32(stack2[..., 0])

    # Find the width and height of the image
    im_size = im1.shape
    width, height = im_size[0], im_size[1]
    print('im1 min, max: ', np.nanmin(im1), ' , ', np.nanmax(im1))
    print('im2 min, max: ', np.nanmin(im2), ' , ', np.nanmax(im2))
    # Find the number of frames in the stacks (should be identical)
    frames = stack1.shape[2]

    # Allocate space for aligned image
    im2_aligned = np.zeros((width, height), dtype=np.uint16, order='F')
    # Define motion model
    warp_mode = cv2.MOTION_TRANSLATION
    # Define 2x3 matrices and initialize the matrix to identity
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Specify the number of iterations
    number_of_iterations = 5000
    # Specify the threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-10
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)

    start = time.time()
    # Warp the second stack image to the first
    # Run the ECC algorithm, the results are stored in warp_matrix
    (cc, warp_matrix) = cv2.findTransformECC(get_gradient(im1), get_gradient(im2),
                                             warp_matrix, warp_mode, criteria)
    # Use Affine warp when the transformation is not a Homography
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (height, width),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    print('im2_aligned min, max: ', np.nanmin(im2_aligned), ' , ', np.nanmax(im2_aligned))
    # Convert aligned image back to uint16
    im2_aligned = np.uint16(im2_aligned)
    print('im2_aligned min, max: ', np.nanmin(im2_aligned), ' , ', np.nanmax(im2_aligned))

    # Align and save every stack2 frame using the same process
    stack2_aligned = np.zeros((width, height, frames), dtype=np.uint16, order='F')
    for i in range(frames):
        # Find the old frame
        stack2_frame = np.float32(stack2[..., i])
        stack2_frame_aligned = cv2.warpAffine(stack2_frame, warp_matrix, (height, width),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        # Convert aligned frame back to uint16
        stack2_frame_aligned = np.uint16(stack2_frame_aligned)
        # Save the aligned frame in the new stack
        stack2_aligned[..., i] = stack2_frame_aligned

    # # ECC Method
    # # Image registration using first frame
    # # Read the images to be aligned
    # im1 = np.float32(stack1[0, ...])
    # im2 = np.float32(stack2[0, ...])
    # # # Convert images to grayscale
    # # im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # # im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # # Find size of image1
    # sz = im1.shape
    # im1_min, im1_max = np.nanmin(im1), np.nanmax(im1)
    # im2_min, im2_max = np.nanmin(im2), np.nanmax(im2)
    # print('im1 min, max: ', im1_min, ' , ', im1_max)
    # print('im2 min, max: ', im2_min, ' , ', im2_max)
    #
    # # Define the motion model
    # warp_mode = cv2.MOTION_TRANSLATION
    # # Define 2x3 matrices and initialize the matrix to identity
    # warp_matrix = np.eye(2, 3, dtype=np.float32)
    # # Specify the number of iterations.
    # number_of_iterations = 5000
    # # Specify the threshold of the increment
    # # in the correlation coefficient between two iterations
    # termination_eps = 1e-10
    # # Define termination criteria
    # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    #
    # # Run the ECC algorithm. The results are stored in warp_matrix.
    # (cc, warp_matrix) = cv2.findTransformECC(im1, im2, warp_matrix, warp_mode, criteria)
    #
    # # Use warpAffine for Translation, Euclidean and Affine
    # im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    #
    # cv2.imshow("Aligned Image 2", im2_aligned)
    # cv2.waitKey(0)
    end = time.time()
    print('Alignment time (s): ', end - start)

    return stack2_aligned
