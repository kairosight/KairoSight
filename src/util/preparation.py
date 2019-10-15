import os
from imageio import volread, get_reader
# import numpy as np


def load_single(source, meta=None):
    """Load images from a file containing one signal source or channel

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

