import numpy as np
import cv2 as cv
from typing import Tuple, Optional

threshold = 0.5  # threshold for binarization


def minimum_length_solid(arr: np.ndarray,
                         phys_size: Optional[Tuple[float, ...]] = None,
                         margin_size: Optional[Tuple[Tuple[float, float],
                                                     ...]] = None,
                         pad_mode: str = 'solid') -> float:
    """
    Compute the minimum length scale of solid regions in a design pattern.

    Args:
        arr: A 1d or 2d array that represents a design pattern.
        phys_size: A tuple that represents the physical size of the design pattern.
        margin_size: A tuple that represents the physical size near edges that need to be disregarded.
        pad_mode: A string that represents the padding mode, which can be 'solid', 'void', or 'edge'.

    Returns:
        A float that represents the minimum length scale of solid regions in the design pattern. The unit is the same as that of `phys_size`. If `phys_size` is None, return the minimum length scale in the number of pixels.
    """

    arr, pixel_size, short_pixel_side, short_entire_side = _ruler_initialize(
        arr, phys_size)

    # If all elements in the array are the same,
    # the code simply regards the shorter side of
    # the entire pattern as the minimum length scale,
    # regardless of whether the pattern is solid or void.

    if len(np.unique(arr)) == 1:
        return short_entire_side

    if arr.ndim == 1:
        if margin_size != None:
            arr = _trim(arr, margin_size, pixel_size)
        solid_min_length, _ = _minimum_length_1d(arr)
        return solid_min_length * short_pixel_side

    def _interior_pixel_number(diameter, arr):
        """
        Evaluate whether a design pattern violates a certain length scale.

        Args:
            diameter: A float that represents the diameter of the kernel, which acts like a probe.
            arr: A 2d array that represents a design pattern.

        Returns:
            A boolean that indicates whether the difference between the design pattern and its opening happens at the interior of solid regions, with the edge regions specified by `margin_size` disregarded.
        """
        open_diff = binary_open(arr, diameter, pixel_size, pad_mode) ^ arr
        interior_diff = open_diff & _get_interior(
            arr, direction="in", pad_mode=pad_mode)
        if margin_size != None:
            interior_diff = _trim(interior_diff, margin_size, pixel_size)
        return interior_diff.any()

    min_len, _ = _search([short_pixel_side, short_entire_side],
                         min(pixel_size) / 2,
                         lambda d: _interior_pixel_number(d, arr))

    return min_len


def minimum_length_void(arr: np.ndarray,
                        phys_size: Optional[Tuple[float, ...]] = None,
                        margin_size: Optional[Tuple[Tuple[float, float],
                                                    ...]] = None,
                        pad_mode: str = 'void') -> float:
    """
    Compute the minimum length scale of void regions in a design pattern.

    Args:
        arr: A 1d or 2d array that represents a design pattern.
        phys_size: A tuple that represents the physical size of the design pattern.
        margin_size: A tuple that represents the physical size near edges that need to be disregarded.
        pad_mode: A string that represents the padding mode, which can be 'solid', 'void', or 'edge'.

    Returns:
        A float that represents the minimum length scale of void regions in the design pattern. The unit is the same as that of `phys_size`. If `phys_size` is None, return the minimum length scale in the number of pixels.
    """

    arr, pixel_size, short_pixel_side, short_entire_side = _ruler_initialize(
        arr, phys_size)
    if pad_mode == 'solid': pad_mode = 'void'
    elif pad_mode == 'void': pad_mode = 'solid'
    else: pad_mode == 'edge'

    return minimum_length_solid(~arr, phys_size, margin_size, pad_mode)


def minimum_length_solid_void(
    arr: np.ndarray,
    phys_size: Optional[Tuple[float, ...]] = None,
    margin_size: Optional[Tuple[Tuple[float, float], ...]] = None,
    pad_mode: Tuple[str, str] = ('solid', 'void')
) -> Tuple[float, float]:
    """
    Compute the minimum length scales of both solid and void regions in a design pattern.

    Args:
        arr: A 1d or 2d array that represents a design pattern.
        phys_size: A tuple that represents the physical size of the design pattern.
        margin_size: A tuple that represents the physical size near edges that need to be disregarded.
        pad_mode: A tuple of two strings that represent the padding modes for measuring solid and void minimum length scales, respectively.

    Returns:
        A tuple of two floats that represent the minimum length scales of solid and void regions, respectively. The unit is the same as that of `phys_size`. If `phys_size` is None, return the minimum length scale in the number of pixels.
    """

    return minimum_length_solid(arr, phys_size, margin_size,
                                pad_mode[0]), minimum_length_void(
                                    arr, phys_size, margin_size, pad_mode[1])


def minimum_length(
    arr: np.ndarray,
    phys_size: Optional[Tuple[float, ...]] = None,
    margin_size: Optional[Tuple[Tuple[float, float], ...]] = None,
    pad_mode: Tuple[str, str] = ('solid', 'void')
) -> Tuple[float, float]:
    """
    Compute the minimum length scales of both solid and void regions in a design pattern.

    Args:
        arr: A 1d or 2d array that represents a design pattern.
        phys_size: A tuple that represents the physical size of the design pattern.
        margin_size: A tuple that represents the physical size near edges that need to be disregarded.
        pad_mode: A tuple of two strings that represent the padding modes for measuring solid and void minimum length scales, respectively.

    Returns:
        A tuple of two floats that represent the minimum length scales of solid and void regions, respectively. The unit is the same as that of `phys_size`. If `phys_size` is None, return the minimum length scale in the number of pixels.
    """

    if isinstance(pad_mode, str): pad_mode = (pad_mode, pad_mode)
    return min(minimum_length_solid(arr, phys_size, margin_size, pad_mode[0]),
               minimum_length_void(arr, phys_size, margin_size, pad_mode[1]))


def minimum_length_min(
    arr: np.ndarray,
    phys_size: Optional[Tuple[float, ...]] = None,
    margin_size: Optional[Tuple[Tuple[float, float], ...]] = None,
    pad_mode: Tuple[str, str] = ('solid', 'void')
) -> float:
    """
    For 2d design patterns, compute the minimum length scale through the difference between morphological opening and closing.
    Ideally, the result should be equal to the smaller one between solid and void minimum length scales.
    For 1d design patterns, just return this smaller one after comparing solid and void minimum length scales.

    Args:
        arr: A 1d or 2d array that represents a design pattern.
        phys_size: A tuple that represents the physical size of the design pattern.
        margin_size: A tuple that represents the physical size near edges that need to be disregarded.
        pad_mode: A tuple of two strings that represent the padding modes for morphological opening nad closing, respectively.

    Returns:
        A float that represents the minimum length scale in the design pattern. The unit is the same as that of `phys_size`. If `phys_size` is None, return the minimum length scale in the number of pixels.
    """

    arr, pixel_size, short_pixel_side, short_entire_side = _ruler_initialize(
        arr, phys_size)

    # If all elements in the array are the same,
    # the code simply regards the shorter side of
    # the entire pattern as the minimum length scale,
    # regardless of whether the pattern is solid or void.

    if len(np.unique(arr)) == 1:
        return short_entire_side

    if arr.ndim == 1:
        if margin_size != None:
            arr = _trim(arr, margin_size, pixel_size)
        solid_min_length, void_min_length = _minimum_length_1d(arr)
        return min(solid_min_length, void_min_length) * short_pixel_side

    if isinstance(pad_mode, str): pad_mode = (pad_mode, pad_mode)

    def _interior_pixel_number(diameter, arr):
        """
        Evaluate whether a design pattern violates a certain length scale.

        Args:
            diameter: A float that represents the diameter of the kernel, which acts like a probe.
            arr: A 2d array that represents a design pattern.

        Returns:
            A boolean that indicates whether the difference between opening and closing happens at the regions that exclude the borders between solid and void regions, with the edge regions specified by `margin_size` disregarded.
        """

        closing = binary_close(arr, diameter, pixel_size, pad_mode[1])
        close_open_diff = binary_open(arr, diameter, pixel_size,
                                      pad_mode[0]) ^ closing
        interior_diff = close_open_diff & _get_interior(
            arr, direction="both", pad_mode=pad_mode)
        if margin_size != None:
            interior_diff = _trim(interior_diff, margin_size, pixel_size)
        return interior_diff.any()

    min_len, _ = _search([short_pixel_side, short_entire_side],
                         min(pixel_size) / 2,
                         lambda d: _interior_pixel_number(d, arr))

    return min_len


def _ruler_initialize(arr, phys_size):
    """
    Convert the input array to a Boolean array without redundant dimensions and compute some basic information of the design pattern.

    Args:
        arr: An array that represents a design pattern.
        phys_size: A tuple, list, array, or number that represents the physical size of the design pattern.

    Returns:
        A tuple with four elements. The first is a Boolean array obtained by squeezing and binarizing the input array, the second is an array that contains the pixel size, the third is the length of the shorter side of the pixel, and the fourth is the length of the shorter side of the design pattern.

    Raises:
        AssertionError: If the physical size `phys_size` does not have the expected format or the length of `phys_size` does not match the dimension of the input array. 
    """

    arr = np.squeeze(arr)

    if isinstance(phys_size, np.ndarray) or isinstance(
            phys_size, list) or isinstance(phys_size, tuple):
        phys_size = np.squeeze(phys_size)
        phys_size = phys_size[
            phys_size.nonzero()]  # keep nonzero elements only
    elif isinstance(phys_size, float) or isinstance(phys_size, int):
        phys_size = np.array([phys_size])
    elif phys_size == None:
        phys_size = arr.shape
    else:
        AssertionError("Invalid format of the physical size.")

    assert arr.ndim == len(
        phys_size
    ), 'The physical size and the dimension of the input array do not match.'

    short_entire_side = min(
        phys_size)  # shorter side of the entire design region
    pixel_size = _get_pixel_size(arr, phys_size)
    short_pixel_side = min(pixel_size)  # shorter side of a pixel
    arr = _binarize(arr)  # Boolean array

    return arr, pixel_size, short_pixel_side, short_entire_side


def _search(arg_range, arg_threshold, function):
    """
    Binary search.

    Args:
        arg_range: Initial range of the argument under search.
        arg_threshold: Threshold of the argument range, below which the search stops.
        function: A function that returns True if the viariable is large enough but False if the variable is not large enough.

    Returns:
        A tuple with two elements. The first is a float that represents the search result. The second is a Boolean, which is True if the search indeed happens, False if the condition for starting search is not satisfied in the beginning.

    Raises:
        AssertionError: If `function` returns True at a smaller input viariable but False at a larger input viariable.
    """

    args = [
        min(arg_range), (min(arg_range) + max(arg_range)) / 2,
        max(arg_range)
    ]

    if not function(args[0]) and function(args[2]):
        while abs(args[0] - args[2]) > arg_threshold:
            arg = args[1]
            if not function(arg):
                args[0], args[1] = arg, (arg +
                                         args[2]) / 2  # radius is too small
            else:
                args[1], args[2] = (arg +
                                    args[0]) / 2, arg  # radius is still large
        return args[1], True  #args[2], True
    elif not function(args[0]) and not function(args[2]):
        return args[2], False  #args[2], False
    elif function(args[0]) and function(args[2]):
        return args[0], False
    else:
        raise AssertionError("The function is not monotonically increasing.")


def _minimum_length_1d(arr):
    """
    Search the minimum lengths of solid and void segments in a 1d array.

    Args:
        arr: A 1d Boolean array.

    Return:
        A tuple of two integers. The first and second intergers represent the numbers of pixels in the shortest solid and void segments, respectively.
    """

    arr = np.append(arr, ~arr[-1])
    solid_lengths, void_lengths = [], []
    counter = 0

    for idx in range(len(arr) - 1):
        counter += 1

        if arr[idx] != arr[idx + 1]:
            if arr[idx]:
                solid_lengths.append(counter)
            else:
                void_lengths.append(counter)
            counter = 0

    if len(solid_lengths) > 0:
        solid_min_length = min(solid_lengths)
    else:
        solid_min_length = 0

    if len(void_lengths) > 0:
        void_min_length = min(void_lengths)
    else:
        void_min_length = 0

    return solid_min_length, void_min_length


def _get_interior(arr, direction, pad_mode):
    """
    Get inner borders, outer borders, or union of both inner and outer borders of solid regions.

    Args:
        arr: A 2d array that represents a design pattern.
        direction: A string that can be "in", "out", or "both" to indicate inner borders, outer borders, and union of inner and outer borders.

    Returns:
        A Boolean array in which all True elements are at and only at borders.

    Raises:
        AssertionError: If the option provided to `direction` is not 'in', 'out', or 'both'.
    """

    pixel_size = (1, ) * arr.ndim
    diameter = 2.8  # With this pixel size and diameter, the resulting kernel has the shape of a plus sign.

    if direction == 'in':  # interior of solid regions
        return binary_erode(arr, diameter, pixel_size, pad_mode)
    elif direction == 'out':  # interior of void regions
        return ~binary_dilate(arr, diameter, pixel_size, pad_mode)
    elif direction == 'both':  # union of interiors of solid and void regions
        eroded = binary_erode(arr, diameter, pixel_size, pad_mode[0])
        dilated = binary_dilate(arr, diameter, pixel_size, pad_mode[1])
        return ~dilated | eroded
    else:
        raise AssertionError(
            "The direction at the border can only be in, out, or both.")


def _get_pixel_size(arr, phys_size):
    """
    Compute the physical size of a single pixel.

    Args:
        arr: An array that represents a design pattern.
        phys_size: A tuple that represents the physical size of the design pattern.

    Returns:
        An array of floats. It represents the physical size of a single pixel.
    """

    squeeze_shape = np.array(np.squeeze(arr).shape)
    return phys_size / squeeze_shape  # sizes of a pixel along all finite-thickness directions


def _binarize(arr):
    """
    Binarize the input array according to the threshold.

    Args:
        arr: An array that represents a design pattern.

    Returns:
        An Boolean array.
    """

    return arr > threshold * max(arr.flatten()) + (1 - threshold) * min(
        arr.flatten())


def _get_kernel(diameter, pixel_size):
    """
    Get the kernel with a given diameter and pixel size.

    Args:
        diameter: A float that represents the diameter of the kernel, which acts like a probe.
        pixel_size: A tuple, list, or array that represents the physical size of one pixel in the design pattern.

    Returns:
        An array of unsigned integers 0 and 1. It represent the kernel for morpological operations.
    """

    pixel_size = np.array(pixel_size)
    se_shape = np.array(np.round(diameter / pixel_size), dtype=int)

    if se_shape[0] <= 2 and se_shape[1] <= 2:
        return np.ones(se_shape, dtype=np.uint8)

    rounded_size = np.round(diameter / pixel_size - 1) * pixel_size

    x_tick = np.linspace(-rounded_size[0] / 2, rounded_size[0] / 2,
                         se_shape[0])
    y_tick = np.linspace(-rounded_size[1] / 2, rounded_size[1] / 2,
                         se_shape[1])

    X, Y = np.meshgrid(x_tick, y_tick, sparse=True,
                       indexing='ij')  # grid over the entire design region
    structuring_element = X**2 + Y**2 <= diameter**2 / 4

    return np.array(structuring_element, dtype=np.uint8)


def binary_open(arr: np.ndarray,
                diameter: float,
                pixel_size: Tuple[float, float],
                pad_mode: str = 'edge'):
    """
    Morphological opening.

    Args:
        arr: A binarized 2d array that represents a design pattern.
        diameter: A float that represents the diameter of the kernel, which acts like a probe.
        pixel_size: A tuple, list, or array that represents the physical size of one pixel in the design pattern.
        pad_mode: A string that represents the padding mode, which can be 'solid', 'void', or 'edge'.

    Returns:
        A Boolean array that represents the outcome of morphological opening. 
    """

    kernel = _get_kernel(diameter, pixel_size)
    arr = _proper_pad(arr, kernel, pad_mode)
    opened = cv.morphologyEx(src=arr, kernel=kernel, op=cv.MORPH_OPEN)
    return _proper_unpad(opened, kernel).astype(bool)


def binary_close(arr: np.ndarray,
                 diameter: float,
                 pixel_size: Tuple[float, float],
                 pad_mode: str = 'edge'):
    """
    Morphological closing.

    Args:
        arr: A binarized 2d array that represents a design pattern.
        diameter: A float that represents the diameter of the kernel, which acts like a probe.
        pixel_size: A tuple, list, or array that represents the physical size of one pixel in the design pattern.
        pad_mode: A string that represents the padding mode, which can be 'solid', 'void', or 'edge'.

    Returns:
        A Boolean array that represents the outcome of morphological closing. 
    """

    kernel = _get_kernel(diameter, pixel_size)
    arr = _proper_pad(arr, kernel, pad_mode)
    closed = cv.morphologyEx(src=arr, kernel=kernel, op=cv.MORPH_CLOSE)
    return _proper_unpad(closed, kernel).astype(bool)


def binary_erode(arr: np.ndarray,
                 diameter: float,
                 pixel_size: Tuple[float, float],
                 pad_mode: str = 'edge'):
    """
    Morphological erosion.

    Args:
        arr: A binarized 2d array that represents a design pattern.
        diameter: A float that represents the diameter of the kernel, which acts like a probe.
        pixel_size: A tuple, list, or array that represents the physical size of one pixel in the design pattern.
        pad_mode: A string that represents the padding mode, which can be 'solid', 'void', or 'edge'.

    Returns:
        A Boolean array that represents the outcome of morphological erosion. 
    """

    kernel = _get_kernel(diameter, pixel_size)
    arr = _proper_pad(arr, kernel, pad_mode)
    eroded = cv.erode(arr, kernel)
    return _proper_unpad(eroded, kernel).astype(bool)


def binary_dilate(arr: np.ndarray,
                  diameter: float,
                  pixel_size: Tuple[float, float],
                  pad_mode: str = 'edge'):
    """
    Morphological dilation.

    Args:
        arr: A binarized 2d array that represents a design pattern.
        diameter: A float that represents the diameter of the kernel, which acts like a probe.
        pixel_size: A tuple, list, or array that represents the physical size of one pixel in the design pattern.
        pad_mode: A string that represents the padding mode, which can be 'solid', 'void', or 'edge'.

    Returns:
        A Boolean array that represents the outcome of morphological dilation. 
    """

    kernel = _get_kernel(diameter, pixel_size)
    arr = _proper_pad(arr, kernel, pad_mode)
    dilated = cv.dilate(arr, kernel)
    return _proper_unpad(dilated, kernel).astype(bool)


def _proper_pad(arr, kernel, pad_mode):
    """
    Pad the input array properly according to the size of the kernel.

    Args:
        arr: A binarized 2d array that represents a design pattern.
        kernel: A 2d array that represents the kernel of morphological operations.
        pad_mode: A string that represents the padding mode, which can be 'solid', 'void', or 'edge'.

    Returns:
        A padded array composed of unsigned integers 0 and 1.

    Raises:
        AssertionError: If `pad_mode` is not `solid`, `void`, or `edge`.
    """

    ((top, bottom), (left, right)) = ((kernel.shape[0], ) * 2,
                                      (kernel.shape[1], ) * 2)

    if pad_mode == 'edge':
        return cv.copyMakeBorder(arr.view(np.uint8),
                                 top=top,
                                 bottom=bottom,
                                 left=left,
                                 right=right,
                                 borderType=cv.BORDER_REPLICATE)
    elif pad_mode == 'solid':
        return cv.copyMakeBorder(arr.view(np.uint8),
                                 top=top,
                                 bottom=bottom,
                                 left=left,
                                 right=right,
                                 borderType=cv.BORDER_CONSTANT,
                                 value=1)
    elif pad_mode == 'void':
        return cv.copyMakeBorder(arr.view(np.uint8),
                                 top=top,
                                 bottom=bottom,
                                 left=left,
                                 right=right,
                                 borderType=cv.BORDER_CONSTANT,
                                 value=0)
    else:
        raise AssertionError(
            "The padding mode should be 'solid', 'void', or 'edge'.")


def _proper_unpad(arr, kernel):
    """
    Remove padding according to the size of the kernel. The code is copied from Martin F. Schubert's code at 
    https://github.com/mfschubert/topology/blob/main/metrics.py

    Args:
        arr: A 2d array that has extra padding.
        kernel: A 2d array that represents the kernel of morphological operations.

    Returns:
        A 2d array without padding.
    """

    unpad_width = (
        (
            kernel.shape[0] + (kernel.shape[0] + 1) % 2,
            kernel.shape[0] - (kernel.shape[0] + 1) % 2,
        ),
        (
            kernel.shape[1] + (kernel.shape[1] + 1) % 2,
            kernel.shape[1] - (kernel.shape[1] + 1) % 2,
        ),
    )

    slices = tuple([
        slice(pad_lo, dim - pad_hi)
        for (pad_lo, pad_hi), dim in zip(unpad_width, arr.shape)
    ])
    return arr[slices]


def _trim(arr, margin_size, pixel_size):
    """
    Obtain the trimmed array with marginal regions discarded.

    Args:
        arr: A 1d or 2d array that represents a design pattern.
        margin_size: A tuple that represents the physical size near edges that need to be disregarded.
        pixel_size: A tuple that represents the physical size of one pixel in the design pattern.

    Returns:
        An array that is a portion of the input array.

    Raises:
        AssertionError: If `margin_size` implies more dimensions than the dimension of the input array `arr`, or the regions to be disregarded is too wide.
    """

    arr = np.squeeze(arr)
    arr_dim = arr.ndim
    margin_size = abs(np.reshape(margin_size, (-1, 2)))
    margin_dim = len(margin_size)

    assert margin_dim <= arr_dim, 'The number of rows of margin_size should not exceeds the dimension of the input array.'

    margin_number = np.array(
        margin_size) / pixel_size[0:len(margin_size)].reshape(
            len(margin_size), 1)
    margin_number = np.round(margin_number).astype(
        int)  # numbers of pixels of marginal regions

    assert (np.array(arr.shape)[0:margin_dim] - np.sum(margin_number, axis=1)
            >= 2).all(), 'Too wide margin or too narrow design region.'

    if margin_dim == 1:
        return arr[margin_number[0][0]:-margin_number[0][1]]
    elif margin_dim == 2:
        return arr[margin_number[0][0]:-margin_number[0][1],
                   margin_number[1][0]:-margin_number[1][1]]
    else:
        AssertionError("The input array has too many dimensions.")