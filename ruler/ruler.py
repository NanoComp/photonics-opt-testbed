import numpy as np
from typing import Tuple, Optional

threshold = 0.5  # threshold for binarization


def solid_minimum_length(
        arr: np.ndarray,
        phys_size: Optional[Tuple[float, ...]] = None,
        margin_size: Optional[Tuple[Tuple[float, float],
                                    ...]] = None) -> float:
    """
    Compute the minimum length scale of solid regions in a design pattern.

    Args:
        arr: A 1d, 2d, or 3d array that represents a design pattern.
        phys_size: A tuple that represents the physical size of the design pattern.
        margin_size: A tuple that represents the physical size near edges that need to be disregarded.

    Returns:
        A float that represents the minimum length scale of solid regions in the design pattern. The unit is the same as that of `phys_size`. If `phys_size` is None, return the minimum length scale in the number of pixels.
    """

    arr, pixel_size, short_pixel_side, short_entire_side = _ruler_initialize(
        arr, phys_size)
    """
    If all elements in the array are the same, the code simply regards the shorter side of the entire pattern as the minimum length scale, regardless of whether the pattern is solid or void.
    """
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
            diameter: A float that represents the diameter of the structuring element, which acts like a probe.
            arr: A 2d or 3d array that represents a design pattern.

        Returns:
            A boolean that indicates whether the difference between the design pattern and its opening happens at the interior of solid regions, with the edge regions specified by `margin_size` disregarded.
        """
        open_diff = heaviside_open(arr, diameter, pixel_size) ^ arr
        interior = arr ^ _get_border(arr, direction="in")
        interior_diff = open_diff & interior
        if margin_size != None:
            interior_diff = _trim(interior_diff, margin_size, pixel_size)
        return interior_diff.any()

    min_len, _ = _search([short_pixel_side, short_entire_side],
                         min(pixel_size),
                         lambda d: _interior_pixel_number(d, arr))

    return min_len


def void_minimum_length(
        arr: np.ndarray,
        phys_size: Optional[Tuple[float, ...]] = None,
        margin_size: Optional[Tuple[Tuple[float, float],
                                    ...]] = None) -> float:
    """
    Compute the minimum length scale of void regions in a design pattern.

    Args:
        arr: A 1d, 2d, or 3d array that represents a design pattern.
        phys_size: A tuple that represents the physical size of the design pattern.
        margin_size: A tuple that represents the physical size near edges that need to be disregarded.

    Returns:
        A float that represents the minimum length scale of void regions in the design pattern. The unit is the same as that of `phys_size`. If `phys_size` is None, return the minimum length scale in the number of pixels.
    """

    arr, pixel_size, short_pixel_side, short_entire_side = _ruler_initialize(
        arr, phys_size)
    return solid_minimum_length(~arr, phys_size, margin_size)


def both_minimum_length(
    arr: np.ndarray,
    phys_size: Optional[Tuple[float, ...]] = None,
    margin_size: Optional[Tuple[Tuple[float, float], ...]] = None
) -> Tuple[float, float]:
    """
    Compute the minimum length scales of both solid and void regions in a design pattern.

    Args:
        arr: A 1d, 2d, or 3d array that represents a design pattern.
        phys_size: A tuple that represents the physical size of the design pattern.
        margin_size: A tuple that represents the physical size near edges that need to be disregarded.

    Returns:
        A tuple of two floats that represent the minimum length scales of solid and void regions, respectively. The unit is the same as that of `phys_size`. If `phys_size` is None, return the minimum length scale in the number of pixels.
    """

    return solid_minimum_length(arr, phys_size,
                                margin_size), void_minimum_length(
                                    arr, phys_size, margin_size)


def dual_minimum_length(
        arr: np.ndarray,
        phys_size: Optional[Tuple[float, ...]] = None,
        margin_size: Optional[Tuple[Tuple[float, float],
                                    ...]] = None) -> float:
    """
    For 2d or 3d design patterns, compute the minimum length scale through the difference between morphological opening and closing.
    Ideally, the result should be equal to the smaller one between solid and void minimum length scales.
    For 1d design patterns, just return this smaller one after comparing solid and void minimum length scales.

    Args:
        arr: A 1d, 2d, or 3d array that represents a design pattern.
        phys_size: A tuple that represents the physical size of the design pattern.
        margin_size: A tuple that represents the physical size near edges that need to be disregarded.

    Returns:
        A float that represents the minimum length scale in the design pattern. The unit is the same as that of `phys_size`. If `phys_size` is None, return the minimum length scale in the number of pixels.
    """

    arr, pixel_size, short_pixel_side, short_entire_side = _ruler_initialize(
        arr, phys_size)
    """
    If all elements in the array are the same,
    the code simply regards the shorter side of the entire pattern as the minimum length scale,
    regardless of whether the pattern is solid or void.
    """
    if len(np.unique(arr)) == 1:
        return short_entire_side

    if arr.ndim == 1:
        arr = arr[margin_number[0, 0]:len(arr) - margin_number[0, 1]]
        solid_min_length, void_min_length = _minimum_length_1d(arr)
        return min(solid_min_length, void_min_length) * short_pixel_side

    def _interior_pixel_number(diameter, arr):
        """
        Evaluate whether a design pattern violates a certain length scale.

        Args:
            diameter: A float that represents the diameter of the structuring element, which acts like a probe.
            arr: A 2d or 3d array that represents a design pattern.

        Returns:
            A boolean that indicates whether the difference between opening and closing happens at the regions that exclude the borders between solid and void regions, with the edge regions specified by `margin_size` disregarded.
        """
        closing = heaviside_close(arr, diameter, pixel_size)
        close_open_diff = heaviside_open(arr, diameter, pixel_size) ^ closing
        interior = closing ^ _get_border(arr, direction="both")
        interior_diff = close_open_diff & interior
        if margin_size != None:
            interior_diff = _trim(interior_diff, margin_size, pixel_size)
        return interior_diff.any()

    min_len, _ = _search([short_pixel_side, short_entire_side],
                         min(pixel_size),
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

    if phys_size == None:
        phys_size = arr.shape
    elif isinstance(phys_size, np.ndarray) or isinstance(
            phys_size, list) or isinstance(phys_size, tuple):
        phys_size = np.squeeze(phys_size)
        phys_size = phys_size[
            phys_size.nonzero()]  # keep nonzero elements only
    elif isinstance(phys_size, float) or isinstance(phys_size, int):
        phys_size = np.array([phys_size])
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
        A tuple with two elements. The first is a float that represents the search result. The second is a Boolean, which is True if the search indeed happens, False if the condition of starting search is not satisfied in the beginning.

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
        return args[2], True
    elif not function(args[0]) and not function(args[2]):
        return args[2], False
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


def _kernel_pad(arr, pad_to):
    """
    Complete the kernel and pad it to the given size.

    Args:
        arr: A 2d or 3d array that represents roughly a quarter of the complete kernel.
        pad_to: An array that represents the size to be padded to, which must have the same shape as arr and contain integers only.
        
    Returns:
        A 2d or 3d padded array.

    Raises:
        AssertionError: If the input array is not 2d or 3d. 
    """
    pad_size = pad_to - 2 * np.array(arr.shape) + 1

    if arr.ndim == 2:
        out = np.concatenate(
            (np.concatenate((arr, np.zeros(
                (pad_size[0], arr.shape[1])), np.flipud(
                    arr[1:, :]))), np.zeros((pad_to[0], pad_size[1])),
             np.concatenate((np.fliplr(
                 arr[:, 1:]), np.zeros(
                     (pad_size[0], arr.shape[1] - 1)), np.flip(arr[1:, 1:])))),
            axis=1)
    elif arr.ndim == 3:
        upper = np.concatenate(
            (np.concatenate(
                (arr, np.zeros((pad_size[0], arr.shape[1], arr.shape[2])),
                 np.flip(arr[1:, :, :],
                         0))), np.zeros(
                             (pad_to[0], pad_size[1], arr.shape[2])),
             np.concatenate(
                 (np.flip(arr[:, 1:, :], 1),
                  np.zeros((pad_size[0], arr.shape[1] - 1, arr.shape[2])),
                  np.flip(arr[1:, 1:, :], (0, 1))))),
            axis=1)
        middle = np.zeros((pad_to[0], pad_to[1], pad_size[2]))
        lower = np.concatenate(
            (np.concatenate(
                (np.flip(arr[:, :, 1:], 2),
                 np.zeros((pad_size[0], arr.shape[1], arr.shape[2] - 1)),
                 np.flip(arr[1:, :, 1:], (0, 2)))),
             np.zeros((pad_to[0], pad_size[1], arr.shape[2] - 1)),
             np.concatenate(
                 (np.flip(arr[:, 1:, 1:], (1, 2)),
                  np.zeros((pad_size[0], arr.shape[1] - 1, arr.shape[2] - 1)),
                  np.flip(arr[1:, 1:, 1:])))),
            axis=1)

        out = np.concatenate((upper, middle, lower), axis=2)
    else:
        raise AssertionError(
            "The function is not implemented for so many or so few dimensions."
        )

    return out


def _convolution(arr, kernel):
    """
    Convolution between the kernel and the input array.

    Args:
        arr: A 2d or 3d array that represents a design pattern.
        kernel: A portion of the kernel in the function cylindrical_filter.

    Returns:
        An array that has the same dimension as the input array but has a larger size due to padding.
    """
    arr_sh = np.array(arr.shape)
    ker_sh = np.array(kernel.shape)
    npad = *((s, s) for s in arr_sh + ker_sh),

    # pad the kernel and the input array to avoid circular convolution and to ensure boundary conditions
    arr = np.pad(arr, pad_width=npad, mode='edge')
    kernel = _kernel_pad(kernel, arr_sh * 3 + ker_sh * 2)
    kernel = np.squeeze(kernel) / np.sum(kernel)  # normalize the kernel

    K, A = np.fft.fftn(kernel), np.fft.fftn(
        arr)  # transform to frequency domain for fast convolution
    arr_out = np.real(np.fft.ifftn(K * A))  # convolution

    return _center(
        arr_out, arr_sh + ker_sh * 2
    )  # padding is not totally removed at this stage in case of unwanted influence from boundaries


def _cylindrical_filter(arr, diameter, pixel_size):
    """
    Cylindrical filter.

    Args:
        arr: A 2d or 3d array that represents a design pattern.
        diameter: A positive float that represents the diameter of the cylindrical filter.
        pixel_size: A tuple that represents the physical size of one pixel in the design pattern.

    Returns:
        An array of floats.

    Raises:
        AssertionError: If the input array is not 2d or 3d, or if the pixel size and the dimension of the input array do not match.
    """

    assert arr.ndim == len(
        pixel_size
    ), 'The pixel size and the dimension of the input array do not match.'

    x_tick = np.arange(0, diameter / 2, pixel_size[0])
    y_tick = np.arange(0, diameter / 2, pixel_size[1])

    if len(pixel_size) == 2:
        X, Y = np.meshgrid(x_tick, y_tick, sparse=True, indexing='ij')
        kernel = X**2 + Y**2 < diameter**2 / 4
    elif len(pixel_size) == 3:
        z_tick = np.arange(0, diameter / 2, pixel_size[2])
        X, Y, Z = np.meshgrid(x_tick,
                              y_tick,
                              z_tick,
                              sparse=True,
                              indexing='ij')
        kernel = X**2 + Y**2 + Z**2 < diameter**2 / 4
    else:
        raise AssertionError("Only 2d or 3d is supported.")

    return _convolution(arr, kernel)


def _heaviside_erode(arr: np.ndarray,
                     diameter: float,
                     pixel_size: Tuple[float, ...],
                     proj_strength: float = 1e6) -> np.ndarray:
    """
    Heaviside erosion.

    Args:
        arr: A 2d or 3d array that represents a design pattern.
        diameter: A positive float that represents the diameter of the structuring element.
        pixel_size: A tuple that represents the physical size of one pixel in the design pattern.
        proj_strength: A float that represents the projection strength relevant to binarization.

    Returns:
        A Boolean array that has a larger size than the input array due to padding.
    """

    filtered = _cylindrical_filter(arr, diameter, pixel_size)
    projected = np.exp(
        -proj_strength *
        (1 - filtered)) + np.exp(-proj_strength) * (1 - filtered)
    return projected > threshold  # convert to a Boolean array


def _heaviside_dilate(arr: np.ndarray,
                      diameter: float,
                      pixel_size: Tuple[float, ...],
                      proj_strength: float = 1e6) -> np.ndarray:
    """
    Heaviside dilation.

    Args:
        arr: A 2d or 3d array that represents a design pattern.
        diameter: A positive float that represents the diameter of the structuring element.
        pixel_size: A tuple that represents the physical size of one pixel in the design pattern.
        proj_strength: A float that represents the projection strength relevant to binarization.

    Returns:
        A Boolean array that has a larger size than the input array due to padding.
    """
    filtered = _cylindrical_filter(arr, diameter, pixel_size)
    projected = 1 - np.exp(
        -proj_strength * filtered) + np.exp(-proj_strength) * filtered
    return projected > threshold  # convert the result to a Boolean array


def heaviside_open(arr: np.ndarray,
                   diameter: float,
                   pixel_size: Tuple[float, ...],
                   proj_strength: float = 1e6) -> np.ndarray:
    """
    Heaviside opening, which is erosion followed by dilation.

    Args:
        arr: A 2d or 3d array that represents a design pattern.
        diameter: A positive float that represents the diameter of the structuring element.
        pixel_size: A tuple that represents the physical size of one pixel in the design pattern.
        proj_strength: A float that represents the projection strength relevant to binarization.

    Returns:
        A Boolean array that has the same size as the input array.
    """

    he = _heaviside_erode(arr, diameter, pixel_size, proj_strength)
    hdhe = _heaviside_dilate(he, diameter, pixel_size, proj_strength)
    return _center(hdhe, arr.shape)  # remove padding


def heaviside_close(arr: np.ndarray,
                    diameter: float,
                    pixel_size: Tuple[float, ...],
                    proj_strength: float = 1e6) -> np.ndarray:
    """
    Heaviside closing, which is dilation followed by erosion.

    Args:
        arr: A 2d or 3d array that represents a design pattern.
        diameter: A positive float that represents the diameter of the structuring element.
        pixel_size: A tuple that represents the physical size of one pixel in the design pattern.
        proj_strength: A float that represents the projection strength relevant to binarization.

    Returns:
        A Boolean array that has the same size as the input array.
    """

    hd = _heaviside_dilate(arr, diameter, pixel_size, proj_strength)
    hehd = _heaviside_erode(hd, diameter, pixel_size, proj_strength)
    return _center(hehd, arr.shape)  # remove padding


def _get_border(arr, direction="in"):
    """
    Get inner borders, outer borders, or union of both inner and outer borders of solid regions.

    Args:
        arr: A 2d or 3d array that represents a design pattern.
        direction: A string that can be "in", "out", or "both" to indicate inner borders, outer borders, and union of inner and outer borders.

    Returns:
        A Boolean array in which all True elements are at and only at borders.

    Raises:
        AssertionError: If the option provided to `direction` is not "in", "out", or "both".
    """

    pixel_size = (1, ) * arr.ndim
    diameter = 2.01  # With this pixel size and diameter, the resulting structuring element has the shape of a plus sign.

    if direction == "in":  # inner borders of solid regions
        eroded = _heaviside_erode(arr, diameter, pixel_size)
        eroded = _center(eroded, arr.shape)
        return eroded ^ arr
    elif direction == "out":  # outer borders of solid regions
        dilated = _heaviside_dilate(arr, diameter, pixel_size)
        eroded = _center(dilated, arr.shape)
        return dilated ^ arr
    elif direction == "both":  # union of inner and outer borders of solid regions
        eroded = _heaviside_erode(arr, diameter, pixel_size)
        dilated = _heaviside_dilate(arr, diameter, pixel_size)
        return _center(dilated ^ eroded, arr.shape)
    else:
        raise AssertionError(
            "The direction at the border can only be in, out, or both.")


def _trim(arr, margin_size, pixel_size):
    """
    Obtain the trimmed array with marginal regions discarded.

    Args:
        arr: A 1d, 2d, or 3d array that represents a design pattern.
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
    elif margin_dim == 3:
        return arr[margin_number[0][0]:-margin_number[0][1],
                   margin_number[1][0]:-margin_number[1][1],
                   margin_number[2][0]:-margin_number[2][1]]
    else:
        AssertionError("The input array has too many dimensions.")


def _get_pixel_size(arr, phys_size):
    """
    Compute the physical size of a single pixel.

    Args:
        arr: An array that represents a design pattern.
        phys_size: A tuple that represents the physical size of the design pattern.

    Returns:
        An array that represents the physical size of a single pixel.
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


def _center(arr, newshape):
    """
    Obtain the center portion at an input array. The shape of the returned array is newshape. 
    Borrowed from scipy:
        https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L263-L270

    Args:
        arr: An input array.
        newshape: A tuple that represents the shape of the desired center portion.

    Returns:
        An array with the shape specified newshape. This array is the center portion of the input array.
    """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]
