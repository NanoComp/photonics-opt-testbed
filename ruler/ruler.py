import numpy as np
from typing import Tuple, List


def solid_minimum_length(
    arr: np.ndarray,
    phys_size: Tuple[float, ...],
    margin_size: List[Tuple[float, float]] = ((0, 0), (0, 0))
) -> float:
    '''
    Compute the minimum length scale of solid regions in a design pattern.

    arr: input array that represents a design pattern.
    phys_size: physical size of the design pattern.
    margin_size: physical size of borders that need to be discarded.
    '''

    phys_size, short_entire_side, pixel_size, short_pixel_side, margin_number, dims, arr = _ruler_initialize(
        arr, phys_size, margin_size)

    if dims == 1:
        arr = arr[margin_number[0, 0]:len(arr) - margin_number[0, 1]]
        solid_min_length, _ = _minimum_length_1d(arr)
        return solid_min_length * short_pixel_side

    arr_shape = np.array(arr.shape)

    # pad_number: numbers of pixels corresponding to the extra widths padded to the design pattern.
    pad_number = np.ceil(arr_shape / 2).astype(int)
    pad_width = *((int(np.ceil(s / 2)), int(np.ceil(s / 2)))
                  for s in arr.shape),
    arr_padded = np.pad(arr, pad_width=pad_width, mode='edge')
    padded_shape = arr_shape + 2 * pad_number
    padded_size = padded_shape * pixel_size

    def _interior_pixel_number(diameter, arr):
        '''
        diameter: diameter of the structuring element, which acts like a probe
        arr: 2d or 3d input array.
        margin_number: numbers of pixels corresponding to the marginal widths that need to be discarded.
        pad_number: numbers of pixels corresponding to the extra widths padded to the design pattern.
        dual: if true, calculate the difference between opening and closing;
              if not true, calculate the difference between opening and the input array.
        '''
        diff_image = abs(heaviside_open(arr, diameter, padded_size) - arr)
        return _interior_solid_pixel_count(diff_image, margin_number,
                                           pad_number, dims)

    min_len, status = _search([short_pixel_side, short_entire_side],
                              min(pixel_size),
                              lambda d: _interior_pixel_number(d, arr_padded))

    return min_len


def void_minimum_length(
    arr: np.ndarray,
    phys_size: Tuple[float, ...],
    margin_size: List[Tuple[float, float]] = ((0, 0), (0, 0))
) -> float:
    '''
    Compute the minimum length scale of void regions in a design pattern.

    arr: input array that represents a design pattern.
    phys_size: physical size of the design pattern.
    margin_size: physical size of borders that need to be discarded.
    '''

    arr = 1 - arr
    return solid_minimum_length(arr, phys_size, margin_size)


def both_minimum_length(
    arr: np.ndarray,
    phys_size: Tuple[float, ...],
    margin_size: List[Tuple[float, float]] = ((0, 0), (0, 0))
) -> Tuple[float, float]:
    '''
    Compute the minimum length scale of both solid and void regions in a design pattern.

    arr: input array that represents a design pattern.
    phys_size: physical size of the design pattern.
    margin_size: physical size of borders that need to be discarded.
    '''

    return solid_minimum_length(arr, phys_size,
                                margin_size), void_minimum_length(
                                    arr, phys_size, margin_size)


def dual_minimum_length(
    arr: np.ndarray,
    phys_size: Tuple[float, ...],
    margin_size: List[Tuple[float, float]] = ((0, 0), (0, 0))
) -> float:
    '''
    For 2d or 3d design patterns, compute the minimum length scale by calculating the difference between open and close operations.
    Ideally, the result should be equal to the smaller one between solid and void minimum length scales.
    For 1d design patterns, compute this smaller one by comparing solid and void minimum length scales.

    arr: input array that represents a design pattern.
    phys_size: physical size of the design pattern.
    margin_size: physical size of borders that need to be discarded.
    '''

    phys_size, short_entire_side, pixel_size, short_pixel_side, margin_number, dims, arr = _ruler_initialize(
        arr, phys_size, margin_size)

    if dims == 1:
        arr = arr[margin_number[0, 0]:len(arr) - margin_number[0, 1]]
        solid_min_length, void_min_length = _minimum_length_1d(arr)
        return min(solid_min_length, void_min_length) * short_pixel_side

    arr_shape = np.array(arr.shape)

    # pad_number: numbers of pixels corresponding to the extra widths padded to the design pattern.
    pad_number = np.ceil(arr_shape / 2).astype(int)
    pad_width = *((int(np.ceil(s / 2)), int(np.ceil(s / 2)))
                  for s in arr.shape),
    arr_padded = np.pad(arr, pad_width=pad_width, mode='edge')
    padded_shape = arr_shape + 2 * pad_number
    padded_size = padded_shape * pixel_size

    def _interior_pixel_number(diameter, arr):
        '''
        diameter: diameter of the structuring element, which acts like a probe
        arr: 2d or 3d input array.
        margin_number: numbers of pixels corresponding to the marginal widths that need to be discarded.
        pad_number: numbers of pixels corresponding to the extra widths padded to the design pattern.
        dual: if true, calculate the difference between opening and closing;
              if not true, calculate the difference between opening and the input array.
        '''
        diff_image = abs(
            heaviside_open(arr, diameter, padded_size) -
            heaviside_close(arr, diameter, padded_size))
        return _interior_solid_pixel_count(diff_image, margin_number,
                                           pad_number, dims)

    min_len, status = _search([short_pixel_side, short_entire_side],
                              min(pixel_size),
                              lambda d: _interior_pixel_number(d, arr_padded))

    return solid_minimum_length(arr, phys_size, margin_size)


def _ruler_initialize(arr, phys_size, margin_size):
    '''
    Compute the minimum length scale in a design pattern.

    arr: input array that represents a design pattern.
    phys_size: physical size of the design pattern.
    margin_size: physical size of borders that need to be discarded.
    '''
    if isinstance(phys_size, np.ndarray):
        phys_size = np.squeeze(phys_size)
        phys_size = phys_size[phys_size.nonzero()]  # physical size
    elif isinstance(phys_size, list) or isinstance(phys_size, tuple):
        phys_size = np.squeeze(np.array(phys_size))
        phys_size = phys_size[phys_size.nonzero()]
    elif isinstance(phys_size, float) or isinstance(phys_size, int):
        phys_size = np.array([phys_size])
    else:
        AssertionError("Invalid format of the physical size.")

    dims = len(phys_size)  # dimension, should be 1, 2 or 3
    arr = np.squeeze(arr)
    assert arr.ndim == dims, 'The physical size and the dimension of the input array do not match.'
    '''
    Users sometimes need to discard certain marginal regions, the withds of which are specified by margin_size.
    The first pair of elements correspond to the first dimension, the second pair of elements correspond to the second dimension,
    and the third pair of elements, if there are, correspond to the third dimension.
    '''
    margin_size = np.reshape(margin_size, (-1, 2))
    if len(margin_size[:, 0]) > dims:
        margin_size = margin_size[0:dims, :]
        print(
            'Warning: The row number of margin_size is larger than the dimension; The redundant row(s) will be discarded.'
        )

    short_entire_side = min(
        phys_size)  # shorter side of the entire design region
    pixel_size = _get_pixel_size(arr, phys_size)
    short_pixel_side = min(pixel_size)  # shorter side of a pixel
    '''
    If all elements in the array are the same,
    the code simply regards the shorter side of the entire pattern as the minimum length scale,
    regardless of whether the pattern is solid or void.
    '''
    if len(np.unique(arr)) == 1:
        return short_entire_side

    arr = _binarize(arr, threshold=0.5)
    margin_number = _margin(arr, phys_size, np.array(margin_size))

    return phys_size, short_entire_side, pixel_size, short_pixel_side, margin_number, dims, arr


def _search(arg_range, arg_threshold, function, function_threshold=0.5):
    '''
    Binary search.

    arg_range: initial range of the argument under search.
    arg_threshold: threshold of the argument range, below which the search stops.
    function: monotonic function of the argument.
    function_threshold: dividing point in the range of the function that directs the search.
    '''
    args = [
        min(arg_range), (min(arg_range) + max(arg_range)) / 2,
        max(arg_range)
    ]

    if function(args[0]) < function_threshold and function(
            args[2]) > function_threshold:
        while abs(args[0] - args[2]) > arg_threshold:
            arg = args[1]
            if function(arg) <= function_threshold:
                args[0], args[1] = arg, (arg +
                                         args[2]) / 2  # radius is too small
            else:
                args[1], args[2] = (arg +
                                    args[0]) / 2, arg  # radius is still large
        return args[2], True
    elif function(args[0]) <= function_threshold and function(
            args[2]) <= function_threshold:
        return args[2], False
    elif function(args[0]) > function_threshold and function(
            args[2]) > function_threshold:
        return args[0], False
    else:
        raise AssertionError("The function is not monotonically increasing.")


def _minimum_length_1d(arr):
    '''
    Search the minimum lengths of solid and void segments in a 1d array.

    arr: input array that must be 1d.
    '''
    arr = np.append(arr, 1 - arr[-1])
    solid_lengths, void_lengths = [], []
    counter = 0

    for idx in range(len(arr) - 1):
        counter += 1

        if arr[idx] != arr[idx + 1]:
            if arr[idx] == 1:
                solid_lengths.append(counter)
            elif arr[idx] == 0:
                void_lengths.append(counter)
            else:
                raise AssertionError("This array is not normalized.")
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
    '''
    Complete the kernel and pad it to the given size.

    arr: input array that must be 2d or 3d.
    pad_to: total size to be padded to, which must be an integer array with the same shape as 'arr'.
    '''
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
            "The function for this dimension is not implemented.")

    return out


def _convolution(arr, kernel):
    '''
    Convolution between the kernel and the input array.

    arr: input array that must be 2d or 3d.
    kernel: kernel returned by 'cylindrical_filter'.
    '''
    arr_sh = arr.shape
    npad = *((s, s) for s in arr_sh),

    # pad the kernel and the input array to avoid circular convolution and to ensure boundary conditions
    kernel = _kernel_pad(kernel, np.array(arr_sh) * 3)
    kernel = np.squeeze(kernel) / np.sum(kernel)  # normalize the kernel
    arr = np.pad(arr, pad_width=npad, mode='edge')

    K, A = np.fft.fftn(kernel), np.fft.fftn(
        arr)  # transform to frequency domain for fast convolution
    arr_out = np.real(np.fft.ifftn(K * A))  # convolution

    return _center(arr_out, arr_sh)  # remove all the extra padding


def _cylindrical_filter(arr, diameter, phys_size):
    '''
    Cylindrical filter with a given diameter.

    arr: 2d or 3d input array corresponding to the design pattern or padded design pattern.
    diameter: positive number that specifies the diameter of the cylindrical filter.
    phys_size: physical size of the design pattern or padded design pattern.
    '''
    arr = _guarantee_2or3d(arr)

    x_tick = np.arange(0, diameter / 2, phys_size[0] / arr.shape[0])
    y_tick = np.arange(0, diameter / 2, phys_size[1] / arr.shape[1])

    if len(phys_size) == 2:
        X, Y = np.meshgrid(x_tick, y_tick, sparse=True, indexing='ij')
        kernel = X**2 + Y**2 < diameter**2 / 4
    elif len(phys_size) == 3:
        z_tick = np.arange(0, phys_size[2] / 2, phys_size[2] / arr.shape[2])
        X, Y, Z = np.meshgrid(x_tick,
                              y_tick,
                              z_tick,
                              sparse=True,
                              indexing='ij')
        kernel = X**2 + Y**2 + Z**2 < diameter**2 / 4
    else:
        raise AssertionError(
            "The function for this dimension is not implemented.")

    return _convolution(arr, kernel)


def heaviside_erode(arr: np.ndarray,
                    diameter: float,
                    phys_size: Tuple[float, ...],
                    proj_strength: float = 1e6) -> np.ndarray:
    '''
    Heaviside erosion with a given diameter.

    arr: input array, which must be 2d or 3d.
    diameter: positive number that specifies the diameter of the operation.
    phys_size: physical size corresponding to the input array 'arr'.
    proj_strength: projection strength relevant to binarization.
    '''
    arr_hat = _cylindrical_filter(arr, diameter, phys_size)
    return np.exp(-proj_strength *
                  (1 - arr_hat)) + np.exp(-proj_strength) * (1 - arr_hat)


def heaviside_dilate(arr: np.ndarray,
                     diameter: float,
                     phys_size: Tuple[float, ...],
                     proj_strength: float = 1e6) -> np.ndarray:
    '''
    Heaviside dilation with a given diameter.

    The arguments have the same meanings as those in 'heaviside_erode'.
    '''
    arr_hat = _cylindrical_filter(arr, diameter, phys_size)
    return 1 - np.exp(
        -proj_strength * arr_hat) + np.exp(-proj_strength) * arr_hat


def heaviside_open(arr: np.ndarray,
                   diameter: float,
                   phys_size: Tuple[float, ...],
                   proj_strength: float = 1e6) -> np.ndarray:
    '''
    Heaviside open, which is erosion followed by dilation.
    
    The arguments have the same meanings as those in 'heaviside_erode'.
    '''
    he = heaviside_erode(arr, diameter, phys_size, proj_strength)
    hdhe = heaviside_dilate(he, diameter, phys_size, proj_strength)
    return hdhe


def heaviside_close(arr: np.ndarray,
                    diameter: float,
                    phys_size: Tuple[float, ...],
                    proj_strength: float = 1e6) -> np.ndarray:
    '''
    Heaviside close, which is dilation followed by erosion.
    
    The arguments have the same meanings as those in 'heaviside_erode'.
    '''
    hd = heaviside_dilate(arr, diameter, phys_size, proj_strength)
    hehd = heaviside_erode(hd, diameter, phys_size, proj_strength)
    return hehd


def _margin(arr, phys_size, margin_size):
    '''
    Compute the numbers of pixels corresponding to the marginal widths that need to be discarded.

    arr: input array that represents a design pattern.
    phys_size: physical size of the design pattern.
    margin_size: physical size of borders that need to be discarded.
    '''
    arr = np.squeeze(arr)
    margin_number = margin_size[0, :].reshape(1,
                                              2) / phys_size[0] * arr.shape[0]

    for dim_idx in range(1, margin_size.shape[0]):
        margin_number = np.vstack((margin_number, margin_size[dim_idx, :] /
                                   phys_size[dim_idx] * arr.shape[dim_idx]))

    margin_number = np.round(margin_number).astype(
        int)  # numbers of pixels of marginal regions

    assert (margin_number >= 0).all(), 'Margin widths should be nonnegative.'
    assert (np.array(arr.shape)[0:len(np.sum(margin_number, axis=1))] - np.sum(margin_number, axis=1) >=
            3).all(), 'Too wide margin or too narrow design region.'

    return margin_number


def _get_pixel_size(arr, phys_size):
    '''
    Compute the size of a single pixels.

    arr: input array that represents a design pattern.
    phys_size: physical size of the design pattern.
    '''
    squeeze_shape = np.array(np.squeeze(arr).shape)
    return phys_size / squeeze_shape  # sizes of a pixel along all finite-thickness directions


def _binarize(arr, threshold=0.5):
    '''
    Binarize the input array according to threshold.

    arr: input array that represents a design pattern.
    threshold: threshold for binarization.
    '''
    arr_binarized = arr > threshold * max(
        arr.flatten()) + (1 - threshold) * min(arr.flatten())
    return arr_binarized


def _guarantee_2or3d(arr):
    '''
    Make the input array 2d or 3d if it is not.

    arr: input array that represents a design pattern.
    '''
    arr = np.squeeze(arr)
    if arr.ndim == 2 or arr.ndim == 3:
        arr_out = arr
    elif arr.ndim == 1:
        arr_out = np.expand_dims(arr, axis=(1, 2))
    elif arr.ndim == 0:
        arr_out = np.expand_dims(arr, axis=(0, 1, 2))
    else:
        raise AssertionError("The input array has too many dimensions.")
    return arr_out


def _center(arr, newshape):
    '''
    Return the center newshape portion of the array.
    Helper function that reformats the padded array of the fft filter operation.
    Borrowed from scipy:
    https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L263-L270
    '''
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _interior_solid_pixel_count(arr,
                                margin_number=np.zeros((2, 2), dtype=int),
                                pad_number=np.zeros(2, dtype=int),
                                dims=2):
    '''
    Count the number of solid interior pixels.

    arr: 2d or 3d input array corresponding to the design pattern or padded design pattern.
    margin_number: numbers of pixels corresponding to the marginal widths that need to be discarded.
    pad_number: numbers of pixels corresponding to the extra widths padded to the design pattern.
    dims: dimension of the design pattern, 2 or 3
    '''
    arr = np.squeeze(arr)
    margin_number, pad_number = margin_number.astype(int), pad_number.astype(
        int)

    row_begin, row_end = margin_number[0, 0] + 1 + pad_number[0], arr.shape[
        0] - margin_number[0, 1] - 1 - pad_number[0]
    column_begin, column_end = 1 + pad_number[1], arr.shape[
        1] - 1 - pad_number[1]

    if len(margin_number[:, 0]) == 2:
        column_begin, column_end = margin_number[1, 0] + 1 + pad_number[
            1], arr.shape[1] - margin_number[1, 1] - 1 - pad_number[1]

    if dims == 2:
        # select interior solid pixels
        selector = arr[row_begin:row_end,column_begin:column_end] * \
                   arr[row_begin-1:row_end-1,column_begin:column_end] * arr[row_begin+1:row_end+1,column_begin:column_end] * \
                   arr[row_begin:row_end,column_begin-1:column_end-1] * arr[row_begin:row_end,column_begin+1:column_end+1]

    elif dims == 3:
        if len(margin_number[:, 0]) == 3:
            page_begin, page_end = margin_number[2, 0] + 1 + pad_number[
                2], arr.shape[2] - margin_number[2, 1] - 1 - pad_number[2]
        else:
            page_begin, page_end = 1 + pad_number[2], arr.shape[
                2] - 1 - pad_number[2]

        selector = arr[row_begin:row_end,column_begin:column_end,page_begin:page_end] * \
                   arr[row_begin-1:row_end-1,column_begin:column_end,page_begin:page_end] * \
                   arr[row_begin+1:row_end+1,column_begin:column_end,page_begin:page_end] * \
                   arr[row_begin:row_end,column_begin-1:column_end-1,page_begin:page_end] * \
                   arr[row_begin:row_end,column_begin+1:column_end+1,page_begin:page_end] * \
                   arr[row_begin:row_end,column_begin:column_end,page_begin-1:page_end-1] * \
                   arr[row_begin:row_end,column_begin:column_end,page_begin+1:page_end+1]

    else:
        raise AssertionError(
            "The function for this dimension is not implemented.")

    return np.sum(selector)