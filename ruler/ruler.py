import numpy as np
import cv2 as cv

def ruler_initialize(arr,phys_size,margin_size):
    phys_size = np.array(phys_size)[np.squeeze(np.array(phys_size)).nonzero()] # physical size
    assert np.squeeze(arr).ndim == len(phys_size), 'Input array and physical size must match.'
    assert np.squeeze(arr).ndim == 2, 'Only 2d patterns are accepted.'

    '''
    Users sometimes need to discard certain marginal regions, the withds of which are specified by margin_size.
    The first pair of elements correspond to the first dimension, the second pair of elements correspond to the second dimension,
    and the third pair of elements, if there are, correspond to the third dimension.
    '''
    margin_size = np.reshape(margin_size,(-1,2))
    return phys_size, margin_size

def minimum_length(arr,phys_size,margin_size=np.zeros((3,2)),threshold=0.5,len_arr=None):
    '''
    Compute the minimum length scale in a design pattern. The design pattern is arr.
    The physical size is phys_size. The size of borders that need to be discarded is margin_size.
    The binary values of pixels are thresholded by threshold.
    '''

    phys_size, margin_size = ruler_initialize(arr,phys_size,margin_size)
    dims = len(phys_size) # dimension, should be 2 or 3
    arr = binarize(arr,threshold)
    margin_number = margin(arr,phys_size,margin_size)
    pixel_size = get_pixel_size(arr,phys_size)

    if np.array(len_arr).any(): # search the minimum length scale within a length array "len_arr"
        diameter_list = sorted(list(np.abs(len_arr)/2))
        for diameter in diameter_list:
            kernel = get_structuring_element(diameter,pixel_size)
            # difference between open and close
            diff_image = \
            abs(cv.morphologyEx(arr,cv.MORPH_OPEN,kernel).astype(np.int8)-cv.morphologyEx(arr, cv.MORPH_CLOSE, kernel).astype(np.int8))

            # number of interior pixels
            pixel_in = interior_pixel_count(diff_image,margin_number,dims)

            if pixel_in>0:
                return diameter

        print("The minimum length scale is not in this array of lengths.")
        return

    else: # find the minimum length scale via binary search if "len_arr" is not provided
        diameter = min(phys_size) # maximum meaningful filter radius
        kernel = get_structuring_element(diameter,pixel_size)
        diff_image = \
        abs(cv.morphologyEx(arr,cv.MORPH_OPEN,kernel).astype(np.int8)-cv.morphologyEx(arr,cv.MORPH_CLOSE,kernel).astype(np.int8))
        pixel_in = interior_pixel_count(diff_image,margin_number,dims) 

        if pixel_in>0:
            diameters = [0,diameter/2,diameter]
            while abs(diameters[0]-diameters[2])>min(pixel_size):
                diameter = diameters[1]
                kernel = get_structuring_element(diameter,pixel_size)
                diff_image = \
                abs(cv.morphologyEx(arr,cv.MORPH_OPEN,kernel).astype(np.int8)-cv.morphologyEx(arr, cv.MORPH_CLOSE, kernel).astype(np.int8))
                pixel_in = interior_pixel_count(diff_image,margin_number,dims)

                if pixel_in==0: diameters[0],diameters[1] = diameter,(diameter+diameters[2])/2 # radius is too small
                else: diameters[1],diameters[2] = (diameter+diameters[0])/2,diameter # radius is still large

            return diameters[0]

        else: # min(phys_size) is not a good starting diameter of the binary search
            diameter_initial,pixel_in_initial = diameter/1.5,0 # decrease the diameter
            # search a starting radius until interior pixels emerge or the diameter is unacceptably small
            while pixel_in_initial==0 and diameter_initial>min(pixel_size):
                diameter_initial /= 1.5
                kernel = get_structuring_element(diameter_initial,pixel_size)
                diff_image_initial = \
                abs(cv.morphologyEx(arr,cv.MORPH_OPEN,kernel).astype(np.int8)-cv.morphologyEx(arr, cv.MORPH_CLOSE, kernel).astype(np.int8))
                pixel_in_initial = interior_pixel_count(diff_image_initial,margin_number,dims)

            if pixel_in_initial>0: # start the binary search
                diameters = [0,diameter/2,diameter]
                while abs(diameters[0]-diameters[2])>min(pixel_size):
                    diameter = diameters[1]
                    kernel = get_structuring_element(diameter,pixel_size)
                    diff_image = abs(cv.morphologyEx(arr,cv.MORPH_OPEN,kernel)-cv.morphologyEx(arr, cv.MORPH_CLOSE, kernel))
                    pixel_in = interior_pixel_count(diff_image,margin_number,dims)

                    if pixel_in==0: diameters[0],diameters[1] = diameter,(diameter+diameters[2])/2 # radius is too small
                    else: diameters[1],diameters[2] = (diameter+diameters[0])/2,radius # radius is still large
                return diameters[0]

            else: # pixel_in_initial==0, fail to find a starting radius
                print("The minimum length scale is at least ", min(pixel_size))
                return

def get_structuring_element(diameter,pixel_size):
    se_shape = np.array(np.round(diameter/pixel_size),dtype=int)
    rounded_size = np.round(diameter/pixel_size)*pixel_size
    if se_shape[0]==0: x_tick = [0]
    else: x_tick = np.linspace(-rounded_size[0]/2,rounded_size[0]/2,se_shape[0])
    if se_shape[1]==0: y_tick = [0]
    else: y_tick = np.linspace(-rounded_size[1]/2,rounded_size[1]/2,se_shape[1])

    if len(pixel_size) == 2:
        X, Y = np.meshgrid(x_tick, y_tick, sparse=True, indexing='ij') # grid over the entire design region
        structuring_element = X**2+Y**2 <= diameter**2/4
    else:
        raise AssertionError("Function for this dimension is not implemented!")

    return np.array(structuring_element,dtype=np.uint8)

def margin(arr,phys_size,margin_size):
    # compute the numbers of pixels corresponding to the marginal widths

    arr = np.squeeze(arr)
    margin_number = margin_size[0,:]/phys_size[0]*arr.shape[0]

    for dim_idx in range(1,len(phys_size)):
        margin_number = np.vstack((margin_number,margin_size[dim_idx,:]/phys_size[dim_idx]*arr.shape[dim_idx]))
    margin_number = np.round(margin_number).astype(int) # numbers of pixels of marginal regions

    assert (margin_number>=0).all(), 'Margin widths should be nonnegative!'
    assert (np.array(arr.shape)-np.sum(margin_number,axis=1)>=3).all(), 'Too wide margin or too narrow design region!'

    for ii in range(margin_number.shape[0]):
        for jj in range(margin_number.shape[1]):
            if margin_number[ii,jj]==0:
                margin_number[ii,jj] = 1 # minimum possible margin_number

    return margin_number

def get_pixel_size(arr,phys_size):
    squeeze_shape = np.array(np.squeeze(arr).shape)
    return phys_size/squeeze_shape # sizes of a pixel along all finite-thickness directions

def binarize(arr,demarcation=0.5):
    arr_normalized = (arr-min(arr.flatten()))/(max(arr.flatten())-min(arr.flatten())) # normalize the data of the array
    arr_binarized = np.sign(arr_normalized-demarcation)/2+0.5 # binarize the data of the array with the threshold demarcation=0.5
    return np.array(arr_binarized,dtype=np.uint8)

def guarantee_2or3d(arr):
    arr = np.squeeze(arr)
    if arr.ndim == 2 or arr.ndim == 3:
        arr_out = arr
    elif arr.ndim == 1:
        arr_out = np.expand_dims(arr, axis=(1, 2)) 
    elif arr.ndim == 0:
        arr_out = np.expand_dims(arr, axis=(0, 1, 2))   
    else:
        raise AssertionError("Too many dimensions!")
    return arr_out

def interior_pixel_count(arr,margin_number=np.ones((2,2),dtype=int),dims=2):
    # return the number of interior pixels with nonzero values

    pixel_int = 0 # initialize before counting
    arr = np.squeeze(arr)
    row_begin, row_end = margin_number[0,0], arr.shape[0]-margin_number[0,1]
    column_begin,column_end = margin_number[1,0], arr.shape[1]-margin_number[1,1]

    if dims==2:
        selector = arr[row_begin:row_end,column_begin:column_end] * \
                   arr[row_begin-1:row_end-1,column_begin:column_end] * arr[row_begin+1:row_end+1,column_begin:column_end] * \
                   arr[row_begin:row_end,column_begin-1:column_end-1] * arr[row_begin:row_end,column_begin+1:column_end+1]

    else:
        raise AssertionError("Function for this dimension is not implemented!")

    return np.sum(selector)