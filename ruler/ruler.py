import numpy as np

solid_indicator = ['solid','solids','black','s','b','true','1',True,1]
void_indicator = ['invert','void','white','v','w','false','0',True,0]
minimum_indicator = ['minimum','minimal','min']
both_indicator = ['both','two','2',2]

def ruler_initialize(arr,phys_size,margin_size):
    if isinstance(phys_size,np.ndarray):
        phys_size = phys_size[np.squeeze(phys_size).nonzero()] # physical size
    elif isinstance(phys_size,list) or isinstance(phys_size,tuple):
        phys_size = np.array(phys_size)[np.squeeze(np.array(phys_size)).nonzero()]
    elif isinstance(phys_size,float) or isinstance(phys_size,int):
        phys_size = np.array([phys_size])
    else:
        AssertionError("Invalid format of the physical size.")

    dims = len(phys_size) # dimension, should be 1, 2 or 3
    assert np.squeeze(arr).ndim == dims, 'The physical size and the dimension of the input array do not match.'

    '''
    Users sometimes need to discard certain marginal regions, the withds of which are specified by margin_size.
    The first pair of elements correspond to the first dimension, the second pair of elements correspond to the second dimension,
    and the third pair of elements, if there are, correspond to the third dimension.
    '''
    margin_size = np.reshape(margin_size,(-1,2))
    if len(margin_size[:,0])>dims:
        margin_size = margin_size[0:dims,:]
        print('Warning: The row number of margin_size is larger than the dimension; The redundant row(s) will be discarded.')
    return phys_size, margin_size


def minimum_length(arr,phys_size,margin_size=np.zeros((2,2)),measure_region='solid',pad_mode=None):
    '''
    Compute the minimum length scale in a design pattern. The design pattern is 'arr'.
    The physical size is 'phys_size'. The size of borders that need to be discarded is 'margin_size'.
    '''

    phys_size, margin_size = ruler_initialize(arr,phys_size,margin_size)
    shorter_entire_side = min(phys_size) # shorter side of the entire design region
    all_pixel_sides = pixel_size(arr,phys_size)
    shorter_pixel_side = min(all_pixel_sides) # shorter side of a pixel
    dims = len(phys_size) # dimension, should be 2 or 3

    '''
    If all elements in the array are the same,
    the code simply regards the shorter side of the entire pattern as the minimum length scale,
    regardless of whether the pattern is solid or void.
    '''
    if len(np.unique(arr))==1:
        return shorter_entire_side

    '''
    Design patterns with non-binary values are thresholded by 'binarize_threshold'.
    The appearance of interior solid pixels is thresholded by 'interior_pixel_threshold'.
    The projection strength in some morphological operations is 'proj_strength'.
    '''
    binarize_threshold, interior_pixel_threshold, proj_strength = 0.5, 0.5, 10**6
    arr = binarize(arr,binarize_threshold)
    margin_number = margin(arr,phys_size,margin_size)

    if isinstance(measure_region,str):
        measure_region = measure_region.lower()

    if dims == 1:
        arr = arr[margin_number[0,0]:len(arr)-margin_number[0,1]]
        solid_min_length, void_min_length = minimum_length_1d(arr)

        if measure_region in solid_indicator:
            return solid_min_length*shorter_pixel_side
        elif measure_region in void_indicator:
            return void_min_length*shorter_pixel_side
        elif measure_region in minimum_indicator:
            return min(solid_min_length,void_min_length)*shorter_pixel_side
        elif measure_region in both_indicator:
            return solid_min_length*shorter_pixel_side, void_min_length*shorter_pixel_side
        else:
            AssertionError("Invalid argument for measure_region in the function minimum_length.")

    arr_shape = np.array(arr.shape)
    pad_number = np.ceil(arr_shape/2).astype(int)
    pad_width = *((int(np.ceil(s/2)), int(np.ceil(s/2))) for s in arr.shape),
    if isinstance(pad_mode,str): pad_mode = pad_mode.lower()

    if pad_mode==None or pad_mode=='edge':
        arr_padded = np.pad(arr,pad_width=pad_width,mode='edge')
    elif pad_mode in solid_indicator:
        arr_padded = np.pad(arr,pad_width=pad_width,constant_values=1)
    elif pad_mode in void_indicator:
        arr_padded = np.pad(arr,pad_width=pad_width,constant_values=0)
    else:
        AssertionError("Invalid pad mode.")

    def interior_pixel_number(diameter,arr,margin_number,pad_number):
        padded_shape = arr_shape+2*pad_number
        padded_size = padded_shape*all_pixel_sides
        diff_image = abs(heaviside_open(arr,diameter,padded_size,proj_strength)-arr)
        return interior_solid_pixel_count(diff_image,margin_number,pad_number,dims)

    if measure_region not in void_indicator:
        min_len_solid_padded, status = search([shorter_pixel_side,shorter_entire_side],
                                              min(pixel_size(arr,phys_size)),
                                              lambda d: interior_pixel_number(d,arr_padded,margin_number,pad_number),
                                              interior_pixel_threshold)
        min_len_solid = min_len_solid_padded

        if pad_mode==None:
            min_len_solid_unpadded, status = search([shorter_pixel_side,shorter_entire_side],
                                                    min(pixel_size(arr,phys_size)),
                                                    lambda d: interior_pixel_number(d,arr,margin_number,
                                                                                    pad_number=np.zeros(dims)),
                                                    interior_pixel_threshold)

            if abs(min_len_solid_unpadded-min_len_solid_padded)<shorter_entire_side/2:
                min_len_solid = min_len_solid_unpadded
            else:
                min_len_solid = min_len_solid_unpadded/2

    if measure_region not in solid_indicator:
        min_len_void_padded, status = search([shorter_pixel_side,shorter_entire_side],
                                              min(pixel_size(arr,phys_size)),
                                              lambda d: interior_pixel_number(d,1-arr_padded,margin_number,pad_number),
                                              interior_pixel_threshold)
        min_len_void = min_len_void_padded

        if pad_mode==None:
            min_len_void_unpadded, status = search([shorter_pixel_side,shorter_entire_side],
                                                    min(pixel_size(arr,phys_size)),
                                                    lambda d: interior_pixel_number(d,1-arr,margin_number,
                                                                                    pad_number=np.zeros(dims)),
                                                    interior_pixel_threshold)

            if abs(min_len_void_unpadded-min_len_void_padded)<shorter_entire_side/2:
                min_len_void = min_len_void_unpadded
            else:
                min_len_void = min_len_void_unpadded/2
  
    if measure_region in solid_indicator:
        return min_len_solid
    elif measure_region in void_indicator:
        return min_len_void
    elif measure_region in minimum_indicator:
        return min(min_len_solid,min_len_void)
    elif measure_region in both_indicator:
        return min_len_solid, min_len_void
    else:
        AssertionError("Invalid argument for measure_region in the function minimum_length.")


def search(arg_range,arg_threshold,function,function_threshold): # binary search
    args = [min(arg_range),(min(arg_range)+max(arg_range))/2,max(arg_range)]

    if function(args[0])<function_threshold and function(args[2])>function_threshold:
        while abs(args[0]-args[2])>arg_threshold:
            arg = args[1]
            if function(arg)<=function_threshold: args[0],args[1] = arg,(arg+args[2])/2 # radius is too small
            else: args[1],args[2] = (arg+args[0])/2,arg # radius is still large
        return args[2], True
    elif function(args[0])<=function_threshold and function(args[2])<=function_threshold:
        return args[2], False
    elif function(args[0])>function_threshold and function(args[2])>function_threshold:
        return args[0], False
    else:
        raise AssertionError("The function is not monotonically increasing.")

def minimum_length_1d(arr):
    arr = np.append(arr,1-arr[-1])
    solid_lengths,void_lengths = [],[]
    counter = 0

    for idx in range(len(arr)-1):
        counter += 1

        if arr[idx] != arr[idx+1]:
            if arr[idx] == 1:
                solid_lengths.append(counter)
            elif arr[idx] == 0:
                void_lengths.append(counter)
            else:
                raise AssertionError("This array is not normalized.")
            counter = 0

    if len(solid_lengths)>0:
        solid_min_length = min(solid_lengths)
    else:
        solid_min_length = 0

    if len(void_lengths)>0:
        void_min_length = min(void_lengths)
    else:
        void_min_length = 0

    return solid_min_length, void_min_length


def kernel_pad(arr,pad_to):
    '''
    The input array is 'arr', which must be 2d or 3d.
    The total size to be padded to is pad_to, which must be an integer.
    '''
    pad_size = pad_to-2*np.array(arr.shape)+1

    if arr.ndim == 2:
        out = np.concatenate((np.concatenate((arr, np.zeros((pad_size[0],arr.shape[1])), np.flipud(arr[1:,:]))),
                              np.zeros((pad_to[0],pad_size[1])),
                              np.concatenate((np.fliplr(arr[:,1:]), np.zeros((pad_size[0],arr.shape[1]-1)), np.flip(arr[1:,1:])))),
                             axis=1)
    elif arr.ndim == 3:
        upper = np.concatenate((np.concatenate((arr, np.zeros((pad_size[0],arr.shape[1],arr.shape[2])), np.flip(arr[1:,:,:],0))),
                                np.zeros((pad_to[0],pad_size[1],arr.shape[2])),
                                np.concatenate((np.flip(arr[:,1:,:],1), np.zeros((pad_size[0],arr.shape[1]-1,arr.shape[2])),
                                                np.flip(arr[1:,1:,:],(0,1))))),axis=1)
        middle = np.zeros((pad_to[0],pad_to[1],pad_size[2]))
        lower = np.concatenate((np.concatenate((np.flip(arr[:,:,1:],2), np.zeros((pad_size[0],arr.shape[1],arr.shape[2]-1)),
                                                np.flip(arr[1:,:,1:],(0,2)))),
                                np.zeros((pad_to[0],pad_size[1],arr.shape[2]-1)),
                                np.concatenate((np.flip(arr[:,1:,1:],(1,2)), np.zeros((pad_size[0],arr.shape[1]-1,arr.shape[2]-1)),
                                                np.flip(arr[1:,1:,1:])))),axis=1)

        out = np.concatenate((upper,middle,lower),axis=2)
    else:
        raise AssertionError("The function for this dimension is not implemented.")

    return out


def convolution(arr,kernel):
    arr_sh = arr.shape
    npad = *((s, s) for s in arr_sh),

    # pad the kernel and the input array to avoid circular convolution and to ensure boundary conditions
    kernel = kernel_pad(kernel,np.array(arr_sh)*3)
    kernel = np.squeeze(kernel) / np.sum(kernel) # normalize the kernel
    arr = np.pad(arr,pad_width=npad, mode='edge')

    K, A = np.fft.fftn(kernel), np.fft.fftn(arr) # transform to frequency domain for fast convolution
    arr_out = np.real(np.fft.ifftn(K * A)) # convolution

    return center(arr_out,arr_sh) # remove all the extra padding


def cylindrical_filter(arr,diameter,phys_size):
    arr = guarantee_2or3d(arr)

    x_tick = np.arange(0,diameter/2,phys_size[0]/arr.shape[0])
    y_tick = np.arange(0,diameter/2,phys_size[1]/arr.shape[1])

    if len(phys_size) == 2:
        X, Y = np.meshgrid(x_tick, y_tick, sparse=True, indexing='ij')
        kernel = X**2+Y**2 < diameter**2/4
    elif len(phys_size) == 3:
        z_tick = np.arange(0,phys_size[2]/2,phys_size[2]/arr.shape[2])
        X, Y, Z = np.meshgrid(x_tick, y_tick, z_tick, sparse=True, indexing='ij')
        kernel = X**2+Y**2+Z**2 < diameter**2/4
    else:
        raise AssertionError("The function for this dimension is not implemented.")

    return convolution(arr,kernel)


def heaviside_erode(arr,diameter,phys_size,proj_strength):
    arr_hat = cylindrical_filter(arr,diameter,phys_size)
    return np.exp(-proj_strength*(1-arr_hat)) + np.exp(-proj_strength)*(1-arr_hat)


def heaviside_dilate(arr,diameter,phys_size,proj_strength):
    arr_hat = cylindrical_filter(arr,diameter,phys_size)
    return 1 - np.exp(-proj_strength*arr_hat) + np.exp(-proj_strength)*arr_hat


def heaviside_open(arr,diameter,phys_size,proj_strength):
    # erosion and then dilation
    he = heaviside_erode(arr,diameter,phys_size,proj_strength)
    hdhe = heaviside_dilate(he,diameter,phys_size,proj_strength)
    return hdhe


def margin(arr,phys_size,margin_size):
    # compute the numbers of pixels corresponding to the marginal widths that need to be discarded
    arr = np.squeeze(arr)
    margin_number = margin_size[0,:].reshape(1,2)/phys_size[0]*arr.shape[0]

    for dim_idx in range(1,margin_size.shape[0]):
        margin_number = np.vstack((margin_number,margin_size[dim_idx,:]/phys_size[dim_idx]*arr.shape[dim_idx]))
        
    margin_number = np.round(margin_number).astype(int) # numbers of pixels of marginal regions

    assert (margin_number>=0).all(), 'Margin widths should be nonnegative.'
    assert (np.array(arr.shape)-np.sum(margin_number,axis=1)>=3).all(), 'Too wide margin or too narrow design region.'

    return margin_number


def pixel_size(arr,phys_size):
    squeeze_shape = np.array(np.squeeze(arr).shape)
    return phys_size/squeeze_shape # sizes of a pixel along all finite-thickness directions


def binarize(arr,demarcation=0.5):
    arr_normalized = (arr-min(arr.flatten()))/(max(arr.flatten())-min(arr.flatten())) # normalize the data of the array
    arr_binarized = arr_normalized > demarcation # binarize the data of the array with the threshold 'demarcation'
    return arr_binarized


def guarantee_2or3d(arr):
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


def center(arr, newshape):
    '''Return the center newshape portion of the array.
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


def interior_solid_pixel_count(arr,margin_number=np.zeros((2,2),dtype=int),pad_number=np.zeros(2,dtype=int),dims=2):
    # return the number of interior pixels with nonzero values

    arr = np.squeeze(arr)
    margin_number, pad_number = margin_number.astype(int), pad_number.astype(int)

    row_begin, row_end = margin_number[0,0]+1+pad_number[0], arr.shape[0]-margin_number[0,1]-1-pad_number[0]
    column_begin,column_end = 1+pad_number[1], arr.shape[1]-1-pad_number[1]

    if len(margin_number[:,0]) == 2:
        column_begin,column_end = margin_number[1,0]+1+pad_number[1], arr.shape[1]-margin_number[1,1]-1-pad_number[1]

    if dims==2:
        # select interior solid pixels
        selector = arr[row_begin:row_end,column_begin:column_end] * \
                   arr[row_begin-1:row_end-1,column_begin:column_end] * arr[row_begin+1:row_end+1,column_begin:column_end] * \
                   arr[row_begin:row_end,column_begin-1:column_end-1] * arr[row_begin:row_end,column_begin+1:column_end+1]

    elif dims==3:
        if len(margin_number[:,0]) == 3:
            page_begin, page_end = margin_number[2,0]+1+pad_number[2], arr.shape[2]-margin_number[2,1]-1-pad_number[2]
        else:
            page_begin, page_end = 1+pad_number[2], arr.shape[2]-1-pad_number[2]

        selector = arr[row_begin:row_end,column_begin:column_end,page_begin:page_end] * \
                   arr[row_begin-1:row_end-1,column_begin:column_end,page_begin:page_end] * \
                   arr[row_begin+1:row_end+1,column_begin:column_end,page_begin:page_end] * \
                   arr[row_begin:row_end,column_begin-1:column_end-1,page_begin:page_end] * \
                   arr[row_begin:row_end,column_begin+1:column_end+1,page_begin:page_end] * \
                   arr[row_begin:row_end,column_begin:column_end,page_begin-1:page_end-1] * \
                   arr[row_begin:row_end,column_begin:column_end,page_begin+1:page_end+1]

    else:
        raise AssertionError("The function for this dimension is not implemented.")

    return np.sum(selector)