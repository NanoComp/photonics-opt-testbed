import numpy as np

def ruler_initialize(arr,phys_size,margin_size):
    phys_size = np.array(phys_size)[np.squeeze(np.array(phys_size)).nonzero()] # physical size
    assert np.squeeze(arr).ndim == len(phys_size), 'Input array and physical size must match!'

    '''
    Users sometimes need to discard certain marginal regions, the withds of which are specified by margin_size.
    The first pair of elements correspond to the first dimension, the second pair of elements correspond to the second dimension,
    and the third pair of elements, if there are, correspond to the third dimension.
    '''
    margin_size = np.reshape(margin_size,(-1,2))
    return phys_size, margin_size

def minimum_length(arr,phys_size,margin_size=np.zeros((3,2)),threshold=0.5,proj_strength=10**6,len_arr=None):
    '''
    Compute the minimum length scale in a design pattern. The design pattern is arr.
    The physical size is phys_size. The size of borders that need to be discarded is margin_size.
    The binary values of pixels are thresholded by threshold.
    The projection strength in some morphological operations is proj_strength.
    '''

    phys_size, margin_size = ruler_initialize(arr,phys_size,margin_size)
    dims = len(phys_size) # dimension, should be 2 or 3
    arr = binarize(arr,threshold)
    margin_number = margin(arr,phys_size,margin_size)

    if np.array(len_arr).any(): # search the minimum length scale within a length array "len_arr"
        radius_list = sorted(list(np.abs(len_arr)/2))
        for radius in radius_list:
            # difference between open and close
            diff_image = abs(open_operator(arr,radius,phys_size,proj_strength)-close_operator(arr,radius,phys_size,proj_strength))

            # number of interior pixels
            pixel_in = interior_pixel_count(diff_image,margin_number,dims,threshold)

            if pixel_in>0:
                return radius*2

        print("The minimum length scale is not in this array of lengths.")
        return

    else: # find the minimum length scale via binary search if "len_arr" is not provided
        radius_ub = min(phys_size)/2 # maximum meaningful filter radius
        diff_image_ub = abs(open_operator(arr,radius_ub,phys_size,proj_strength)-close_operator(arr,radius_ub,phys_size,proj_strength))
        pixel_in_ub = interior_pixel_count(diff_image_ub,margin_number,dims,threshold) 

        if pixel_in_ub>0:
            radii = [0,radius_ub/2,radius_ub]
            while abs(radii[0]-radii[2])>min(pixel_size(arr,phys_size))/2:
                radius = radii[1]
                diff_image = abs(open_operator(arr,radius,phys_size,proj_strength)-close_operator(arr,radius,phys_size,proj_strength))
                pixel_in = interior_pixel_count(diff_image,margin_number,dims,threshold)

                if pixel_in==0: radii[0],radii[1] = radius,(radius+radii[2])/2 # radius is too small
                else: radii[1],radii[2] = (radius+radii[0])/2,radius # radius is still large

            return radii[2]*2

        else: # radius_ub is not a good starting radius of the binary search
            radius_initial,pixel_in_initial = radius_ub/1.5,0 # decrease the radius
            # search a starting radius until interior pixels emerge or the radius is unacceptably small
            while pixel_in_initial==0 and radius_initial>min(pixel_size(arr,phys_size))/2:
                radius_initial /= 1.5
                diff_image_initial = \
                abs(open_operator(arr,radius_initial,phys_size,proj_strength)-close_operator(arr,radius_initial,phys_size,proj_strength))
                pixel_in_initial = interior_pixel_count(diff_image_initial,margin_number,dims,threshold)

            if pixel_in_initial>0: # start the binary search
                radii = [0,radius_initial/2,radius_initial]
                while abs(radii[0]-radii[2])>min(pixel_size(arr,phys_size))/2:
                    radius = radii[1]
                    diff_image = abs(open_operator(arr,radius,phys_size,proj_strength)-close_operator(arr,radius,phys_size,proj_strength))
                    pixel_in = interior_pixel_count(diff_image,margin_number,dims,threshold)
                    if pixel_in==0: radii[0],radii[1] = radius,(radius+radii[2])/2
                    else: radii[1],radii[2] = (radius+radii[0])/2,radius
                return radii[2]*2
            else: # pixel_in_initial==0, fail to find a starting radius
                print("The minimum length scale is at least ", radius_ub*2)
                return

def proper_pad(arr,pad_to):
    '''
    arr: Input array. Must be 2D.
    pad_to: int. Total size to be padded to.
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
        raise AssertionError("Function for this dimension is not implemented!")

    return out
            
def simple_filter(arr,kernel):
    arr_sh = arr.shape
    npad = *((s, s) for s in arr_sh),

    # pad the kernel and the input array to avoid circular convolution and to ensure boundary conditions
    kernel = proper_pad(kernel,np.array(arr_sh)*3)
    kernel = np.squeeze(kernel) / np.sum(kernel) # Normalize the kernel
    arr = np.pad(arr,pad_width=npad, mode='edge')

    K, A = np.fft.fftn(kernel), np.fft.fftn(arr) # transform to frequency domain for fast convolution
    arr_out = np.real(np.fft.ifftn(K * A)) # convolution

    return center(arr_out,arr_sh) # Remove all the extra padding

def cylindrical_filter(arr,radius,phys_size):
    arr = guarantee_2or3d(arr)

    x_tick = np.arange(0,phys_size[0]/2,phys_size[0]/arr.shape[0])
    y_tick = np.arange(0,phys_size[1]/2,phys_size[1]/arr.shape[1])

    if len(phys_size) == 2:
        X, Y = np.meshgrid(x_tick, y_tick, sparse=True, indexing='ij') # grid over the entire design region
        kernel = X**2+Y**2 < radius**2
    elif len(phys_size) == 3:
        z_tick = np.arange(0,phys_size[2]/2,phys_size[2]/arr.shape[2])
        X, Y, Z = np.meshgrid(x_tick, y_tick, z_tick, sparse=True, indexing='ij')
        kernel = X**2+Y**2+Z**2 < radius**2
    else:
        raise AssertionError("Function for this dimension is not implemented!")

    return simple_filter(arr,kernel) # Filter the input array

def heaviside_erosion(arr,radius,phys_size,proj_strength):
    arr_hat = cylindrical_filter(arr,radius,phys_size)
    return np.exp(-proj_strength*(1-arr_hat)) + np.exp(-proj_strength)*(1-arr_hat)

def heaviside_dilation(arr,radius,phys_size,proj_strength):
    arr_hat = cylindrical_filter(arr,radius,phys_size)
    return 1 - np.exp(-proj_strength*arr_hat) + np.exp(-proj_strength)*arr_hat

def open_operator(arr,radius,phys_size,proj_strength):
    # erosion and then dilation

    he = heaviside_erosion(arr,radius,phys_size,proj_strength)
    hdhe = heaviside_dilation(he,radius,phys_size,proj_strength)
    return hdhe

def close_operator(arr,radius,phys_size,proj_strength):
    # dilation and then erosion

    hd = heaviside_dilation(arr,radius,phys_size,proj_strength)
    hehd = heaviside_erosion(hd,radius,phys_size,proj_strength)
    return hehd

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

def pixel_size(arr,phys_size):
    squeeze_shape = np.array(np.squeeze(arr).shape)
    return phys_size/squeeze_shape # sizes of a pixel along all finite-thickness directions

def binarize(arr,demarcation=0.5):
    arr_normalized = (arr-min(arr.flatten()))/(max(arr.flatten())-min(arr.flatten())) # normalize the data of the array
    arr_binarized = np.sign(arr_normalized-demarcation)/2+0.5 # binarize the data of the array with the threshold demarcation=0.5
    return arr_binarized

def adjacency(index): # return adjacent indices
    if len(index) == 2: # 2d
        return [index[0],index[0],index[0],index[0]-1,index[0]+1],[index[1],index[1]-1,index[1]+1,index[1],index[1]]
    elif len(index) == 3: # 3d
        return [index[0],index[0],index[0],index[0],index[0],index[0]-1,index[0]+1],[
            index[1],index[1],index[1],index[1]-1,index[1]+1,index[1],index[1]],[
            index[2],index[2]-1,index[2]+1,index[2],index[2],index[2],index[2]]
    else:
        raise AssertionError("Function for this dimension is not implemented!")

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

def interior_pixel_count(arr,margin_number=np.ones((2,2),dtype=int),dims=2,threshold=0.5):
    # return the number of interior pixels with nonzero values

    pixel_int = 0 # initialize before counting
    arr = np.squeeze(arr)
    arr = (np.sign(arr-threshold)+1)*0.5
    row_begin, row_end = margin_number[0,0], arr.shape[0]-margin_number[0,1]
    column_begin,column_end = margin_number[1,0], arr.shape[1]-margin_number[1,1]

    if dims==2:
        selector = arr[row_begin:row_end,column_begin:column_end] * \
                   arr[row_begin-1:row_end-1,column_begin:column_end] * arr[row_begin+1:row_end+1,column_begin:column_end] * \
                   arr[row_begin:row_end,column_begin-1:column_end-1] * arr[row_begin:row_end,column_begin+1:column_end+1]

    elif dims==3:
        page_begin, page_end = margin_number[2,0], arr.shape[2]-margin_number[2,1]
        selector = arr[row_begin:row_end,column_begin:column_end,page_begin:page_end] * \
                   arr[row_begin-1:row_end-1,column_begin:column_end,page_begin:page_end] * \
                   arr[row_begin+1:row_end+1,column_begin:column_end,page_begin:page_end] * \
                   arr[row_begin:row_end,column_begin-1:column_end-1,page_begin:page_end] * \
                   arr[row_begin:row_end,column_begin+1:column_end+1,page_begin:page_end] * \
                   arr[row_begin:row_end,column_begin:column_end,page_begin-1:page_end-1] * \
                   arr[row_begin:row_end,column_begin:column_end,page_begin+1:page_end+1]

    else:
        raise AssertionError("Function for this dimension is not implemented!")

    return np.sum(selector)