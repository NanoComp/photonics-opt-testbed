import numpy as np
# Some code is copied or adapted from Meep at https://github.com/smartalecH/meep/blob/jax_rebase/python/adjoint/filters.py 

class morph:
    def __init__(self,phys_size,margin_size=np.zeros((3,2)),proj_strength=10**6):
        self.phys_size = np.array(phys_size)[np.array(phys_size).nonzero()] # physical size
        self.dims = len(self.phys_size) # dimension, should be 2 or 3
        '''
        Users sometimes need to discard certain marginal regions, the withds of which are specified by margin_size.
        The first pair of elements correspond to the first dimension, the second pair of elements correspond to the second dimension,
        and the third pair of elements, if there are, correspond to the third dimension.
        '''
        self.margin_size = np.reshape(margin_size,(-1,2))
        self.proj_strength = proj_strength # parameter for the functions heaviside_erosion and heaviside_dilation

    def cylindrical_filter(self,arr,radius):
        arr = guarantee_3d(arr)

        x_tick = np.linspace(-self.phys_size[0]/2,self.phys_size[0]/2,arr.shape[0])
        y_tick = np.linspace(-self.phys_size[1]/2,self.phys_size[1]/2,arr.shape[1])
        if self.dims == 2:
            z_tick = [0]
        elif self.dims == 3:
            z_tick = np.linspace(-self.phys_size[2]/2,self.phys_size[2]/2,arr.shape[2])
        else:
            raise AssertionError("Code for this dimension is not implemented!")
        X, Y, Z = np.meshgrid(x_tick, y_tick, z_tick, sparse=True, indexing='ij') # grid over the entire design region

        kernel = X**2+Y**2+Z**2 <= radius**2 # Calculate the kernel
        kernel = np.squeeze(kernel) / np.sum(kernel.flatten()) # Normalize the kernel
        arr_out = simple_filter(np.squeeze(arr),kernel) # Filter the input array

        return arr_out

    def heaviside_erosion(self,arr,radius):
        beta = self.proj_strength
        arr_hat = self.cylindrical_filter(arr,radius)
        
        return np.exp(-beta*(1-arr_hat)) + np.exp(-beta)*(1-arr_hat)

    def heaviside_dilation(self,arr,radius):
        beta = self.proj_strength
        arr_hat = self.cylindrical_filter(arr,radius)
        
        return 1 - np.exp(-beta*arr_hat) + np.exp(-beta)*arr_hat

    def open_operator(self,arr,radius): # erosion and then dilation
        he = self.heaviside_erosion(arr,radius)
        hdhe = self.heaviside_dilation(he,radius)
        
        return hdhe

    def close_operator(self,arr,radius): # dilation and then erosion
        hd = self.heaviside_dilation(arr,radius)
        hehd = self.heaviside_erosion(hd,radius)

        return hehd

    def margin(self,arr): # compute the numbers of pixels corresponding to the marginal widths
        arr = np.squeeze(arr)
        assert arr.ndim == self.dims, 'Input array and physical size must match!'

        margin_number = self.margin_size[0,:]/self.phys_size[0]*arr.shape[0]
        for dim_idx in range(1,self.dims):
            margin_number = np.vstack((margin_number,self.margin_size[dim_idx,:]/self.phys_size[dim_idx]*arr.shape[dim_idx]))
        margin_number = np.round(margin_number).astype(int) # numbers of pixels of marginal regions

        assert (margin_number>=0).all(), 'Margin widths should be nonnegative!'
        assert (np.array(arr.shape)-np.sum(margin_number,axis=1)>=3).all(), 'Too wide margin or too narrow design region!'

        for ii in range(margin_number.shape[0]):
            for jj in range(margin_number.shape[1]):
                if margin_number[ii,jj]==0:
                    margin_number[ii,jj] = 1 # minimum possible margin_number

        return margin_number

    def pixel_size(self,arr):
        squeeze_shape = np.array(np.squeeze(arr).shape)
        squeeze_size = np.array(self.phys_size)[np.array(self.phys_size).nonzero()]
        return squeeze_size/squeeze_shape # sizes of a pixel along all finite-thickness directions

    def minimum_length(self,arr,len_arr=None):
        arr = binarize(arr)
        margin_number = self.margin(arr)

        if np.array(len_arr).any(): # search the minimum length scale within a length array "len_arr"
            radius_list = sorted(list(np.abs(len_arr)/2))
            for radius in radius_list:
                diff_image = np.abs(self.open_operator(arr,radius)-self.close_operator(arr,radius)) # difference between open and close
                pixel_in = in_pixel_count(diff_image,margin_number=margin_number,dims=self.dims)
                if pixel_in>0:
                    return radius*2

            print("The minimum length scale is not in this array of lengths.")
            return
        else: # find the minimum length scale via binary search if "len_arr" is not provided
            radius_ub = min(self.phys_size)/2 # maximum meaningful filter radius of open and close operations
            diff_image_ub = np.abs(self.open_operator(arr,radius_ub)-self.close_operator(arr,radius_ub)) # difference between open and close
            pixel_in_ub = in_pixel_count(diff_image_ub,margin_number=margin_number,dims=self.dims) # number of interior pixels

            if pixel_in_ub>0:
                radii = [0,radius_ub/2,radius_ub]
                while np.abs(radii[0]-radii[2])>min(self.pixel_size(arr))/2:
                    radius = radii[1]
                    diff_image = np.abs(self.open_operator(arr,radius)-self.close_operator(arr,radius)) # difference between open and close
                    pixel_in = in_pixel_count(diff_image,margin_number=margin_number,dims=self.dims) # number of interior pixels

                    if pixel_in==0: radii[0],radii[1] = radius,(radius+radii[2])/2 # radius is too small
                    else: radii[1],radii[2] = (radius+radii[0])/2,radius # radius is still large

                return radii[1]*2

            else: # radius_ub is not a good starting radius of the binary search
                radius_initial,pixel_in_initial = radius_ub/1.5,0 # decrease the radius
                # search a starting radius until interior pixels emerge or the radius is unacceptably small
                while pixel_in_initial==0 and radius_initial>min(self.pixel_size(arr))/2:
                    radius_initial /= 1.5
                    diff_image_initial = np.abs(self.open_operator(arr,radius_initial)-self.close_operator(arr,radius_initial))
                    pixel_in_initial = in_pixel_count(diff_image_initial,margin_number=margin_number,dims=self.dims)

                if pixel_in_initial>0: # start the binary search
                    radii = [0,radius_initial/2,radius_initial]
                    while np.abs(radii[0]-radii[2])>min(self.pixel_size(arr))/2:
                    #min(np.array(self.phys_size)/np.array(arr.shape))/2:
                        radius = radii[1]
                        diff_image = np.abs(self.open_operator(arr,radius)-self.close_operator(arr,radius))
                        pixel_in = in_pixel_count(diff_image,margin_number=margin_number,dims=self.dims)
                        if pixel_in==0: radii[0],radii[1] = radius,(radius+radii[2])/2
                        else: radii[1],radii[2] = (radius+radii[0])/2,radius
                    return radii[1]*2
                else: # pixel_in_initial==0, fail to find a starting radius
                    print("The minimum length scale is at least ", radius_ub*2)
                    return

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
        raise AssertionError("Code for this dimension is not implemented!")

def guarantee_3d(arr):
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        arr_out = np.expand_dims(arr, axis=(0, 1, 2))
    elif arr.ndim == 1:
        arr_out = np.expand_dims(arr, axis=(1, 2))
    elif arr.ndim == 2:
        arr_out = np.expand_dims(arr, axis=(2))
    elif arr.ndim == 3:
        arr_out = arr
    else:
        raise AssertionError("Too many dimensions!")
    return arr_out

def simple_filter(arr,kernel):
    arr_shape = arr.shape
    npad = *((s, s) for s in arr_shape),

    # pad the kernel and the input array to avoid circular convolution and to ensure boundary conditions
    kernel = np.pad(kernel,pad_width=npad, mode='edge')
    arr = np.pad(arr,pad_width=npad, mode='edge')

    K, A = np.fft.fftn(kernel), np.fft.fftn(arr) # transform to frequency domain for fast convolution
    KA = K * A # convolution, i.e., multiplication in frequency domain
    arr_out = np.fft.fftshift(np.real(np.fft.ifftn(KA)))
    arr_out = _centered(arr_out,arr_shape) # Remove all the extra padding

    return arr_out

def _centered(arr, newshape): # Return the center newshape portion of the array.
    '''Helper function that reformats the padded array of the fft filter operation.
    Borrowed from scipy:
    https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L263-L270
    '''
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def in_pixel_count(arr,margin_number=np.ones((2,2),dtype=int),dims=2,threshold=0.5): # return the number of interior pixels with nonzero values
    pixel_int = 0 # initialize before counting
    arr = np.squeeze(arr)

    if dims==2:
        for ii in range(margin_number[0,0],arr.shape[0]-margin_number[0,1]):
            for jj in range(margin_number[1,0],arr.shape[1]-margin_number[1,1]):
                if (arr[adjacency([ii,jj])]>threshold).all(): # if the value of a pixel exceeds the threshold, it is regarded as nonzero
                    pixel_int += 1
    elif dims==3:
        for ii in range(margin_number[0,0],arr.shape[0]-margin_number[0,1]):
            for jj in range(margin_number[1,0],arr.shape[1]-margin_number[1,1]):
                for kk in range(margin_number[2,0],arr.shape[2]-margin_number[2,1]):
                    if (arr[adjacency([ii,jj,kk])]>threshold).all():
                        pixel_int += 1
    else:
        raise AssertionError("Code for this dimension is not implemented!")

    return pixel_int