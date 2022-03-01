import numpy as np
# Some code is copied or adapted from the Meep code at https://github.com/smartalecH/meep/blob/jax_rebase/python/adjoint/filters.py 

class morph:
    def __init__(self,size,margin_size=np.zeros((2,2)),proj_strength=10**6):
        self.size = size # physical size in the row and column directions
        self.margin_size = np.reshape(margin_size,(2,2)) # widths of marginal regions to be ignored
        #in margin_size, the two elements in the first row correspond to the top and bottom, and the two elements in the second row correspond to the left and right
        self.proj_strength = proj_strength # This is a parameter for the functions heaviside_erosion and heaviside_dilation
        
    def cylindrical_filter(self,arr,radius):
        # Construct the grid over the entire design region
        xv, yv = np.meshgrid(np.linspace(
            -self.size[0]/2,self.size[0]/2,arr.shape[0]), np.linspace(-self.size[1]/2,self.size[1]/2,arr.shape[1]), sparse=True, indexing='ij')

        # Calculate the kernel
        kernel = np.where(np.abs(xv ** 2 + yv ** 2) <= radius**2,1,0)#.T

        # Normalize the kernel
        kernel = kernel / np.sum(kernel.flatten()) # Normalize the filter

        # Filter the response
        arr_out = simple_2d_filter(arr,kernel,arr.shape[0],arr.shape[1])

        return arr_out
    
    def heaviside_erosion(self,arr,radius):

        beta = self.proj_strength
        arr_hat = self.cylindrical_filter(arr,radius)
        
        return np.exp(-beta*(1-arr_hat)) + np.exp(-beta)*(1-arr_hat)

    def heaviside_dilation(self,arr,radius):

        beta = self.proj_strength
        arr_hat = self.cylindrical_filter(arr,radius)
        
        return 1 - np.exp(-beta*arr_hat) + np.exp(-beta)*arr_hat
    
    def open_operator(self,arr,radius):
        # erosion and then dilation
        he = self.heaviside_erosion(arr,radius)
        hdhe = self.heaviside_dilation(he,radius)
        
        return hdhe
    
    def close_operator(self,arr,radius):
        # dilation and then erosion
        hd = self.heaviside_dilation(arr,radius)
        hehd = self.heaviside_erosion(hd,radius)
        
        return hehd
    
    def margin(self,arr): # compute the numbers of pixels corresponding to the marginal widths
        margin_number = np.round(np.vstack((self.margin_size[0,:]/self.size[0]*arr.shape[0],self.margin_size[1,:]/self.size[1]*arr.shape[1])))
        if np.sum(margin_number[0,:])+3>arr.shape[0] or np.sum(margin_number[1,:])+3>arr.shape[1]:
            raise ValueError("Too wide margin or too narrow design region!")

        for ii in range(margin_number.shape[0]):
            for jj in range(margin_number.shape[1]):
                if margin_number[ii,jj]==0:
                    margin_number[ii,jj] = 1
                if margin_number[ii,jj]<0:
                    raise ValueError("Margin widths should be positive!")

        return margin_number.astype(int)
        
    
    def minimum_length(self,arr,len_arr=None):
        arr = binarize(arr)
        margin_number = self.margin(arr)

        if np.array(len_arr).any(): # search the minimum length scale within a length array "len_arr"
            radius_list = sorted(list(np.abs(len_arr)/2))
            for radius in radius_list:
                diff_image = np.abs(self.open_operator(arr,radius)-self.close_operator(arr,radius)) # difference between open and close
                pixel_in = in_pixel_count(diff_image,margin_number=margin_number)
                if pixel_in>0:
                    print("The minimum length scale is ",radius*2)
                    return radius*2

            print("The minimum length scale is not in this array of lengths.")
            return
        else: # find the minimum length scale via binary search if "len_arr" is not provided
            radius_ub = min(self.size[0],self.size[1])/2 # largest meaningful radius of open and close operations
            diff_image_ub = np.abs(self.open_operator(arr,radius_ub)-self.close_operator(arr,radius_ub)) # difference between open and close
            pixel_in_ub = in_pixel_count(diff_image_ub,margin_number=margin_number)
            
            if pixel_in_ub>0:
                radii = [0,radius_ub/2,radius_ub]
                while np.abs(radii[0]-radii[2])>min(np.array(self.size)/np.array(arr.shape))/2:
                    radius = radii[1]
                    diff_image = np.abs(self.open_operator(arr,radius)-self.close_operator(arr,radius)) # difference between open and close
                    pixel_in = in_pixel_count(diff_image,margin_number=margin_number)
                    if pixel_in==0: radii[0],radii[1] = radius,(radius+radii[2])/2
                    else: radii[1],radii[2] = (radius+radii[0])/2,radius

                return radii[1]*2
            else:
                print("The minimum length scale is at least ", radius_ub*2)
                return
            

def binarize(arr,demarcation=0.5):
    arr_normalized = (arr-min(arr.flatten()))/(max(arr.flatten())-min(arr.flatten())) # normalize the data of the image
    arr_binarized = np.sign(arr_normalized-demarcation)/2+0.5 # binarize the data of the image with the threshold 0.5
    return arr_binarized

def simple_2d_filter(arr,kernel,Nx,Ny):
    # Get 2d parameter space shape
    (kx,ky) = kernel.shape

    # Ensure the input is 2D
    arr = arr.reshape(Nx,Ny)

    # pad the kernel and input to avoid circular convolution and
    # to ensure boundary conditions are met.
    kernel = _zero_pad(kernel,((kx,kx),(ky,ky)))
    arr = _edge_pad(arr,((kx,kx),(ky,ky)))
    
    # Transform to frequency domain for fast convolution
    K = np.fft.fft2(kernel)
    A = np.fft.fft2(arr)
    
    # Convolution (multiplication in frequency domain)
    KA = K * A
    
    # We need to fftshift since we padded both sides if each dimension of our input and kernel.
    arr_out = np.fft.fftshift(np.real(np.fft.ifft2(KA)))
    
    # Remove all the extra padding
    arr_out = _centered(arr_out,(kx,ky))
    return arr_out

def _centered(arr, newshape):
    '''Helper function that reformats the padded array of the fft filter operation.

    Borrowed from scipy:
    https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L263-L270
    '''
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]

def _edge_pad(arr, pad):
    
    # fill sides
    left = np.tile(arr[0,:],(pad[0][0],1)) # left side
    right = np.tile(arr[-1,:],(pad[0][1],1)) # right side
    top = np.tile(arr[:,0],(pad[1][0],1)).transpose() # top side
    bottom = np.tile(arr[:,-1],(pad[1][1],1)).transpose() # bottom side)
    
    # fill corners
    top_left = np.tile(arr[0,0], (pad[0][0],pad[1][0])) # top left
    top_right = np.tile(arr[-1,0], (pad[0][1],pad[1][0])) # top right
    bottom_left = np.tile(arr[0,-1], (pad[0][0],pad[1][1])) # bottom left
    bottom_right = np.tile(arr[-1,-1], (pad[0][1],pad[1][1])) # bottom right
    
    out = np.concatenate((
        np.concatenate((top_left,top,top_right)),
        np.concatenate((left,arr,right)),
        np.concatenate((bottom_left,bottom,bottom_right))    
    ),axis=1)
    
    return out

def _zero_pad(arr, pad):
    
    # fill sides
    left = np.tile(0,(pad[0][0],arr.shape[1])) # left side
    right = np.tile(0,(pad[0][1],arr.shape[1])) # right side
    top = np.tile(0,(arr.shape[0],pad[1][0])) # top side
    bottom = np.tile(0,(arr.shape[0],pad[1][1])) # bottom side
    
    # fill corners
    top_left = np.tile(0, (pad[0][0],pad[1][0])) # top left
    top_right = np.tile(0, (pad[0][1],pad[1][0])) # top right
    bottom_left = np.tile(0, (pad[0][0],pad[1][1])) # bottom left
    bottom_right = np.tile(0, (pad[0][1],pad[1][1])) # bottom right
    
    out = np.concatenate((
        np.concatenate((top_left,top,top_right)),
        np.concatenate((left,arr,right)),
        np.concatenate((bottom_left,bottom,bottom_right))    
    ),axis=1)
    
    return out

def in_pixel_count(arr,margin_number=np.ones((2,2),dtype=int),threshold=0.5):
    # if the value of a pixel exceeds the threshold, it is regarded as nonzero
    pixel_int = 0 # number of interior pixels with nonzero values
    for ii in range(margin_number[0,0],arr.shape[0]-margin_number[0,1]):
        for jj in range(margin_number[1,0],arr.shape[1]-margin_number[1,1]):
            if arr[ii-1,jj]>threshold and arr[ii+1,jj]>threshold and arr[ii,jj-1]>threshold and arr[ii,jj+1]>threshold:
                pixel_int += 1

    return pixel_int # numbers of interior pixels with nonzero values

def pixel_count(arr,threshold): # if the value of a pixel exceeds the threshold, it is regarded as nonzero
    pixel_tot,pixel_int = 0,0 # number of pixels with nonzero values, and among which the number of interior pixels
    for ii in range(arr.shape[0]):
        for jj in range(arr.shape[1]):
            if arr[ii,jj]>threshold:
                pixel_tot += 1
                if ii>0 and ii<arr.shape[0]-1 and jj>0 and jj<arr.shape[1]-1:
                    if arr[ii-1,jj]>threshold and arr[ii+1,jj]>threshold and arr[ii,jj-1]>threshold and arr[ii,jj+1]>threshold:
                        pixel_int += 1

    return pixel_tot-pixel_int,pixel_int # numbers of exterior and interior pixels with nonzero values
