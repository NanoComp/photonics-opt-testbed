This is a small program for computing the minimum length scale of a design pattern given by topology optimization. The theoretical basis of the method for 2d and 3d design patterns lies in morphological transformations [1,2]. Some code is copied or adapted from the filter.py file in Meep [3].

For a 1d design pattern, the code simply searches for the minimum length among all solid or void regions. For 2d or 3d design patterns, the method is outlined as follows.
1. Normalize and binarize the design pattern. The input design pattern should be a 1d, 2d or 3d array.
2. For a given circular structuring element that acts like a probe, compute the difference between the original pattern and the morphological opening. This difference is a 2d or 3d array with the same shape as the original pattern.
3. Count the number of interior solid pixels in the image of difference. The interior solid pixels are the solid pixels surrounded by other solid pixels, as shown in the figure below.
4. Using binary search, seek the smallest probe diameter at which an interior solid pixel emerges. This diameter is considered as the minimum length scale.

![image](https://github.com/mawc2019/ruler/blob/main/classification%20of%20pixels.jpg)

The method has limitations due to discretization. The error can be the size of a few pixels. Therefore, a design pattern with its minimum length scale much larger than the pixel size is preferred. If the minimum length scale is comparable to the pixel size, the relative error may be large. In addition, if a design pattern contains a sharp corner that are neither small perturbations at the single-pixel level nor a portion of the pattern boundary, the minimum length scale should be zero or the size of one pixel, but the method gives a nonzero value that are proportional to but much larger than the size of one pixel.

References  
[1] Linus Hägg and Eddie Wadbro, On minimum length scale control in density based topology optimization, Struct. Multidisc Optim. 58(3), 1015–1032 (2018).  
[2] Rafael C. Gonzalez and Richard E. Woods, Digital Image Processing (Fourth Edition), Chapter 9 (Pearson, 2017).  
[3] Alec Hammond et al., Adjoint solver in Meep: https://github.com/smartalecH/meep/blob/jax_rebase/python/adjoint/filters.py