This Python program computes minimum length scales of 1d or 2d design patterns given by topology optimization. The theoretical basis of the 2d method lies in morphological transformations [1,2], which are realized by the OpenCV library [3].

The method for estimating the minimum length scale of solid regions in a 2d design pattern is outlined as follows.
1. Binarize the design pattern so that each element is a Boolean. For convenience, a pixel with a `True` value is called a solid pixel while a pixel with a `False` value is called a void pixel.
2. For a disc kernel with a given diameter $d$, compute the morphological opening $\mathcal{O}_d(\rho)$ of the binarized design pattern $\rho$ and obtain their difference $\mathcal{O}_d(\rho) \oplus \rho$.
3. Evaluate whether $\mathcal{O}_d(\rho) \oplus \rho$ contains a solid pixel that overlaps a non-interfacial pixel of solid regions in $\rho$. If no, the diameter of the disc kernel is less than the minimum length scale of solid regions; if yes, the diameter of the disc kernel is not less than the minimum length scale of solid regions.
4. Use binary search and repeat Steps 2 and 3 to seek the smallest kernel diameter at which a non-interfacial solid pixel emerges in the pattern of difference. This diameter is considered as the minimum length scale of solid regions.

To estimate the minimum length scale of void regions in a 2d design pattern, the design pattern is inverted after binarization, i.e., $\rho \rightarrow \sim \rho$ so that solid and void regions are interchanged. The subsequent procedures are the same as described above. This approach is equivalent to computing the difference between the original binary pattern $\rho$ without inversion and its morphological closing $\mathcal{C}_d(\rho)$ in Step 2 and compare that with non-interfacial pixels of void regions in $\rho$ in Step 3.

To obtain the minimum of the two minimum length scales, one can compute the two separately and take the minimum between them. If this minimum is the only desired quantity, a more efficient approach is to compute $\mathcal{O}_d(\rho) \oplus \mathcal{C}_d(\rho)$ in Step 2 and compare that with the union of non-interfacial pixels of solid and void regions in $\rho$ in Step 3.

For a 1d design pattern, the code simply searches for the minimum lengths among all solid or void segments. 

For 2d design patterns, discretization leads to inaccuracy. Errors of minimum length scales can be the size of a few pixels. Therefore, a design pattern with its minimum length scale much larger than the pixel size is preferred. If the minimum length scale is comparable to the pixel size, the relative error may be large. In addition, if a design pattern contains a sharp corner that are neither small perturbations at the single-pixel level nor a portion of the boundaries of the design pattern, the minimum length scale should be zero or the size of one pixel, but the method gives a nonzero value that are proportional to but larger than the size of one pixel.

References  
[1] Linus Hägg and Eddie Wadbro, On minimum length scale control in density based topology optimization, Struct. Multidisc Optim. 58(3), 1015–1032 (2018).  
[2] Rafael C. Gonzalez and Richard E. Woods, Digital Image Processing (Fourth Edition), Chapter 9 (Pearson, 2017).  
[3] OpenCV: Open Source Computer Vision Library, https://github.com/opencv/opencv