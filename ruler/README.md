This is a small program for computing the minimum length scale of a design pattern given by topology optimization. The theoretical basis of the method lies in morphological transformations, which are detailed in this paper:
L. Hägg and E.Wadbro, On minimum length scale control in density based topology optimization, Struct. Multidisc Optim. 58(3), 1015–1032 (2018).


This method is outlined as follows.
1. Normalize and binarize the design pattern. Make sure the input design pattern is a 2d array composed of 0 and 1.
2. Select a filter radius and compute the difference between the operations of open and close operators. This difference should be a 2d or 3d array with the same shape as the design pattern.
3. Count the number of interior nonzero pixels in the image of difference. The interior nonzero pixels are those surrounded by other nonzeros pixels, as shown in the figure below.
4. Repeat Steps 2 and 3 for a series of filter radii, and seek the smallest filter radius at which an interior nonzero pixel emegrges. The minimum length scale is considered as twice this filter radius.

![image](https://github.com/mawc2019/ruler/blob/main/classification%20of%20pixels.jpg)
