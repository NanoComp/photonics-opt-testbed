This is an example of RGB metalens designs. Rasmus [@raelch](https://github.com/raelch) generated designs using his own code and reported the FOMs, which were validated on Meep by Mo. Mo also generated designs using Meep. The results are plotted for comparison.

``python3 metalens_check.py path/to/design_file design_resolution`` checks designs (in .csv format) on Meep, and reports FOMs and lengthscales.

The specific problem setup (credit to Rasmus) is shown below. The design region is 10nm by 1nm. Users may include arbitrary padding outside of the design region. The focal spot is approximately 2.4nm from the top of the lens. The source is a planewave with in-plane (Ex) polarization.

![image](https://user-images.githubusercontent.com/25192039/154070493-57449d3c-c647-4ee8-9cb5-35703a6dc67e.png)
