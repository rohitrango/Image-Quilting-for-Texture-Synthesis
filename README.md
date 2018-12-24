## Image Quilting for Texture Synthesis

This repository is an implementation of the image quilting algorithm for texture synthesis using Python. For more details on the algorithm, refer to the original paper by Alexei A. Efros and Willian T. Freeman [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-siggraph01.pdf).

## Usage
To run the code on a given texture, run the following code:

`python main.py --image_path <image_path> --block_size <block_size> --overlap <overlap> --scale <scale> --num_outputs <num_outputs> --output_file <filename> --plot <plot> --tolerance <tolerance>`

For more details, use `python main.py -h`

## Results
Here are some results of the image quilting for texture synthesis algorithm.


![](https://raw.githubusercontent.com/rohitrango/Image-Quilting-for-Texture-Synthesis/master/textures/t16.png)  ![](https://raw.githubusercontent.com/rohitrango/Image-Quilting-for-Texture-Synthesis/master/results/t16.png)

![](https://raw.githubusercontent.com/rohitrango/Image-Quilting-for-Texture-Synthesis/master/textures/t6.png)  ![](https://raw.githubusercontent.com/rohitrango/Image-Quilting-for-Texture-Synthesis/master/results/t6.png)


![](https://raw.githubusercontent.com/rohitrango/Image-Quilting-for-Texture-Synthesis/master/textures/t12.png)  ![](https://raw.githubusercontent.com/rohitrango/Image-Quilting-for-Texture-Synthesis/master/results/t12.png)

![](https://raw.githubusercontent.com/rohitrango/Image-Quilting-for-Texture-Synthesis/master/textures/t14.png)  ![](https://raw.githubusercontent.com/rohitrango/Image-Quilting-for-Texture-Synthesis/master/results/t14.png)

![](https://raw.githubusercontent.com/rohitrango/Image-Quilting-for-Texture-Synthesis/master/textures/t18.png)  ![](https://raw.githubusercontent.com/rohitrango/Image-Quilting-for-Texture-Synthesis/master/results/t18.png)

## References
This is the main paper. <br>
<div id='cul_citation_2386435' class='cul_citation'>
	<a href='http://www.citeulike.org/user/aldershoff/article/2386435'><img class='cul_citation_icon' src='http://www.citeulike.org/static/img/cul_icon.gif' /></a>
	<span class='cul_citation_text'>Image Quilting for Texture Synthesis and Transfer <i>Proceedings of SIGGRAPH 2001</i> (August 2001), pp. 341-346 by Alexei A. Efros, William T. Freeman edited by Eugene Fiume</span>
</div>