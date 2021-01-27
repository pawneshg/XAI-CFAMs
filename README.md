# CAM

### Goal:
The goal of this project is to predict how much discriminative image regions used by CNNs to identify the categories. 
##### 1. Class Activation Maps: 
Procedure to generate class activation map is by using weights of last layer and activation value of global average pooling layer. 
The product of these values helps to select the top "k" activation maps which are responsible for prediction.
<br>
[code](https://git.opendfki.de/pawneshg/cam/-/blob/master/activation/utils.py#L60)
##### 2. Computation of Omega value:
Omega represents the intersection area between threshold activation map and the object area, divided by the threshold activation map area.
<br>
[code](https://git.opendfki.de/pawneshg/cam/-/blob/master/activation/utils.py#L246)
##### 3. Computation of Evaluation matrix: 
Evaluation Matrix stores the values of omega. 
[code](https://git.opendfki.de/pawneshg/cam/-/blob/master/activation/visualize.py#L77)
##### 4. Predict the foreground/background percentage areas focused by CNNs.
How much area a CNNs focus while predicting the object categories. 
Approach 1: Average of omega values. <br>
Approach 2: Average of weighted omega values.<br>
[code](https://git.opendfki.de/pawneshg/cam/-/blob/master/activation/visualize.py#L118) <br>


Results :
![Example #1][img1]


![Example #2][img2]


![Example #3][img3]

[img1]: docs/results_1.png
[img2]: docs/results_2.png
[img3]: docs/results_3.png