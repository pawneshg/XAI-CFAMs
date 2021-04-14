# XAI: CFAM

### Goal:
The goal of this project is to predict how much discriminative image regions used by CNNs to identify the categories. 
![Goal][ppt_goal]

##### 1. Class Filter Activation Maps: 
Procedure to generate class filter activation map is by using weights of last layer and activation value of global average pooling layer. 
The product of these values helps to select the top "k" activation maps which are responsible for prediction.
<br>
![CFAM][ppt_cfam]

[code](https://git.opendfki.de/pawneshg/cam/-/blob/master/activation/utils.py#L60)
##### 2. Computation of Omega value:
Omega represents the ratio of common area to the threshold activation map area.
![Omega][ppt_omega]
<br>
[code](https://git.opendfki.de/pawneshg/cam/-/blob/master/activation/utils.py#L246)

[code](https://git.opendfki.de/pawneshg/cam/-/blob/master/activation/visualize.py#L118) <br>

### System Flow:
![Process][ppt_process]


###  Results :
![Example #1][ppt_result1]
![Example #1][ppt_result2]


[img1]: docs/results_1.png
[img2]: docs/results_2.png
[img3]: docs/results_3.png
[ppt_cfam]: docs/ppt_cfam.png
[ppt_omega]: docs/ppt_omega.png
[ppt_process]: docs/ppt_process.png
[ppt_result1]: docs/ppt_result1.png
[ppt_result2]: docs/ppt_result2.png
[ppt_goal]: docs/ppt_goal.png