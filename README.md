# Automatic Erroneous Gesture Detection in Robotic Surgery

## Motivations
* Siamese  networks have  been  used  in  image  recognition [[1]](#1)  and  eeg-based brain-computer interfaces signal classification [[2]](#2). The dual input  setup  for  Siamese  networks  could  help  increase  the number of training examples that could potentially increase the  network  performance  for  smaller  datasets. In this work, we  used the  Siamese  network to detect erroneous gestures during robotic surgeries and demonstrated its superiority under certain training setups.
* Our prior work [[3]](#3) shows the importance of context in terms of gesture and task. We used the code from this repo to study the how the contexual information affects our neural network performance. 
![image](https://github.com/zongyu-zoey-li/error_detection_Siamese/blob/main/hierarchy.png)
![image_context](https://github.com/zongyu-zoey-li/error_detection_Siamese/blob/main/G2_S_NP.PNG)

##  Environment 
python 3.8 <br />
pytorch 1.8.0


## References
<a id="1">[1]</a> 
G. R. Koch, “Siamese neural networks for one-shot image recognition,”ICML deep learning workshop, vol. 2. 2015. 

<a id="2">[2]</a> 
S. Shahtalebi, A. Asif, and A. Mohammadi, “Siamese neural networksfor eeg-based brain-computer interfaces.”Annual International Confer-ence of the IEEE Engineering in Medicine and Biology Society. IEEE Engineering  in  Medicine  and  Biology  Society.  Annual  International Conference, vol. 2020, p. 442–446, 2020.

<a id="3">[3]</a> 
K.   Hutchinson,   Z.   Li,   L.   A.   Cantrell,   N.   S.   Schenkman,   and H.  Alemzadeh,  “Analysis  of  executional  and  procedural  errors  in  dry-lab robotic surgery experiments,”arXiv, Jun 2021.
