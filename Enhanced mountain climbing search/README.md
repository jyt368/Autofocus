We developed an autofocus method with threshold denoising to enhance the stability, which utilised the combination of Laplacian function and variance operator as the focus evaluation function. 
Several modifications have been made to improve the autofocus performance based on the traditional mountain climbing search algorithm that moves the stage back and forth. 
Additionally, two-step curve fitting and NIQE final focus evaluation are used to make the mountain climbing search stride adaptive and to make the final focus position prediction more precise. 
![image](https://github.com/user-attachments/assets/5a1b12a1-19d3-4f41-a12d-f85f7da5cea6)
Instead of using two or three initial points to determine the focus direction, we used linear curve fitting to ensure more accurate search direction prediction.
Additionally, when the stage is moving forward, z-stack images are captured by the camera at each position. And each image is processed through the evluation function to get a score. The optimal focus position should have the highest evaluation score, representing the peak in the focus evaluation curve on the right figure. 
