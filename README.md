# Autofocus
Autofocus has been widely used in biological imaging to free users from tedious and repetitive works. However, due to background noise and different combination of sample types and staining method, the stability and reproducability of autofocus method is a main challenge. Therefore, we developed a fast and accurate autofocus method based on enhanced mountain climbing search algorithm which can be broadly used in different senarios. 
![image](https://github.com/user-attachments/assets/5a6414de-0683-47d8-858c-c218a69c45bc)

We developed an autofocus method with threshold denoising to enhance the stability, which utilised the combination of Laplacian function and variance operator as the focus evaluation function. 
Several modifications have been made to improve the autofocus performance based on the traditional mountain climbing search algorithm that moves the stage back and forth. 
Additionally, two-step curve fitting and NIQE final focus evaluation are used to make the mountain climbing search stride adaptive and to make the final focus position prediction more precise. The NIQE assessment code that we used is from https://github.com/guptapraful/niqe, which is based on skvideo's NIQE.
![image](https://github.com/user-attachments/assets/904ab8a7-fec3-49bb-be47-39761ad72b07)
Instead of using two or three initial points to determine the focus direction, we used linear curve fitting to ensure more accurate search direction prediction.
Additionally, when the stage is moving forward, z-stack images are captured by the camera at each position. And each image is processed through the evluation function to get a score. The optimal focus position should have the highest evaluation score, representing the peak in the focus evaluation curve on the right figure. 

To evaluate our method's performance, we compared our method with JAF (H&P) and OughtaFocus, which is integrated in Micro-manager software (https://github.com/micro-manager/micro-manager/tree/6a17971c3c61a1d04722c539b5003784b477bf74). The source code is added here as reference. JAF (H&P) is based on mountain climbing search. The search stride size and number of search should be predefined, which makes it hard to be applied to situations that the initial position range is unknown. OughtaFocus is based on Brent's algorithm, which shows faster speed but limited a accuracy and search range. 
