# EyeNet_GANs
ACCV'18 - Synthesizing New Retinal Symptom Images by Multiple Generative Models - Pytorch Implementation 

Retinalimage collection contains three types of photography that are fluorescein angiography(FA), optical coherence tomography (OCT) and color fundus photography(CFP). FA are gray-scale images and CFP are colorful images. CFP and FA imaging are reliable for whole fundus, and used as our dataset.

-----------------------------------------------------------------

To improve the quality of generated images, we chose DCGANs and WGANs to establish our generative model. In this part, with a random initialized parameter, we build a generator of retinal diseases while the improving discriminator. For a specific symptom, generated images contain similar optical traits. Furthermore, high dimensional neural networks for computer vision sometimes materialize higher forms of neglected visual features. Therefore, generated retinal images not only become the aid of diagnosis and strategy to explore diseases, but also provide diverse computer training data.


## Fundus images with symptom of geographic atrophy (GA) via WGAN

![image](https://github.com/huckiyang/EyeNet-GANs/blob/master/final_imgs/GA_01.png)

![image](https://github.com/huckiyang/EyeNet-GANs/blob/master/wgan_img/wgan0929_GA/39500.png)

### path: /EyeNet-GANs/blob/masterwgan_img/wgan0929_GA/

## Drusen with symptom of fluorescein angiography (FA) via WGAN

![image](https://github.com/huckiyang/EyeNet-GANs/blob/master/final_imgs/drusen_01.png)

![image](https://github.com/huckiyang/EyeNet-GANs/blob/master/wgan_img/wgan0929_GA/39500.png)

### path EyeNet-GANs/wgan_img/wgan_drusen_fa/

## Class Activation Map (CAM) result on geographic atrophy (GA)
The class activation maps (CAMs) in [8] provide a method that localizes features on images. 
Through this method, not only the similarity of images is tested with high-level disease features, but a series of pathological details is built.

![image](https://github.com/huckiyang/EyeNet-GANs/blob/master/CAMs/CAM-GA-GA_01-resnet50.jpg)

If you find this useful in your work, please consider citing the following reference:

@article{liu2019synthesizing,
  title={Synthesizing New Retinal Symptom Images by Multiple Generative Models},
  author={Liu, Yi-Chieh and Yang, Hao-Hsiang and Yang, Chao-Han Huck and Huang, Jia-Hong and Tian, Meng and Morikawa, Hiromasa and Tsai, Yi-Chang James and Tegner, Jesper},
  journal={AI for Retinal Image Analysis(AIRIA) Workshop, ACCV 2018},
  year={2018}
}
