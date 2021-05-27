# 3D Object Classification

A project to build an RGB-D object classification model. It uses a dataset by Washington University:<br>
K. Lai, L. Bo, X. Ren, and D. Fox, “A large-scale hierarchical multi-view RGB-D object dataset,” <i>2011 IEEE International Conference on Robotics and Automation</i>, 2011.

The aim of this project is to build a computer vision object classification model which outperforms models which use only 2-dimensional RGB data, and to at least match the performance of state-of-the-art 3-dimensional RGB-D models.

A literature review informed an approach based on two key ideas: The first is to split the model into two processing streams, one for the RGB channels and one for the depth channel, that are finally merged at the end before making a prediction. The second is to transform the depth information into a more sophisticated representation; namely a geocentric encoding propsed by Gupta et. al called HHA:

S. Gupta, R. Girshick, P. Arbeláez and J. Malik, "Learning Rich Features from RGB-D Images for Object Detection and Segmentation," <i>Computer Vision – ECCV 2014</i>, pp. 345-360, 2014. <br>

This approach is superior to the naive approach of simply stacking an additional depth channel and building a 4-channel CNN since it exploits the structural differences between visual information and depth information. Additionally, it enables the use of pre-trained well-researched RGB feature extractors in the RGB processing stream. Indeed, we found that the added depth information improved performance on our dataset.

The model and training code is in the `src` directory. The directory also contains some useful utilities. `demo.ipynb` demos the trained models on 3-dimensional RGB-D images. `model_evaluation.ipynb` presents an evaluation of the performance of the various models. `depth_validation.ipynb` presents an evaluation of the utilization of depth information by the models.
