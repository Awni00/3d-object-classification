# 3D Object Classification

A project to build an RGB-D object classification model. It is based on a dataset by Washington University:<br>
K. Lai, L. Bo, X. Ren, and D. Fox, “A large-scale hierarchical multi-view RGB-D object dataset,” <i>2011 IEEE International Conference on Robotics and Automation</i>, 2011.

The aim of this project is to build a model which outperforms models which use only 2 dimensional RGB data, and to match the performance of state-of-the-art depth models.

The general approach followed is to split the model into two processing streams, one for the RGB channels and one for the depth channel, that are combined at the end before making a prediction. In the RGB processing stream, we make use of the existing powerful RGB models through transfer learning. In the depth processing stream, we use more sophisticated representations of the depth information.