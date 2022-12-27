# Tracing Hα Fibrils through Bayesian Deep Learning

Haodi Jiang, Ju Jing, Jiasheng Wang, Chang Liu, Qin Li, Yan Xu, Jason T. L. Wang and Haimin Wang

Institute for Space Weather Sciences, New Jersey Institute of Technology

## Abstract

We present a new deep-learning method, named FibrilNet, for tracing chromospheric fibrils in Hα images of 
solar observations. Our method consists of a data preprocessing component that prepares training data 
from a threshold- based tool, a deep-learning model implemented as a Bayesian convolutional neural network 
for probabilistic image segmentation with uncertainty quantification to predict fibrils, 
and a post-processing component containing a fibril- fitting algorithm to determine fibril orientations. 
The FibrilNet tool is applied to high-resolution Hα images from an active region (AR 12665) 
collected by the 1.6 m Goode Solar Telescope (GST) equipped with high-order adaptive optics 
at the Big Bear Solar Observatory (BBSO). We quantitatively assess the FibrilNet tool, 
comparing its image segmentation algorithm and fibril-fitting algorithm with those employed 
by the threshold-based tool. Our experimental results and major findings are summarized as follows. 
First, the image segmentation results (i.e., the detected fibrils) of the two tools are quite similar, 
demonstrating the good learning capability of FibrilNet. Second, FibrilNet finds more accurate 
and smoother fibril orientation angles than the threshold-based tool. Third, FibrilNet is faster than 
the threshold-based tool and the uncertainty maps produced by FibrilNet not only 
provide a quantitative way to measure the confidence on each detected fibril, 
but also help identify fibril structures that are not detected by the threshold-based tool 
but are inferred through machine learning. Finally, we apply FibrilNet to full-disk Hα images 
from other solar observatories and additional high-resolution Hα images collected by BBSO/ GST, 
demonstrating the tool’s usability in diverse data sets.

----
References:

Tracing Hα Fibrils through Bayesian Deep Learning. Haodi Jiang et al 2021 ApJS 256 20

https://iopscience.iop.org/article/10.3847/1538-4365/ac14b7

https://arxiv.org/abs/2107.07886


Requirements: 

Python 3.6, Tensorflow-GPU 1.11.0, Keras 2.2.4, numpy 1.19.1, skimage 0.17.2, matplotlib 3.3.4, opencv-python 4.5.3, astropy 4.1.

Usage: 

The source code package contains the following folders/directories. 

1. The “data” folder contains two sub-folders “train” and “test”. The “train” folder contains training images and labels, and the "test" folder contains testing samples from BBSO/GST. 
2. The “models” folder contains the trained model (Create by yourself). You may directly use the pre-trained model “model.hdf5” to trace fibrils on the testing samples. (Due to the uploading file limitation of github, you may have to go here to download)
3. The “results” folder contains three sub-folders, “fibrils_on_ha”, “predicted_mask” and “uncertainty”. The testing results will be output to these folders after running the code.

Run “train.py” to train the model if you want to re-train the model. You may save the model with a different name by modifying the name in the code. Here, we use “model1.hdf5” as the default value.

Run “test.py” to test the model. The code will use the default pre-trained model “model.hdf5” to perform fibril tracing with uncertainty quantification, and save the results in the "results" folder. However, you can change the model's name in the code to use your re-trained model. 

Please note that you can always keep the folders as explained above for simple usage. If you want to re-organize the folders, please do remember to change the code accordingly in the “train.py” and “test.py” programs.
