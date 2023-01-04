Requirements:

Python 3.6, Tensorflow-GPU 1.11.0, Keras 2.2.4, numpy 1.19.1, skimage 0.17.2, 
matplotlib 3.3.4, opencv-python 4.5.3, astropy 4.1.

Usage:

The source code package contains the following folders/directories.

The “data” folder contains two sub-folders “train” and “test”. 
The “train” folder contains training images and labels, 
and the "test" folder contains testing samples from BBSO/GST.

To get the pre-trained model “model.hdf5” in the "models" folder to trace fibrils on the testing samples, 
go to https://web.njit.edu/~wangj/deepsuncode/FibrilNet/

The “results” folder contains three sub-folders, 
“fibrils_on_ha”, “predicted_mask” and “uncertainty”. 
The testing results will be output to these folders after running the code.

Run “train.py” to train the model if you want to re-train the model. 
You may save the model with a different name by modifying the name in the code. 
Here, we use “model1.hdf5” as the default value.

Run “test.py” to test the model. 
The code will use the default pre-trained model “model.hdf5” to perform fibril tracing with uncertainty quantification, 
and save the results in the "results" folder. 
However, you can change the model's name in the code to use your re-trained model.

Please note that you can always keep the folders as explained above for simple usage. 
If you want to re-organize the folders, please do remember to change the code accordingly 
in the “train.py” and “test.py” programs.