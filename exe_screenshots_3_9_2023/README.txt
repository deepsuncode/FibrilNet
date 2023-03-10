
Ran with: 

tensorflow 2.11.0
numpy 1.21.6
pandas 1.3.5
scikit-learn 1.0.2
matplotlib 3.5.3
scikit-image 0.19.3
astropy 5.1
opencv-python 4.7.0.68

Changes made:

* FibrilNet.py:

Import statement for numpy:
	import numpy as np

line 93 changed to:
	with K.tf.compat.v1.variable_scope()(self.name):

In the MaxPoolingWithArgmax2D class:

Override get_config added:
    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
        })
        return config

In the MaxUnpooling2d class:

Override get_config added:
    def get_config(self):
        config = super().get_config()
        config.update({
            "size": self.size,
        })
        return config

* test.py:

Import statement for numpy:
	import numpy as np