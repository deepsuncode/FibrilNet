# =========================================================================
#   (c) Copyright 2021
#   All rights reserved
#   Programs written by Haodi Jiang
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================

from FibrilNet import *

input_path ='data/'
train_datagen = train_generator(1, input_path + 'train', 'image', 'label')
model = FibrilNet()
model_checkpoint = ModelCheckpoint('models/model1.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(train_datagen, steps_per_epoch=1000, epochs=10, verbose=2, callbacks=[model_checkpoint])
