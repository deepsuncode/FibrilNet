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
import cv2
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np


test_data_path = 'data/test/'
results_path = 'results/'

model = FibrilNet('models/model.hdf5')

for name in os.listdir(test_data_path):
    print('Working on: {}'.format(name[:-4]))
    image_rgb = cv2.imread(test_data_path+name, cv2.IMREAD_COLOR)
    image = cv2.imread(test_data_path+name, 0)
    image = image / 255

    image = np.reshape(image, image.shape + (1,))
    image = np.reshape(image, (1,) + image.shape)

    prediction, aleatoric, epistemic = predict(model, image, T=2)  # change to T=50 for precise uncertainty maps

    prediction_ = prediction.copy()
    prediction1 = np.where(prediction_ > 0.5, 1, 0)  #
    prediction1 = prediction1 * 255
    cv2.imwrite(results_path+'predicted_mask/mask_{}.png'.format(name[:-4]), prediction1)

    # draw fibrils on Ha
    height, width = prediction1.shape
    for i in range(height):
        for j in range(width):
            if prediction1[i][j] == 255:
                image_rgb[i][j] = (0, 0, 255)
    cv2.imwrite(results_path+'fibrils_on_ha/fibrils_on_Ha_{}.png'.format(name[:-4]), image_rgb)

    # produce uncertainty maps
    fig, ax = plt.subplots(1, 1, figsize=(4, 4.2))
    cax1 = ax.imshow(aleatoric, cmap='jet')
    ax1_divider = make_axes_locatable(ax)
    cax11 = ax1_divider.append_axes("right", size="4%", pad="4%")
    fig.colorbar(cax1, cax=cax11, ticks=[0.05, 0.10, 0.15, 0.20])
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(results_path+'uncertainty/aleatoric_uncertainty_{}'.format(name), bbox_inches='tight', transparent=True, dpi=500)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4.2))
    cax2 = ax.imshow(epistemic, cmap='jet')
    ax2_divider = make_axes_locatable(ax)
    cax21 = ax2_divider.append_axes("right", size="4%", pad="4%")
    fig.colorbar(cax2, cax=cax21, ticks=[0.02, 0.04, 0.06, 0.08])
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(results_path+'uncertainty/epistmeic_uncertainty_{}'.format(name), bbox_inches='tight', transparent=True, dpi=500)

    print('Done')

