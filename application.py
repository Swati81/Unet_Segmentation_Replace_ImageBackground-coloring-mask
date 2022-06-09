import os
import cv2
import numpy as np

from prediction import predict, color_mask, replace_image

model_path = 'unet_models/unet_model_car_256.hdf5'
img_path = "car_data/images"
lst = os.listdir(img_path)

# out = cv2.VideoWriter('car-unet.avi', cv2.VideoWriter_fourcc(*'DIVX'), 4, (2877, 1280))
for im in lst:
    img = cv2.imread(os.path.join(img_path, im), 1)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_NEAREST)
    p = predict(model_path=model_path, img=img)
    mask = p.prediction().copy()
    mask2 = replace_image(mask=mask, img=img, type='RI').copy()
    mask3 = replace_image(mask=mask, img=img, type='RB').copy()
    mask4 = replace_image(mask=mask, img=img, type='WM').copy()
    mask1 = color_mask(mask).copy()
    join1 = np.concatenate((img, mask1), axis=1)
    join2 = np.concatenate((join1, mask*255), axis=1)
    join3 = np.concatenate((mask3, mask2), axis=1)
    join4 = np.concatenate((join3, mask4), axis=1)
    jn = np.concatenate((join2, join4), axis=0)
    cv2.imshow('Super Masking', jn)
    #out.write(jn)
    cv2.waitKey(1)
#out.release()
cv2.destroyAllWindows()

"""
code written by @swati sinha, @maitry sinha, @bibekananda sinha 
"""
