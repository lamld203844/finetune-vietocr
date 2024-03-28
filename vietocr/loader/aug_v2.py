import numpy as np
from PIL import Image
import albumentations as A

class AlbumentationsTransform:
    def __init__(self):
        # Define a lambda function that encapsulates the A.OneOf with a probability of 0.3
        sometimes = lambda aug: A.OneOf([aug], p=0.3)

        self.transform = A.Compose([
            # blur
            sometimes(A.OneOf([A.GaussianBlur(blur_limit=(0, 1.0), p=0.5),
                                A.MotionBlur(blur_limit=3, p=0.5)])),

            # color

            sometimes(A.HueSaturationValue(hue_shift_limit=(-10, 10),
                                            sat_shift_limit=0, val_shift_limit=0,
                                            p=0.5)),
            sometimes(A.RandomContrast(limit=(3, 10), p=0.5)),
            sometimes(A.InvertImg(p=0.5)),
            sometimes(A.Solarize(threshold=(32, 128), p=0.5)),
            sometimes(A.CoarseDropout(max_holes=8, max_height=8, max_width=8, min_holes=None,
                                        min_height=None, min_width=None, fill_value=0, p=0.5)),
            sometimes(A.MultiplicativeNoise(multiplier=(0.5, 1.5), per_channel=True, p=0.5)),
            sometimes(A.RandomBrightnessContrast(brightness_limit=(-40, 40),
                                    contrast_limit=0, brightness_by_max=True, p=0.5)),
            sometimes(A.JpegCompression(quality_lower=5, quality_upper=80, p=0.5)),
            
            # distort
            sometimes(A.RandomCrop(height=0.01, width=0.05, p=0.5)),
            sometimes(A.Perspective(scale=(0.01, 0.1), p=0.5)),
            sometimes(A.ShiftScaleRotate(shift_limit=(-0.1, 0.1),
                                            scale_limit=(0.7, 1.3),
                                            rotate_limit=0, interpolation=1,
                                            border_mode=4, value=(0,255), p=0.5)),
            
            sometimes(A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1,
                                        border_mode=4, value=None, mask_value=None,
                                        always_apply=False, approximate=False,
                                        p=0.5)),

            sometimes(A.OneOf([
                        A.Dropout(p=0.1, always_apply=False),
                        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=None, min_height=None, min_width=None, fill_value=0, mask_fill_value=None, always_apply=False, p=0.1),
                    ], p=0.5)),
        ], p=0.5)

    def __call__(self, img):
        img = np.array(img)
        augmented = self.transform(image=img)
        img = augmented['image']
        img = Image.fromarray(img)
        return img