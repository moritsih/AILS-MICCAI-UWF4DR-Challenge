from torchvision.transforms import v2
import torch

# use this augmentation pipeline in the case of:
# 1. training
# 2. both datasets are included (therefore: resizing or cropping)
rotate_affine_flip_choice = v2.Compose([
                            v2.ToPILImage(),
                            v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            #v2.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.3, hue=0.3),
                            v2.RandomHorizontalFlip(),
                            v2.RandomVerticalFlip(),
                            #v2.RandomRotation(degrees=15, expand=True),
                            v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                            v2.RandomChoice([
                                # we have to think about this!!! DeepDRiD images get suuper zoomed in
                                v2.RandomResizedCrop(size=(800, 1016), scale=(0.8, 1.0)),
                                v2.Resize(size=(800, 1016)),
                            ])
                        ])


       

# use this augmentation pipeline in the case of:
# 1. validation
# 2. both datasets are included (therefore: resizing)
resize_only = v2.Compose([
            v2.ToPILImage(),
            v2.ToImage(), 
            v2.ToDtype(torch.float32, scale=True),

            v2.Resize(size=(512, 512))
        ])
