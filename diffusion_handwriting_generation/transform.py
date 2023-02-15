import albumentations as albu
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
import torchvision.transforms.functional as TF

from diffusion_handwriting_generation.config import CfgDict

__all__ = ["Transform"]


def augmentations():
    result = []
    return result


def augmentations_light():
    result = []
    return result


class ToTensor(DualTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        im_tensor = TF.to_tensor(img)
        return im_tensor

    def apply_to_mask(self, mask, **params):
        mask_tensor = TF.to_tensor(mask)
        return mask_tensor


class Normalize(ImageOnlyTransform):
    def __init__(self, image_params: CfgDict, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.img_norm_cfg = image_params.img_norm_cfg

    def apply(self, img, **params):
        img_norm = TF.normalize(
            img, mean=self.img_norm_cfg.mean, std=self.img_norm_cfg.std
        )
        return img_norm


class UnNormalize(object):
    """Useful class for further use in visualization needs"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def post_transform(image_params: CfgDict):
    return [
        albu.Resize(
            height=image_params.height,
            width=image_params.width,
            always_apply=True,
        ),
        ToTensor(),
        Normalize(image_params=image_params),
    ]


def train_transform(
    image_params: CfgDict,
    augs_lvl: str = "light",
):
    if augs_lvl == "hard":
        transforms = augmentations()
    elif augs_lvl == "light":
        transforms = augmentations_light()
    else:
        raise ValueError("Incorrect `augs_lvl`")

    result = albu.Compose(transforms=[*transforms, *post_transform(image_params)])
    return result


def valid_transform(image_params: CfgDict):
    return albu.Compose(transforms=[*post_transform(image_params)])


def infer_transform(image_params: CfgDict):
    return albu.Compose(transforms=[*post_transform(image_params)])


class Transform:
    def __init__(
        self,
        image_params: CfgDict,
        kind: str = "train",
        augs_lvl: str = "light",
    ):
        self.image_params = image_params
        self.kind = kind
        self.augs_lvl = augs_lvl

    def get_transforms(self):
        if self.kind == "train":
            transforms = train_transform(
                image_params=self.image_params,
                augs_lvl=self.augs_lvl,
            )
        elif self.kind == "val":
            transforms = valid_transform(self.image_params)
        else:
            return infer_transform(self.image_params)
        return transforms
