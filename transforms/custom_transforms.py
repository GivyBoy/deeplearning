__author__ = "Gorkem Can Ates", "Anthony Givans"

__email__ = "gca45@miami.edu", "agg136@miami.edu"

from PIL import Image, ImageOps, ImageEnhance
from PIL import ImageFilter as IF
from typing import Any
import numpy as np
import torchvision.transforms.functional as TF
import torch
import cv2
import scipy.ndimage as ndimage
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt


def plot(d):
    plt.figure()
    plt.imshow(d)
    plt.show()


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, data, target):
        for transform in self.transforms:
            data, target = transform(data, target)
        return data, target


class Resize:
    def __init__(self, shape: tuple, mode='torch', multi_dim=False) -> None:
        self.shape = shape
        self.mode = mode
        self.multi_dim = multi_dim

    def __call__(self, data, target) -> Any:
        if self.mode == 'torch':
            data, target = TF.resize(data, self.shape, interpolation=Image.BICUBIC), \
                TF.resize(target, self.shape, interpolation=Image.NEAREST)
        elif self.mode == 'cv2':
            if self.multi_dim:
                data = self.resample_3d(data, self.shape)
                target = self.resample_3d(target, self.shape)
            else:
                data = cv2.resize(data,
                                  self.shape,
                                  interpolation=cv2.INTER_CUBIC)
                target = cv2.resize(target,
                                    self.shape,
                                    interpolation=cv2.INTER_NEAREST)
        else:
            raise Exception('Transform mode not found.')
        return data, target

    def resample_3d(self, img, target_size):
        imx, imy, imz = img.shape
        tx, ty, tz = target_size
        zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
        img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
        return img_resampled


class RandomRotation:

    def __init__(self, angles: tuple((int, int)) = (-60, 60), p=0.5, mode='torch', multi_dim=False) -> None:
        self.angles = angles
        self.p = p
        self.mode = mode
        self.multi_dim = multi_dim

    def __call__(self, data, target):
        if random.random() <= self.p:
            angle = random.randint(self.angles[0], self.angles[1])
            if self.mode == 'torch':
                data, target = TF.rotate(data, angle), TF.rotate(target, angle)
            elif self.mode == 'cv2':
                if self.multi_dim:
                    data_i, target_i = [], []
                    for i in range(data.shape[2]):
                        Md = cv2.getRotationMatrix2D((data[..., i].shape[0] / 2, data[..., i].shape[1] / 2), angle, 1)
                        Mt = cv2.getRotationMatrix2D((target[..., i].shape[0] / 2, target[..., i].shape[1] / 2), angle,
                                                     1)
                        d = cv2.warpAffine(data[..., i], Md, (data[..., i].shape[0], data[..., i].shape[1]))
                        t = cv2.warpAffine(target[..., i], Mt, (target[..., i].shape[0], target[..., i].shape[1]))
                        data_i.append(np.expand_dims(d, axis=2))
                        target_i.append(np.expand_dims(t, axis=2))
                        data, target = np.concatenate(data_i, axis=-1), np.concatenate(target_i, axis=-1)
                else:
                    Md = cv2.getRotationMatrix2D((data.shape[0] / 2, data.shape[1] / 2), angle, 1)
                    Mt = cv2.getRotationMatrix2D((target.shape[0] / 2, target.shape[1] / 2), angle, 1)
                    data = cv2.warpAffine(data, Md, (data.shape[0], data.shape[1]))
                    target = cv2.warpAffine(target, Mt, (target.shape[0], target.shape[1]))
            else:
                raise Exception('Transform mode not found.')
        return data, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5, mode='torch', multi_dim=False) -> None:
        self.p = p
        self.mode = mode
        self.multi_dim = multi_dim

    def __call__(self, data, target):
        if random.random() <= self.p:
            if self.mode == 'torch':
                data, target = TF.hflip(data), TF.hflip(target)
            elif self.mode == 'cv2':
                if self.multi_dim:
                    data_i, target_i = [], []
                    for i in range(data.shape[2]):
                        d, t = cv2.flip(data[..., i], 1), cv2.flip(target[..., i], 1)
                        data_i.append(np.expand_dims(d, axis=2))
                        target_i.append(np.expand_dims(t, axis=2))
                        data, target = np.concatenate(data_i, axis=-1), np.concatenate(target_i, axis=-1)
                else:
                    data, target = cv2.flip(data, 1), cv2.flip(target, 1)
            else:
                raise Exception('Transform mode not found.')
        return data, target


class RandomVerticalFlip:

    def __init__(self, p=0.5, mode='torch', multi_dim=False) -> None:
        self.p = p
        self.mode = mode
        self.multi_dim = multi_dim

    def __call__(self, data, target):
        if random.random() <= self.p:
            if self.mode == 'torch':
                data, target = TF.vflip(data), TF.vflip(target)
            elif self.mode == 'cv2':
                if self.multi_dim:
                    data_i, target_i = [], []
                    for i in range(data.shape[2]):
                        d, t = cv2.flip(data[..., i], 0), cv2.flip(target[..., i], 0)
                        data_i.append(np.expand_dims(d, axis=2))
                        target_i.append(np.expand_dims(t, axis=2))
                        data, target = np.concatenate(data_i, axis=-1), np.concatenate(target_i, axis=-1)
                else:
                    data, target = cv2.flip(data, 0), cv2.flip(target, 0)
            else:
                raise Exception('Transform mode not found.')
        return data, target


class GrayScale:
    def __init__(self, p=0.5, mode='torch', multi_dim=False) -> None:
        self.p = p
        self.mode = mode
        self.multi_dim = multi_dim

    def __call__(self, data, target) -> Any:
        if random.random() <= self.p:
            if self.mode == 'torch':
                # data = TF.to_grayscale(data, num_output_channels=3)
                # target = TF.to_grayscale(target, num_output_channels=3)

                data = ImageOps.grayscale(data)
                target = ImageOps.grayscale(target)

            elif self.mode == 'cv2':

                if self.multi_dim:

                    data_i, target_i = [], []

                    for i in range(data.shape[2]):
                        d = np.dot(data[..., :i], [0.299, 0.587, 0.114])
                        t = np.dot(target[..., :i], [0.299, 0.587, 0.114])

                        data_i.append(np.expand_dims(d, axis=2))

                        target_i.append(np.expand_dims(t, axis=2))

                        data, target = np.concatenate(data_i, axis=-1), np.concatenate(target_i, axis=-1)

                else:
                    # data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
                    # target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
                    data = np.dot(data[..., :3], [0.299, 0.587, 0.114])
                    target = np.dot(target[..., :3], [0.299, 0.587, 0.114])

                """
                Why does the formula above work? Well, grayscale is just the sum of the average of the RGB channels.
                The formula, however, gives a weighted average (aka luminosity method) that takes into consideration
                the contribution of each channel (RGB) to the formation of the image.
                
                ``Since red color has more wavelength of all the three colors, and green is the color that has not only 
                less wavelength then red color but also green is the color that gives more soothing effect to the eyes. 
                It means that we have to decrease the contribution of red color, and increase the contribution of the 
                green color, and put blue color contribution in between these two.``
                
                This is why the formula (to 3dp) works
                """

            else:
                raise Exception('Transform mode not found.')

        return data, target


class GaussianBlur:
    def __init__(self, kernel_size: list[(int, int)] = [11, 11], p=0.5, mode='torch', multi_dim=False) -> None:
        self.kernel_size = kernel_size
        self.p = p
        self.mode = mode
        self.multi_dim = multi_dim

    def __call__(self, data, target) -> Any:
        if random.random() <= self.p:
            if self.mode == 'torch':

                data = data.filter(IF.GaussianBlur(radius=5))
                target = target.filter(IF.GaussianBlur(radius=5))

            elif self.mode == 'cv2':

                if self.multi_dim:

                    data_i, target_i = [], []

                    for i in range(data.shape[2]):
                        d, t = cv2.GaussianBlur(data[..., i], self.kernel_size, 6), \
                            cv2.GaussianBlur(target[..., i], self.kernel_size, 6)

                        data_i.append(np.expand_dims(d, axis=2))
                        target_i.append(np.expand_dims(t, axis=2))
                        data, target = np.concatenate(data_i, axis=-1), np.concatenate(target_i, axis=-1)

                else:

                    # data = ndimage.gaussian_filter(data, sigma=6)
                    # target = ndimage.gaussian_filter(target, sigma=6)

                    data = cv2.GaussianBlur(data, self.kernel_size, 6)
                    target = cv2.GaussianBlur(target, self.kernel_size, 6)
            else:
                raise Exception('Transform mode not found.')

        return data, target


class Normalize:
    """
    I don't think this works properly
    """

    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, data, target):
        if random.random() <= self.p:
            data = self._normalize(data)
            target = self._normalize(target)

        return data, target

    def _normalize(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        MEAN = 255 * np.array([0.485, 0.456, 0.406])
        STD = 255 * np.array([0.229, 0.224, 0.225])

        x = img
        x = (x - MEAN) / STD

        print(x)

        return x


class Equalize:
    def __init__(self, p=0.5, mode="torch"):
        self.p = p
        self.mode = mode

    def __call__(self, data, target):
        if random.random() <= self.p:
            if self.mode == "torch":
                data = ImageOps.equalize(data)
                target = ImageOps.equalize(target)

            elif self.mode == "cv2":
                """
                We have to split the image in its channels and equalize them individually, then merge them after
                
                This implementation produces a near identical img to the ImageOps above
                """
                data = self._equalize(data)
                target = self._equalize(target)

            else:
                raise Exception('Transform mode not found.')

        return data, target

    def _equalize(self, img):
        b, g, r = cv2.split(img)
        equ_b = cv2.equalizeHist(b)
        equ_g = cv2.equalizeHist(g)
        equ_r = cv2.equalizeHist(r)
        equ = cv2.merge((equ_b, equ_g, equ_r))

        return equ


class Mirror:
    def __init__(self, p=0.5, mode="torch"):
        self.p = p
        self.mode = mode

    def __call__(self, data, target):
        if random.random() <= self.p:

            if self.mode == "torch":
                data = ImageOps.mirror(data)
                target = ImageOps.mirror(target)

            elif self.mode == "cv2":
                " can also use `np.fliplr(img)` "
                data = cv2.flip(data, 1)
                target = cv2.flip(target, 1)

            else:
                raise Exception('Transform mode not found.')

        return data, target


class Sharpen:
    def __init__(self, p=0.5, mode="torch"):
        self.p = p
        self.mode = mode

    def __call__(self, data, target):
        if random.random() <= self.p:

            if self.mode == "torch":
                data = ImageEnhance.Sharpness(data).enhance(8)  # val of 8 is similar to the sharper kernels below
                target = ImageEnhance.Sharpness(target).enhance(8)

            elif self.mode == "cv2":

                """
                Below is the kernel used for image sharpening, found here: 
                https://en.wikipedia.org/wiki/Kernel_(image_processing). I have seen people reference different kernels
                though, for eg:
                
                kernel = np.array([[0, -1, 0], # wikipedia version
                                   [-1, 5, -1],
                                   [0, -1, 0]])
                
                kernel = np.array([[-1, -1, -1], # sharper than the wikipedia kernel above
                                   [-1, 8, -1],
                                   [-1, -1, 0]])
                                   
                kernel = np.array([[-1, -1, -1], # similar to the kernel above (maybe a bit sharper)
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
                
                """
                kernel = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])

                data = cv2.filter2D(data, ddepth=-1, kernel=kernel)
                target = cv2.filter2D(target, ddepth=-1, kernel=kernel)

            else:
                raise Exception('Transform mode not found.')

        return data, target


class Invert:
    def __init__(self, p=0.5, mode="torch"):
        self.p = p
        self.mode = mode

    def __call__(self, data, target):
        if random.random() <= self.p:

            if self.mode == "torch":
                data = ImageOps.invert(data)
                target = ImageOps.invert(target)

            elif self.mode == "cv2":
                data = 255 - data  # very simple inversion trick
                target = 255 - target

            else:
                raise Exception('Transform mode not found.')

        return data, target


class Brightness:
    def __init__(self, p=0.5, mode="torch"):
        self.p = p
        self.mode = mode

    def __call__(self, data, target):
        if random.random() <= self.p:

            if self.mode == "torch":
                data = ImageEnhance.Brightness(data).enhance(1.5)  # 1 is the original img
                target = ImageEnhance.Brightness(target).enhance(1.5)

            elif self.mode == "cv2":

                """
                To increase (and default, decrease) you can also add a value to the np.array. Example:

                data = data.astype(float)
                data += 1.5
                data[data > 255] = 255
                data = data.astype(np.uint8)

                """

                data = cv2.convertScaleAbs(data, alpha=1, beta=1.5)  # alpha - contrast and beta - brightness
                target = cv2.convertScaleAbs(target, alpha=1, beta=1.5)

            else:
                raise Exception('Transform mode not found.')

        return data, target


class Contrast:
    def __init__(self, p=0.5, mode="torch"):
        self.p = p
        self.mode = mode

    def __call__(self, data, target):
        if random.random() <= self.p:

            if self.mode == "torch":
                data = ImageEnhance.Contrast(data).enhance(1.5)  # 1 is the original img
                target = ImageEnhance.Contrast(target).enhance(1.5)

            elif self.mode == "cv2":
                """
                To increase (and default, decrease) you can also multiply the np.array by a value. Example:
                
                data = data.astype(float)
                data *= 1.5
                data[data > 255] = 255
                data = data.astype(np.uint8)
                
                """
                data = cv2.convertScaleAbs(data, alpha=1.5, beta=0)  # alpha - contrast and beta - brightness
                target = cv2.convertScaleAbs(target, alpha=1.5, beta=0)

            else:
                raise Exception('Transform mode not found.')

        return data, target


# added these last minute, so I will work on them over the weekend

class Hue:
    def __init__(self):
        pass

    def __call__(self):
        pass


class Gamma:
    def __init__(self):
        pass

    def __call__(self):
        pass


class Solarize:
    def __init__(self):
        pass

    def __call__(self):
        pass


class Fog:
    def __init__(self):
        pass

    def __call__(self):
        pass


class PixelDropout:
    def __init__(self):
        pass

    def __call__(self):
        pass


class ToTensor:

    def __call__(self, data, target):
        data = np.array(data)
        target = np.array(target, dtype=np.float32)
        data = transforms.ToTensor()(data)
        target = transforms.ToTensor()(target)
        return data, target


if __name__ == "__main__":
    p_input = Image.open("cvc_input.png")
    p_target = Image.open("cvc_target.png")

    np_input = np.array(p_input)
    np_target = np.array(p_target)

    "I set all the probabilities to 1, so that it is certain that the transformations are performed"

    rr = RandomRotation(angles=(30, 45), p=1, mode="cv2")
    rr_input, rr_target = rr(np_input, np_target)

    gb = GaussianBlur(kernel_size=[5, 13], p=1, mode="cv2", multi_dim=False)
    gb_input, gb_target = gb(rr_input, rr_target)

    # gs = GrayScale(p=1, mode='cv2', multi_dim=False)
    # gs_input, gs_target = gs(gb_input, gb_target)

    # n = Normalize(p=1)
    # n_input, n_target = n(gb_input, gb_target)
    # plot(n_input)

    s = Sharpen(p=1, mode="cv2")
    s_input, s_target = s(gb_input, gb_target)
    plot(s_input)

    m = Mirror(p=1, mode="cv2")
    m_input, m_target = m(s_input, s_target)
    plot(m_input)

    e = Equalize(p=1, mode="cv2")
    e_input, e_target = e(m_input, m_target)
    plot(e_input)

    i = Invert(p=1, mode="cv2")
    i_input, i_target = i(e_input, e_target)
    plot(i_input)

    b = Brightness(p=1, mode="cv2")
    b_input, b_target = b(i_input, i_target)
    plot(b_input)

    c = Contrast(p=1, mode="cv2")
    c_input, c_target = c(b_input, b_target)
    plot(c_input)

    # the above is just proof that chaining them works
