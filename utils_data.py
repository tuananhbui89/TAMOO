import torch 
import torchvision.transforms as T
import torchvision.transforms.functional as TF 
import numpy as np 

GLOBAL_FACTOR = None

def check_input(images): 
    assert(len(images.shape) == 4) # [B,C,H,W]
    assert(images.shape[1] == 3) # color images 
    assert(torch.torch.is_tensor(images))

def identity(images): 
    return images

def random_hflip(images):
    check_input(images)
    hflipper = T.RandomHorizontalFlip(p=0.5)
    return hflipper(images)

def random_vflip(images):
    check_input(images)
    vflipper = T.RandomVerticalFlip(p=0.5)
    return vflipper(images)

def random_rotate(images):
    check_input(images)
    rotater = T.RandomRotation(degrees=(0, 360))
    return rotater(images)

def random_resized_crop(images):
    """"
    scale (tuple of python:float) – Specifies the lower and upper bounds for the random area of the crop, 
    before resizing. The scale is defined with respect to the area of the original image.

    ratio (tuple of python:float) – lower and upper bounds for the random aspect ratio of the crop, 
    before resizing.

    """
    check_input(images)
    H,W = images.shape[2], images.shape[3]
    resize_cropper = T.RandomResizedCrop(size=(H, W), scale=(0.5, 1), ratio=(3/4,4/3))
    return resize_cropper(images)

def random_affine(images): 
    check_input(images)
    affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
    return affine_transfomer(images)

def random_autocontrast(images):
    """
    Note: inplace function, cannot work because of autograd 
    """
    check_input(images)
    autocontraster = T.RandomAutocontrast()
    return autocontraster(images)

def random_invert(images): 
    inverter = T.RandomInvert()
    return inverter(images)

def random_equalize(images): 
    check_input(images)
    equalizer = T.RandomEqualize()
    return equalizer(images)

def random_brightness(images):
    """"
    brightness (float or tuple of python:float (min, max)) – How much to jitter brightness. 
    brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness] or the given [min, max]. 
    Should be non negative numbers.
    """
    brightness = 1.0 
    jitter = T.ColorJitter(brightness=brightness, hue=.3)
    return jitter(images)

def random_gaussianblur(images): 
    blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    return blurrer(images)

def random_perspective(images): 
    perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=1.0)
    return perspective_transformer(images)


def hflip(images):
    p=GLOBAL_FACTOR
    assert(p >= 0 and 1 >= p)
    if p >= 0.5:  
        return TF.hflip(images)
    else: 
        return images

def vflip(images):
    p=GLOBAL_FACTOR
    assert(p >= 0 and 1 >= p)
    if p >= 0.5:  
        return TF.vflip(images)
    else: 
        return images    

def rotate(images): 
    p=GLOBAL_FACTOR
    assert(p >= 0 and 1 >= p)
    angle = -10. + p * 20. # range [-10,10] 
    return TF.rotate(images, angle)

def center_crop(images):
    """
    random with scale in range = [0.6, 1]
    """
    p=GLOBAL_FACTOR
    assert(p >= 0 and 1 >= p) 
    scale = 1.0 - p * 0.4
    H,W = images.shape[2], images.shape[3]
    img = TF.center_crop(images, output_size=[int(H*scale), int(W*scale)])
    return TF.resize(img, (H,W))

def affine(images): 
    """
    Affine transformation
    """
    p=GLOBAL_FACTOR
    assert(p >= 0 and 1 >= p) 
    angle = 30 + int(p * 40) # range [30, 70]
    translate = (0,0)
    scale = 0.5 + p * 0.25 # range [0.5, 0.75]
    shear = angle
    return TF.affine(images, angle=angle, translate=translate, scale=scale, shear=shear)

def adjust_contrast(images):
    """
    contrast_factor (float) – How much to adjust the contrast. Can be any non negative number. 
    0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by 
    a factor of 2.
    """
    p=GLOBAL_FACTOR
    assert(p >= 0 and 1 >= p)     
    contrast_factor = p * 3 # range [0, 3]
    return TF.adjust_contrast(images, contrast_factor)

def adjust_brightness(images): 
    """
    brightness_factor (float) – How much to adjust the brightness. Can be any non negative number. 
    0 gives a black image, 1 gives the original image while 2 increases the brightness by a factor 
    of 2.
    """
    p=GLOBAL_FACTOR
    assert(p >= 0 and 1 >= p) 
    brightness_factor = 1 + p * 0.3 # range [1,1.3]
    return TF.adjust_brightness(images, brightness_factor)

def adjust_gamma(images): 
    """
    gamma (float) – Non negative real number, same as 𝛾 in the equation. gamma larger than 1 make the 
    shadows darker, while gamma smaller than 1 make dark regions lighter.
    """
    p=GLOBAL_FACTOR
    assert(p >= 0 and 1 >= p)
    assert(torch.max(images) <= 1.0, torch.min(images) >= 0.0)
    gamma = 0.7 + p * 0.6 # range [0.7, 1.3]
    # return torch.pow(images, gamma)
    return TF.adjust_gamma(images, gamma)

def adjust_hue(images): 
    """
    
    """
    p=GLOBAL_FACTOR
    assert(p >= 0 and 1 >= p)
    hue_factor = -0.5 + p # range [-0.5, 0.5]
    return TF.adjust_hue(images, hue_factor)

def adjust_saturation(images):
    """
    
    """
    p=GLOBAL_FACTOR
    assert(p >= 0 and 1 >= p)
    saturation_factor = p * 3 # range [0, 3]
    return TF.adjust_saturation(images, saturation_factor)

def change_factor(rand=True):
    global GLOBAL_FACTOR
    if rand:
        GLOBAL_FACTOR = np.random.uniform()
    else:
        GLOBAL_FACTOR = 1.0

def test_autograd(T): 
    change_factor()
    print('GLOBAL_FACTOR=',GLOBAL_FACTOR)
    d = torch.rand(size=(1,3,10,10))
    img = torch.tensor(d, requires_grad=True)
    Ti = T(img)
    print(T.__name__, 'Ti shape', Ti.shape)
    L = torch.sum(Ti)
    G = torch.autograd.grad(L, img)[0]
    assert(G is not None)

def test_range(T): 
    change_factor()
    print('GLOBAL_FACTOR=',GLOBAL_FACTOR)
    d = torch.ones(size=(1,3,2,2))
    d = torch.rand(size=(1,3,2,2))
    img = torch.tensor(d)
    Ti = T(img)
    print(img)
    print(Ti)

if __name__ == '__main__':
    # test_autograd(affine)
    # test_autograd(adjust_contrast)
    # test_autograd(adjust_saturation)
    # test_autograd(adjust_hue)
    # test_autograd(adjust_gamma)
    # test_autograd(adjust_brightness)
    # test_autograd(center_crop)
    # test_autograd(rotate)
    # test_autograd(vflip)
    # test_autograd(hflip)
    # test_range(adjust_brightness)
    # test_range(adjust_hue)
    # test_range(adjust_saturation)
    # test_range(adjust_contrast)
    test_range(adjust_gamma)

