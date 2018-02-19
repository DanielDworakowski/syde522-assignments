import torch
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, transform
from torchvision import transforms, utils
from torchvision.transforms import functional

def plotSample(sample):
    ''' Plot an image batch.

    '''
    images_batch = sample['img']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.show()

class Rescale(object):
    '''Rescale the image in a sample to a given size.
    http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['img'], sample['labels']
        w, h = image.size
        #
        # Check input params.
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        #
        # New height and width
        new_h, new_w = int(new_h), int(new_w)
        #
        # Can add label transformation here.
        return {'img': image.resize((new_w, new_h), Image.BILINEAR),
                'labels': labels,
                'meta': sample['meta']}


class RandomCrop(object):
    '''Crop randomly the image in a sample.
    http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, labels = sample['img'], sample['labels']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        labels = labels - [left, top]

        return {'img': image,
                'labels': labels,
                'meta': sample['meta']}


class ToTensor(object):

    def __call__(self, sample):
        image, labels = sample['img'], sample['labels']

        return {'img': functional.to_tensor(image),
                'labels': torch.from_numpy(labels).squeeze_(),
                'meta': sample['meta']}

class Normalize(object):
    '''Normalizes an image.
    http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    '''

    def __init__(self, means, variances):
        assert len(means) == 1
        assert len(variances) == 1
        self.norm = transforms.Normalize(means, variances)

    def __call__(self, sample):
        sample['img'] = self.norm(sample['img'])
        return sample

class RandomFlips(object):
    '''Randomly horizontally and vertically flip images.
    '''

    def __init__(self, hFlip = 0.5, vFlip = 0.5):
        self.hFlip = transforms.RandomHorizontalFlip()
        self.vFlip = transforms.RandomVerticalFlip()

    def __call__(self, sample):
        img = sample['img']
        img = self.hFlip(img)
        img = self.vFlip(img)
        sample['img'] = img
        return sample

class ToPIL(object):
    ''' Convert to PIL.
    '''

    def __init__(self):
        self.topil = transforms.ToPILImage()

    def __call__(self, sample):
        sample['img'] = self.topil(sample['img'])
        return sample

class ToTensor(object):
    ''' Convert to PIL.
    '''

    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, sample):
        sample['img'] = self.totensor(sample['img'])
        return sample

class RandomRotation(object):
    """Randomly rotate an image between the specified angles.

    Args:
        deg (float): Extents of the random rotation.
    """
    def __init__(self, deg):
        #
        # TODO: Possibly extend this to work for a larger range and choose the closest correct grid location.
        self.t = transforms.RandomRotation(deg, Image.BILINEAR)

    def __call__(self, sample):
        sample['img'] = self.t(sample['img'])
        return sample

class RandomResizedCrop(object):

    def __init__(self, size, scale = (0.08, 1.0), ratio=(0.75, 1.3333333333333333)):
        self.t = transforms.RandomResizedCrop(size, scale=scale, ratio=ratio, interpolation=2)

    def __call__(self, sample):
        sample['img'] = self.t(sample['img'])
        return sample

class FiveCrop(object):

    def __init__(self, size, mean, var):
        self.t = transforms.FiveCrop(size)
        self.norm = transforms.Normalize(mean, var)


    def __call__(self, sample):
        imgs = self.t(sample['img'])
        imgs = torch.stack([self.norm(transforms.ToTensor()(crop)) for crop in imgs])
        sample['img'] = imgs
        return sample