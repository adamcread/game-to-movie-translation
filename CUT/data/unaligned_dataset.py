import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF



class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.mask = opt.mask

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")    

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size, opt.sample_file))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size, opt.sample_file))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def scale_width(self, img, target_width, msk=None, method=Image.BICUBIC):
        w = target_width
        h = target_width // 2

        if self.mask:
            return img.resize((w, h), method), msk.resize((w, h), method)
        return img.resize((w, h), method)

    def transform(self, img, msk=None):
        if self.mask:
            img, msk = self.scale_width(img, self.opt.load_size, msk=msk)
        else:
            img = self.scale_width(img, self.opt.load_size)
                    
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(self.opt.crop_size//2, self.opt.crop_size))
        if self.mask:
            msk = TF.crop(msk, i, j, h, w)
        img = TF.crop(img, i, j, h, w)

        if random.random() > 0.5:
            if self.mask:
                msk = TF.hflip(msk)
            img = TF.hflip(img)

        if self.mask:
            msk = TF.to_tensor(msk)
        img = TF.to_tensor(img)

        img = TF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        if self.mask:
            return img, msk
        return img

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range

        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        if self.mask:
            M_A_path = A_path.replace('train/', 'mask/').replace('.jpg', '.png')
            M_B_path = B_path.replace('train/', 'mask/').replace('.jpg', '.png')

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        if self.mask:
            A_mask = Image.open(M_A_path).convert('RGB')
            B_mask = Image.open(M_B_path).convert('RGB')


        # Apply image transformation
        # For FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
#        print('current_epoch', self.current_epoch)

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        if self.mask:
            A, M_A = self.transform(A_img, A_mask)
            B, M_B = self.transform(B_img, B_mask)
        else:
            A = self.transform(A_img)
            B = self.transform(B_img)

        if self.mask:
            return {'A': A, 'B': B, 'M_A': M_A, 'M_B': M_B, 'A_paths': A_path, 'B_paths': B_path, 'M_A_path': M_A_path, 'M_B_path': M_B_path}
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
