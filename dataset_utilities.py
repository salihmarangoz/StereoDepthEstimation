import os
import sys
import glob
from io_routines import readPFM
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torchvision
import psutil
import signal
from collections import Counter
from torchvision import transforms
import matplotlib

import training_utilities as t_utils

# this needs virtual environment installation. see the notebook
local_pip_path = os.path.join(os.getcwd(), "venv_for_webp/lib/python3.8/site-packages")
sys.path.append(local_pip_path)
import webp


def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = 13
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def imshow_samples_and_print_information(dataset, idx=42, unnormalize=False, rainbow=False):
    """Shows left image, right image, and ground truth disparity map.

    Note:
        If rainbow=True, results should look similar to evaluation of algorithms in http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo

    Args:
        dataset: PyTorch dataset object
        idx: Index of sample in dataset
        rainbow: Uses better visualization for sparse disparity maps with limited interpolation.
        unnormalize: Unnormalizes input RGB data for visualization according to recommended torchvision.models normalization parameter.
    """

    left_, right_, disparity = dataset[idx]

    if unnormalize:
        unnormalizer_std = torchvision.transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225])
        unnormalizer_mean = torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        left = unnormalizer_mean(unnormalizer_std(left_))
        right = unnormalizer_mean(unnormalizer_std(right_))
    else:
        left = left_
        right = right_

    fig, axs = plt.subplots(1,3, figsize=(17,6))
    axs[0].imshow(left.moveaxis(0,2))
    axs[1].imshow(right.moveaxis(0,2))

    if rainbow:
        gt_int,_ = t_utils.interpolate_zerofilled_image(np.array(disparity).squeeze(0).copy(), method="nearest")
        rainbow_cmap = matplotlib.cm.get_cmap("gist_rainbow_r").copy()
        rainbow_cmap.set_bad('black',1.)
        axs[2].imshow(gt_int, cmap=rainbow_cmap, interpolation="nearest")
    else:
        axs[2].imshow(disparity.squeeze(0), cmap="gray", interpolation="nearest")


    if unnormalize:
        axs[0].set_title("Left Image (Un-normalized)")
        axs[1].set_title("Right Image (Un-normalized)")
    else:
        axs[0].set_title("Left Image")
        axs[1].set_title("Right Image")
    axs[2].set_title("Disparity")
    axs[0].grid(False)
    axs[1].grid(False)
    axs[2].grid(False)
    fig.tight_layout()

    print("Size of the {}:".format(dataset.name), len(dataset))
    print("Left/Right image shapes:", left.shape)
    print("Disparity shape:", disparity.shape)
    print("")

    return fig, axs

def split_dataset(dataset, first_part=0.9, second_part=0.1):
    """Splits dataset according to the given ratios.

    Note:
        first_part+second_part should be 1.0 and not checked in the function!

    Args:
        dataset: Input dataset to be splitted into two different datasets.
        first_part: Ratio for the first part.
        second_part: Ratio for the second part.

    Returns:
        First part of the dataset and second part of the dataset
    """

    first_size = int(len(dataset) * first_part)
    second_size = len(dataset) - first_size
    return torch.utils.data.random_split(dataset, (first_size, second_size))

# Used in make_dataloader(..) only! see: https://discuss.pytorch.org/t/dataloader-multiple-workers-and-keyboardinterrupt/9740/2
def _worker_init(x):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def make_dataloader(dataset, batch_size, shuffle=True, num_workers=-1, pin_memory=True):
    """Generates dataloaders that are robust to jupyter notebook interrupts and optimized parameters for training/validation/test.

    Note:
        Leave num_workers and pin_memory in default values for faster traning and inference with models.

    Args:
        dataset: Input dataset.
        batch_size: Batch size of dataloader output when next item is requested.
        shuffle: Shuffle indexes in the dataloader.
        num_workers: Defines how many CPU cores will be used. -1 by default will use all CPU cores.
        pin_memory: Enables caching mechanisms for faster training. Leave True for optimized configuration.
    Returns:
        Dataloader object.
    """

    if num_workers <= 0:
        num_workers = psutil.cpu_count()
    data_loader = torch.utils.data.DataLoader(dataset, 
                                               batch_size=batch_size, 
                                               shuffle=shuffle, 
                                               num_workers=num_workers, 
                                               pin_memory=pin_memory,
                                               worker_init_fn=_worker_init) # see: https://discuss.pytorch.org/t/dataloader-multiple-workers-and-keyboardinterrupt/9740/2

    return data_loader


def count_pixels(dataset, round_values=1, filter_zeros=False, normalize=True, counter=None):
    """Counts valid and non-valid disparity values in the stereo depth estimation datasets.

    Args:
        dataset: Input dataset for disparity counting.
        round_values: Counted values will be rounded before counted. Setting this value high may impact performance and memory use.
        filter_zeros: Defines if zero values should be discarded from the calculations.
        normalize: Makes sum of c_vals and maximum value of c_vals_cumsum equal to 1.0
        counter: Counter reference to continue counting on the new dataset. Can be used for cumilative counting on multiple datasets.

    Returns:
        True if successful, False otherwise.
    """
    
    if counter is None:
        c = Counter()
    else:
        c = counter
    
    for l,r,gt in dataset:
        gt = np.array(gt)
        if filter_zeros:
            gt = gt[gt>0.0]
        gt = np.round(gt.flatten(), round_values)
        c = c + Counter(gt)
    
    # Sort keys and values
    c_keys = np.array( list(c.keys()) )
    c_vals = np.array( list(c.values()) )
    sortmask = np.argsort(c_keys)
    c_keys = c_keys[sortmask]
    c_vals = c_vals[sortmask]
    
    c_vals_cumsum = np.cumsum(c_vals)
    
    if normalize:
        sum_value = c_vals_cumsum[-1]
        c_vals = c_vals / sum_value
        c_vals_cumsum = c_vals_cumsum / sum_value

    return c, c_keys, c_vals, c_vals_cumsum


def analyze_dataset_disparity_coverage(plt_subplots=None, use_file=False, compute_zeros_seperately=False, driving_path=None, monkaa_path=None, flyingthings3d_path=None, kitti_path=None):
    """Class methods are similar to regular functions.
    Note:
        Do not include the `self` parameter in the ``Args`` section.
    Args:
        plt_subplots: Quick fix for jupyter notebook plots. Example parameter: plt.subplots(1,2, figsize=(15,5))
        use_file: Input .npy for passing counting process and plotting the results directly.
        compute_zeros_seperately: The second parameter.
        driving_path: Dataset folder path. Will not be counted if given None.
        monkaa_path: Dataset folder path. Will not be counted if given None.
        flyingthings3d_path: Dataset folder path. Will not be counted if given None.
        kitti_path: Dataset folder path. Will not be counted if given None.

    Returns:
        (c, c_keys, c_vals, c_vals_cumsum) cumulative stats for specified datasets
    """

    if use_file:
        c = None
        c_keys, c_vals, c_vals_cumsum = np.load(use_file)
    else:
        c = Counter()
        if kitti_path is not None:
            c, c_keys, c_vals, c_vals_cumsum = count_pixels(KittiDataset(kitti_path), counter=c)
        if driving_path is not None:
            c, c_keys, c_vals, c_vals_cumsum = count_pixels(DrivingDataset(driving_path), counter=c)
        if monkaa_path is not None:
            c, c_keys, c_vals, c_vals_cumsum = count_pixels(MonkaaDataset(monkaa_path), counter=c)
        if flyingthings3d_path is not None:
            c, c_keys, c_vals, c_vals_cumsum = count_pixels(Flyingthings3dDataset(flyingthings3d_path), counter=c)
        np.save("dataset_analyze_precalculated", [c_keys, c_vals, c_vals_cumsum])

    # remove negative values
    mask = c_keys>=0
    c_keys = c_keys[mask]
    c_vals = c_vals[mask]
    c_vals_cumsum = c_vals_cumsum[mask]


    if compute_zeros_seperately:
        for idx, v in enumerate(c_keys):
            if v==0.0:
                break
        zero_vals = c_vals[idx]
        c_vals = c_vals / (1-c_vals[idx])
        c_vals_cumsum = (c_vals_cumsum-c_vals_cumsum[idx]) / (1-c_vals_cumsum[idx])
        c_vals[idx] = 0
        c_vals_cumsum[idx] = 0

    if plt_subplots is None:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5))
    else:
        fig, (ax1, ax2) = plt_subplots # quick fix showing plots in jupyter notebook

    ax1.plot(c_keys, c_vals)
    ax1.set_ylabel("Density")
    ax1.set_xlabel("Pixel Values")
    ax1.set_xlim([-10, 350])
    ax2.plot(c_keys, c_vals_cumsum)
    ax2.set_ylabel("Coverage")
    ax2.set_xlabel("Max Disparity")
    ax2.set_xlim([-10, 350])
    fig.tight_layout()

    print("Specific disparity values and dataset coverage:")
    for d in [32,64,96,128,160,192,256]:
        for idx, k in enumerate(c_keys):
            if k >= d:
                break
        print("> Disparity:", d, "Coverage:", c_vals_cumsum[idx])

    if compute_zeros_seperately:
        print("\n[!]Important Note: zero values hold", zero_vals, "of the total data and excluded from the calculations.")

    return c, c_keys, c_vals, c_vals_cumsum


########################## DATASET #############################################################

class DrivingDataset():
    """Defines Driving dataset for Scene Flow dataset. Uses only webp for RGB data.
    After analyzing the dataset we decided to use only "slow" samples since "fast" samples are low frequency samples of them.
    Also we decided to use "scene_forwards" and "scene_backwards". The simulated car path can be same but view angle is different.
    """
    
    def __init__(self, dataset_path, transforms=None):
        self.name = "Driving Dataset"
        self.dataset_path = dataset_path
        self.transforms = transforms
        
        self.data_path_list = []
        
        # scene backwards
        left_path = os.path.join(dataset_path, "frames_cleanpass_webp/15mm_focallength/scene_backwards/slow/left")
        right_path = os.path.join(dataset_path, "frames_cleanpass_webp/15mm_focallength/scene_backwards/slow/right")
        disparity_path = os.path.join(dataset_path, "disparity/15mm_focallength/scene_backwards/slow/left")
        self.list_files(left_path, right_path, disparity_path, self.data_path_list)
        
        # scene forwards
        left_path = os.path.join(dataset_path, "frames_cleanpass_webp/15mm_focallength/scene_forwards/slow/left")
        right_path = os.path.join(dataset_path, "frames_cleanpass_webp/15mm_focallength/scene_forwards/slow/right")
        disparity_path = os.path.join(dataset_path, "disparity/15mm_focallength/scene_forwards/slow/left")
        self.list_files(left_path, right_path, disparity_path, self.data_path_list)


    def list_files(self, left_path, right_path, disparity_path, out_list):
        for pl, pr, pd in zip(sorted(os.listdir(left_path)), sorted(os.listdir(right_path)), sorted(os.listdir(disparity_path))):
            if (pl.strip(".")[0] == pr.strip(".")[0] == pd.strip(".")[0]):
                pl_full = os.path.join(left_path, pl)
                pr_full = os.path.join(right_path, pr)
                pd_full = os.path.join(disparity_path, pd)
                out_list.append((pl_full, pr_full, pd_full))
            else:
                print("Err: File name mismatch", pl, pr, pd)
        
    def __len__(self):
        return len(self.data_path_list)


    def __getitem__(self, idx):
        left_path, right_path, disparity_path = self.data_path_list[idx]
        
        left_img = webp.load_image(left_path, mode="RGB")
        right_img = webp.load_image(right_path, mode="RGB")
        disparity = readPFM(disparity_path)[0].copy()
        
        if(self.transforms is not None):
            left_img, right_img, disparity = self.transforms((left_img, right_img, disparity))

        return left_img, right_img, disparity


class MonkaaDataset():
    """Defines Monkaa dataset for Scene Flow dataset. Uses only webp for RGB data.
    """
    
    def __init__(self, dataset_path, transforms=None):
        self.name = "Monkaa Dataset"
        self.dataset_path = dataset_path
        self.transforms = transforms
        
        self.data_path_list = []
        
        images_path = os.path.join(dataset_path, "frames_cleanpass_webp")
        disparity_path = os.path.join(dataset_path, "disparity")
        
        for images_subpath, disparity_subpath in zip(sorted(os.listdir(images_path)),
                                                     sorted(os.listdir(disparity_path))):
            lp = os.path.join(images_path, images_subpath, "left")
            rp = os.path.join(images_path, images_subpath, "right")
            dp = os.path.join(disparity_path, disparity_subpath, "left")
            self.list_files(lp, rp, dp, self.data_path_list)
        

    def list_files(self, left_path, right_path, disparity_path, out_list):
        for pl, pr, pd in zip(sorted(os.listdir(left_path)), sorted(os.listdir(right_path)), sorted(os.listdir(disparity_path))):
            if (pl.strip(".")[0] == pr.strip(".")[0] == pd.strip(".")[0]):
                pl_full = os.path.join(left_path, pl)
                pr_full = os.path.join(right_path, pr)
                pd_full = os.path.join(disparity_path, pd)
                out_list.append((pl_full, pr_full, pd_full))
            else:
                print("Err: File name mismatch", pl, pr, pd)
        
    def __len__(self):
        return len(self.data_path_list)


    def __getitem__(self, idx):
        left_path, right_path, disparity_path = self.data_path_list[idx]
        
        left_img = webp.load_image(left_path, mode="RGB")
        right_img = webp.load_image(right_path, mode="RGB")
        disparity = readPFM(disparity_path)[0].copy()
        
        if(self.transforms is not None):
            left_img, right_img, disparity = self.transforms((left_img, right_img, disparity))

        return left_img, right_img, disparity


class Flyingthings3dDataset():
    """Defines Flyingthings3d dataset for Scene Flow dataset. Uses only webp for RGB data.
    """

    def __init__(self, dataset_path, transforms=None):
        self.name = "Flyingthings 3D Dataset"
        self.dataset_path = dataset_path
        self.transforms = transforms

        self.data_path_list = []
        
        images_path = os.path.join(dataset_path, "frames_cleanpass_webp")
        disparity_path = os.path.join(dataset_path, "disparity")

        for images_subpath, disparity_subpath in zip(sorted(glob.iglob(images_path + '**/*/*/*', recursive=True)),
                                                     sorted(glob.iglob(disparity_path + '**/*/*/*', recursive=True))):
            lp = os.path.join(images_subpath, "left")
            rp = os.path.join(images_subpath, "right")
            dp = os.path.join(disparity_subpath, "left")
            self.list_files(lp, rp, dp, self.data_path_list)
        

    def list_files(self, left_path, right_path, disparity_path, out_list):
        for pl, pr, pd in zip(sorted(os.listdir(left_path)), sorted(os.listdir(right_path)), sorted(os.listdir(disparity_path))):
            if (pl.strip(".")[0] == pr.strip(".")[0] == pd.strip(".")[0]):
                pl_full = os.path.join(left_path, pl)
                pr_full = os.path.join(right_path, pr)
                pd_full = os.path.join(disparity_path, pd)
                out_list.append((pl_full, pr_full, pd_full))
            else:
                print("Err: File name mismatch", pl, pr, pd)

    def __len__(self):
        return len(self.data_path_list)


    def __getitem__(self, idx):
        left_path, right_path, disparity_path = self.data_path_list[idx]
        
        left_img = webp.load_image(left_path, mode="RGB")
        right_img = webp.load_image(right_path, mode="RGB")
        disparity = readPFM(disparity_path)[0].copy()
        
        if(self.transforms is not None):
            left_img, right_img, disparity = self.transforms((left_img, right_img, disparity))

        return left_img, right_img, disparity


class KittiDataset():
    """Defines KITTI 2015 dataset. first 150 samples are used for generating training/validation dataset, and last 50 are for test dataset.
    Only "training/image_2", "training/image_3", and "training/disp_occ_0" are used.

    NOTES FROM KITTI README:
        Disparity and flow values range [0..256] and [-512..+512] respectively. For
        both image types documented MATLAB and C++ I/O functions are provided
        within this development kit in the folders matlab and cpp. If you want to
        use your own code instead, you need to follow these guidelines:

        Disparity maps are saved as uint16 PNG images, which can be opened with
        either MATLAB or libpng++. A 0 value indicates an invalid pixel (ie, no
        ground truth exists, or the estimation algorithm didn't produce an estimate
        for that pixel). Otherwise, the disparity for a pixel can be computed by
        converting the uint16 value to float and dividing it by 256.0:

        disp(u,v)  = ((float)I(u,v))/256.0;
        valid(u,v) = I(u,v)>0;
    """

    def __init__(self, dataset_path, train_transforms=None, eval_transforms=None):
        self.name = "Kitti 2015 Dataset"
        self.dataset_path = dataset_path
        self.train_transforms = train_transforms
        self.eval_transforms = eval_transforms

        self.data_path_list = []
        
        left_images_path = os.path.join(dataset_path, "training/image_2")
        right_images_path = os.path.join(dataset_path, "training/image_3")
        disparity_path = os.path.join(dataset_path, "training/disp_occ_0")

        self.list_files(left_images_path, right_images_path, disparity_path, self.data_path_list)

    def only_stereo_images(self, file_list):
        filtered_list = []
        for f in file_list:
            if "_10.png" in f:
                filtered_list.append(f)
        return filtered_list
        

    def list_files(self, left_path, right_path, disparity_path, out_list):
        for pl, pr, pd in zip(sorted(self.only_stereo_images(os.listdir(left_path))), 
                              sorted(self.only_stereo_images(os.listdir(right_path))), 
                              sorted(os.listdir(disparity_path))):
            if (pl.strip(".")[0] == pr.strip(".")[0] == pd.strip(".")[0]):
                pl_full = os.path.join(left_path, pl)
                pr_full = os.path.join(right_path, pr)
                pd_full = os.path.join(disparity_path, pd)
                out_list.append((pl_full, pr_full, pd_full))
            else:
                print("Err: File name mismatch", pl, pr, pd)
        
    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, idx):
        left_path, right_path, disparity_path = self.data_path_list[idx]

        transforms = self.train_transforms
        if idx >= 150:
            transforms = self.eval_transforms
        
        left_img = Image.open(left_path)
        right_img = Image.open(right_path)
        disparity = np.array(Image.open(disparity_path), dtype=np.float32) / 256.0

        # kitti data is sparse and zero values are not registered points
        mask = disparity==0
        nan_vals = np.zeros(mask.shape).fill(np.nan)
        disparity[mask] = nan_vals
        
        if(transforms is not None):
            left_img, right_img, disparity = transforms((left_img, right_img, disparity))

        return left_img, right_img, disparity

    def split_dataset(self):
        trainset = torch.utils.data.Subset(self, range(0,150))
        testset = torch.utils.data.Subset(self, range(150,200))
        trainset.name = "Kitti 2015 Dataset Training+Validation"
        testset.name = "Kitti 2015 Dataset Evaluation"
        return trainset, testset


########################## TRANSFORMS #############################################################


class ToTensorMulti(torchvision.transforms.ToTensor):
    """Custom version of torchvision.transforms.ToTensor for stereo depth estimation datasets.
    ToTensor operation done in left, right and disparity maps.
    """
    def __call__(self, imgs):
        out_imgs = []
        for img in imgs:
            out_imgs.append(torchvision.transforms.functional.to_tensor(img))
        return out_imgs


class RandomCropMulti(torchvision.transforms.RandomCrop):
    """Custom version of torchvision.transforms.RandomCrop for stereo depth estimation datasets.
    Crop operation done in left, right and disparity maps with the same parameters.
    """
    def forward(self, imgs):
        out_imgs = []
        is_init=False
        i, j, h, w = 0,0,0,0
        for img in imgs:
            if not is_init:
                i, j, h, w = self.get_params(img, self.size)
                is_init = True
                
            if self.padding is not None:
                img = F.pad(img, self.padding, self.fill, self.padding_mode)

            width, height = img.shape[1], img.shape[2]
            # pad the width if needed
            if self.pad_if_needed and width < self.size[1]:
                padding = [self.size[1] - width, 0]
                img = torchvision.transforms.functional.pad(img, padding, self.fill, self.padding_mode)
            # pad the height if needed
            if self.pad_if_needed and height < self.size[0]:
                padding = [0, self.size[0] - height]
                img = torchvision.transforms.functional.pad(img, padding, self.fill, self.padding_mode)
                
            img = torchvision.transforms.functional.crop(img, i, j, h, w)
            
            out_imgs.append(img)

        return out_imgs

class SanitizeImageSizesMulti(object):
    """Clips input RGB images and disparity maps from left and right, also from top to match to the target size.
    This is needed for feeding input image with suitable shapes for neural networks.
    """
    def __call__(self, imgs, H_target=256+64+32, W_target=1024+128+64):
        out_imgs = []
        for img in imgs:
            B, H, W = img.shape
            H_clip = (H-H_target)//2
            W_clip = (W-W_target)//2
            out = img[:, H_clip:H_target+H_clip, W_clip:W_target+W_clip]
            out_imgs.append(out)

        return out_imgs

class ColorJitterMulti(torchvision.transforms.ColorJitter):
    """Custom version of torchvision.transforms.ColorJitter for stereo depth estimation datasets.
    ColorJitter operation only done in left and right RGB images. Disparity map doesn't get affected.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, imgs):
        l,r,d = imgs

        lr = torch.cat([l,r], dim=1)
        lr_jit = super().forward(lr)
        l_jit, r_jit = torch.chunk(lr_jit, 2, dim=1)

        return (l_jit, r_jit, d)


class NormalizeMulti(torchvision.transforms.Normalize):
    """Custom version of torchvision.transforms.Normalize for stereo depth estimation datasets.
    Normalize operation only done in left and right RGB images. Disparity map doesn't get affected.
    """
    def __init__(self, mean, std):
        super().__init__(mean=mean, std=std)

    def __call__(self, imgs):
        l,r,d = imgs

        lr = torch.cat([l,r], dim=1)
        lr_norm = super().forward(lr)
        l_norm, r_norm = torch.chunk(lr_norm, 2, dim=1)

        return (l_norm, r_norm, d)
