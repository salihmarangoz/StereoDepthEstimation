import torch; import torch.nn as nn
from torchvision.utils import save_image
import optuna
import numpy as np
import matplotlib; import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from tqdm import tqdm
from scipy.interpolate import griddata
import torchvision
from scipy.spatial import cKDTree
from torch.utils.tensorboard import SummaryWriter
import random
import os
from datetime import datetime
import json
import matplotlib.cm as cm
from tqdm import tqdm
import copy



class Experiment:
    """Experiment class makes it easier to do deep learning experiments. This class;
    - Encapsulates as much as possible information into one place, so makes it easier to track experiments
    - Easy to save and load
    - Possible to interrupt and continue training on the same machine or on another machine without losing any information or training states

    Here are some examples:

    -> For pretraining: Trains GCnet with 192 max disparity for 10 epoch on pretraining dataset.
    ```
        model = GCnet(192).to(device)
        e1 = t_utils.Experiment(name         = "GCnet-pretraining", 
                                description  = "Pretraining GCnet with maxdisp=192 for 10 epochs",
                                model        = model,
                                criterion    = nn.SmoothL1Loss(), 
                                scheduler    = None,
                                optimizer    = torch.optim.Adam(model.parameters(), lr=1e-4), 
                                train_loader = pretraining_train_dataloader, 
                                val_loader   = pretraining_val_dataloader, 
                                max_iter     = len(pretraining_train_dataset)*10,
                                val_interval = 2500,
                                vis_interval = 500,
                                save_interval= 5000,
                                device       = device)
        e1.train_model()
        e1.save()
    ```

    -> For finetuning: Loads pretrained PSMNet model and finetunes for 20000 iterations.
    ```
        checkpoint = torch.load("runs/PSM-pretraining-2021_09_22-08_31_27_192disp_10epoch_default/model_manual_save.pt")
        model = PSMNet(192).to(device)
        model.load_state_dict(checkpoint['model'])

        e1 = t_utils.Experiment(name         = "PSM-finetuning", 
                                description  = "Finetuning PSMNet with maxdisp=192 with lr=1e-5 on Pretrained model with 10 epochs.",
                                model        = model,
                                criterion    = nn.SmoothL1Loss(), 
                                scheduler    = None,
                                optimizer    = torch.optim.Adam(model.parameters(), lr=1e-5), 
                                train_loader = finetuning_train_dataloader, 
                                val_loader   = finetuning_val_dataloader, 
                                max_iter     = 20000,
                                val_interval = 200,
                                vis_interval = 100,
                                save_interval= 500,
                                device       = device)
        e1.train_model()
        e1.save()
    ```

    -> To Continue an Experiment: We made it easy to interrupt and continue the training later. name, description, etc. parameters are loaded automatically.
    ```
        model = PSMNet(192).to(device)
        e1 = t_utils.Experiment(model        = model,
                                criterion    = nn.SmoothL1Loss(), 
                                scheduler    = None,
                                optimizer    = torch.optim.Adam(model.parameters(), lr=1e-4), 
                                train_loader = finetuning_train_dataloader, 
                                val_loader   = finetuning_val_dataloader, 
                                device       = device)
        e1.load("runs/PSM-finetuning-2021_09_25-01_29_49_20000iter_lr1e-4/model_manual_save.pt")
        e1.train_model()
        e1.save()
    ```

    -> To Continue an Experiment More: Experiment parameters can be modified after loading.
    ```
        e1.max_iter = 30000
        e1.train_model()
        e1.save()
    ```

    Attributes:
        train_loader             :   Torchvision dataloader for training
        val_loader               :   Torchvision dataloader for validation
        optimizer                :   PyTorch optimizer
        criterion                :   PyTorch loss function. (e.g. nn.Smoothl1Loss)
        device                   :   "cpu", "cuda", "cuda:0", etc. where the training/validation/test will happen
        scheduler                :   PyTorch Scheduler for modifying learning rate
        name (optional)          :   Name of the model
        description (optional)   :   Experiment description
        model (optional)         :   PyTorch model
        max_iter (optional)      :   Maximum iteration. Training will stop when cur_iter reach this value.
        val_interval (optional)  :   Interval of validation steps in iterations. Setting this value too low may prolong the whole process.
        save_interval (optional) :   Interval of saving intermediate checkpoints in iterations. Setting this value too low may use a lot of storage.
        vis_interval (optional)  :   Interval of generating visualization for Tensorboard in iterations. Settings this value too low may use a lot of storage.
    """
    def __init__(self, train_loader, val_loader, optimizer, criterion, device, scheduler=None, 
                    name=None, description=None, model=None, max_iter=None, val_interval=100, save_interval=50, vis_interval=10):
        self.name = name
        self.description = description
        self.date_created = self.get_datetime_str()
        self.date_saved = None
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.max_iter = max_iter
        self.current_iter = 0
        self.device = device
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.vis_interval = vis_interval
        
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)
        
        self.train_loss = []
        self.val_loss =  []
        self.valid_acc = []
        self.lr_hist = []
        
        if self.name is not None:
            if(not os.path.exists("runs")):
                os.makedirs("runs")
            self.save_path = 'runs/' + self.name + "-" + self.date_created
            self.writer = SummaryWriter(self.save_path)   #Tensorboard writer
            print("Experiment Save Path:", self.save_path)

        vis_data_l = []
        vis_data_r = []
        vis_data_gt = []
        for i, (l,r,gt) in enumerate(self.val_loader):
            vis_data_l.append(l)
            vis_data_r.append(r)
            vis_data_gt.append(gt)
            if i == 3:
                break
        self.vis_data = ( torch.cat(vis_data_l, dim=0).to(device), 
                          torch.cat(vis_data_r, dim=0).to(device), 
                          torch.cat(vis_data_gt, dim=0).to(device) )

        
    def train_model(self):
        """Starts training with the given parameters until current iteration counter reaches to the maximum iteration counter.
        """
        
        progress_bar = tqdm(range(self.current_iter+1, self.max_iter+1))
        
        for itr in progress_bar:
            self.current_iter = itr

            sample = next(self.train_iter, None)
            if sample is None:
                self.train_iter = iter(self.train_loader)
                sample = next(self.train_iter, None)
                
            left_img, right_img, gt = sample    #gt: grount truth
            
            # training
            self.model.train()  # important for dropout and batch norms

            left_img = left_img.to(self.device)
            right_img = right_img.to(self.device)
            gt = gt.to(self.device)

            # Clear gradients w.r.t. the parameters
            self.optimizer.zero_grad()

            mask = torch.bitwise_and(gt >= 0, gt < self.model.maxdisp)           # use only non-negative disparities and that are smaller than maxdisp
            mask = torch.bitwise_and(mask , torch.bitwise_not(torch.isnan(gt)))  # kitti zero values will get here as NaN
            
            if torch.sum(mask) == 0:
                continue

            if self.model.name=="psm_stacked_hourglass":
                # Forward pass to get output/logits
                output1, output2, output3 = self.model(left_img, right_img)

                loss = 0.5*self.criterion(output1[mask], gt[mask]) + 0.7*self.criterion(output2[mask], gt[mask]) + self.criterion(output3[mask], gt[mask])

            else:
                # Forward pass to get output/logits
                outputs = self.model(left_img, right_img)

                # Calculate Loss
                loss = self.criterion(outputs[mask], gt[mask])

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            self.optimizer.step()

            # Update stats after the model update
            self.train_loss.append(loss.detach().cpu().numpy())
            self.writer.add_scalar('training loss', loss.detach().cpu().numpy(), itr)

            progress_bar.set_description(f"Iter {itr}: loss {loss.detach().cpu().numpy():.5f}. ")

            if self.scheduler is not None:
                self.lr_hist.append(self.scheduler.get_last_lr())
                self.writer.add_scalar('learning rate', np.array(self.scheduler.get_last_lr()), itr)
                self.scheduler.step()

            # VALIDATION
            if itr % self.val_interval == 0:
                
                self.model.eval()  # important for dropout and batch norms
                eval_result = self.eval_model()
                accuracy, loss = eval_result["accuracy"], eval_result["loss"]
            
                self.valid_acc.append(accuracy)
                self.val_loss.append(loss)
                
                self.writer.add_scalar('validation loss', loss, itr)
                self.writer.add_scalar('validation accuracy', accuracy, itr)

            # VISUALIZATION
            if itr % self.vis_interval == 0:
                with torch.no_grad():
                    self.model.eval()  # important for dropout and batch norms
                    left_img, right_img, gt = self.vis_data
                    model_out = self.model(left_img, right_img) # images should be already on the device
                    model_out = torch.cat([model_out, gt], dim=3) # put output and gt side by side
                    model_out = model_out / 128.0 # normalize data for better visualization (value set specifically for Kitti dataset)
                    model_out = np.moveaxis(cm.plasma(model_out.detach().cpu().numpy()), -1,2).squeeze(axis=1) # colorize outputs using colormap
                    vis_img = torchvision.utils.make_grid(torch.tensor(model_out), nrow=1)
                    self.writer.add_image('images', vis_img, itr)

            if itr % self.save_interval == 0 and itr>0:
                if(not os.path.exists(self.save_path+"/checkpoints")):
                    os.makedirs(self.save_path+"/checkpoints")
                self.save(self.save_path + "/checkpoints" + "/ep_" + str(itr) + ".pt")

            #print(f"Train loss: {round(mean_loss, 5)} Valid loss: {round(loss, 5)} Accuracy: {round(accuracy, 2)}%\n")
   

    @torch.no_grad()
    def eval_model(self, dataloader=None):
        """Starts evaluation/validation of the model. In dataloader is not given validation dataloader will be used by default.
        For accuracy metric 3-Pixel-Error is used. Loss value depends on the criterion.

        Returns:
            Dict: {"accuracy": XXX, "loss": XXX}
        """

        self.model.eval()
        if dataloader is None:
            dataloader = self.val_loader
        err = []
        loss = []
        for left_img,right_img,gt in dataloader:
            # inference
            model_out = self.model(left_img.to(self.device), right_img.to(self.device))

            # remove pixels where data is not present in the ground truth
            mask = torch.bitwise_and(gt >= 0, gt < self.model.maxdisp)           # use only non-negative disparities and that are smaller than maxdisp
            mask = torch.bitwise_and(mask , torch.bitwise_not(torch.isnan(gt)))  # kitti zero values will get here as NaN
            
            gt_masked = gt.detach().cpu().numpy()[mask]
            out_masked = model_out.detach().cpu().numpy()[mask]

            err.append( three_pixel_error(out_masked, gt_masked) )
            loss.append( self.criterion(torch.tensor(out_masked), torch.tensor(gt_masked)) )

        return {"accuracy": 1.0 - np.mean(err), "loss": np.mean(loss)}


    def save(self, path=None):
        """Saves the experiment as a file. But try to save whole folder to store tensorboard outputs as well.

        Args:
            path: (optional) Path of the experiment file (.pt). If not specified, save_path is used to find model_manual_save.pt file by default.

        Returns:
            dict: Short summary of loaded experiment checkpoint.
        """

        self.model.experiment_state = {}

        if path is None:
            path = self.save_path + "/model_manual_save.pt"

        # NOT ADDED:
        #self.train_loader
        #self.val_loader
        #self.train_iter
        #self.val_iter

        torch.save({
            'name': self.name,
            'description': self.description,
            'date_created': self.date_created,
            'date_saved': self.get_datetime_str(),
            'max_iter': self.max_iter,
            'current_iter': self.current_iter,
            'val_interval': self.val_interval,
            'save_interval': self.save_interval,
            'vis_interval': self.vis_interval,
            'train_loss': list(self.train_loss),
            'val_loss': list(self.val_loss),
            'valid_acc': list(self.valid_acc),
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)

        self.name = checkpoint["name"]
        self.description = checkpoint["description"]
        self.date_created = checkpoint["date_created"]
        self.date_saved = checkpoint["date_saved"]
        self.max_iter = checkpoint["max_iter"]
        self.current_iter = checkpoint["current_iter"]
        self.val_interval = checkpoint["val_interval"]
        self.save_interval = checkpoint["save_interval"]
        self.vis_interval = checkpoint["vis_interval"]
        self.train_loss = checkpoint["train_loss"]
        self.val_loss = checkpoint["val_loss"]
        self.valid_acc = checkpoint["valid_acc"]

        self.save_path = os.path.dirname(path)
        self.writer = SummaryWriter(self.save_path)   #Tensorboard writer

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        short_checkpoint = checkpoint.copy()
        del short_checkpoint["model"]
        del short_checkpoint["optimizer"]
        del short_checkpoint["train_loss"]
        del short_checkpoint["val_loss"]
        del short_checkpoint["valid_acc"]
        short_checkpoint["last_train_loss"] = self.train_loss[-1] if len(self.train_loss)>0 else None
        short_checkpoint["last_val_loss"] = self.val_loss[-1] if len(self.val_loss)>0 else None
        short_checkpoint["last_valid_acc"] = self.valid_acc[-1] if len(self.valid_acc)>0 else None
        short_checkpoint["optimizer"] = str(self.optimizer)
        return short_checkpoint

    def get_datetime_str(self):
        return str( datetime.now().strftime("%Y_%m_%d-%H_%M_%S") )


def three_pixel_error(out, gt):
    """Defines 3PE standard disparity error metric. 
    Parameters can be numpy.ndarray or torch.tensor. 
    Make sure to detach() and move input data to cpu() before calling this function.

    Args:
        out: Model disparity output.
        gt: Ground truth disparity image.

    Returns
        3-pixel-error between 0.0 and 1.0 (being 10 highest possible error)
    """

    mask1 = np.absolute(gt-out) < 3
    mask2 = np.absolute(gt-out) < 0.05 * gt

    mask = np.bitwise_or(mask1, mask2)
    N = np.prod(gt.shape)
    if N == 0: # perfect score
        return 0.
    return 1. - 1./N * np.sum(np.array(mask))


#########################################################
##### PLOT ##############################################
#########################################################


def smooth(f, p=0.1):
    """Smoothing a function using a low-pass filter (mean) of size K. 
       But here we are using `p` as a parameter which defines K using the size of f.

    Args:
        f: Input loss, accuracy, etc. as an array
        p: Smoothing parameter between 0 and 1.

    Returns:
        smoothed f
    """

    K = int(len(f)*p)

    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]])  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f


def plot_loss_train_val(loss_iters, train_loss, val_loss, valid_acc):
    """
    Plots 3 graphs; loss/iteraions, train_loss/epoch, val_loss/epoch respectively and indicates validation accuracy
    """

    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(24,5)

    smooth_loss = smooth(loss_iters)
    ax[0].plot(loss_iters, c="blue", label="Loss", linewidth=0.5, alpha=0.5)
    ax[0].plot(smooth_loss, c="red", label="Smoothed Loss", linewidth=1, alpha=1)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training Progress")
    ax[0].set_yscale("log")

    epochs = np.arange(len(train_loss)) + 1
    ax[1].plot(epochs[1:], train_loss[1:], c="red", label="Train Loss", linewidth=3)
    ax[1].plot(epochs[1:], val_loss[1:], c="blue", label="Valid Loss", linewidth=3)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Loss Curves")

    epochs = np.arange(len(val_loss)) + 1
    ax[2].plot(epochs[1:], valid_acc[1:], c="red", label="Valid accuracy", linewidth=3)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Accuracy (%)")
    ax[2].set_title(f"Validation Accuracy (max={round(np.max(valid_acc),2)}% @ epoch {np.argmax(valid_acc)+1})") # todo: max?

    #plt.show()
    return fig


def interpolate_zerofilled_image(img, method='cubic', interpolate_on_depth=False, max_neighbor_dist=2, invalid_fill_val=np.nan):
    """Interpolates single channel input image on 2D and discards interpolated points far away from the input points.

    Notes:
        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

    Args:
        img: Input disparity map as numpy.ndarray with shape of [height, width]
        method: Can be "linear", "nearest", or "cubic". If max_neighbor_dist is very small will not affect the outcome noticably.
        interpolate_on_depth: Applies interpolation on 1.0/img and turns it back. Not recommended, needs more work.
        max_neighbor_dist: Interpolated points which are far away from the original points will be discarded. In pixels.
        invalid_fill_val: This value will be set for discarded values
    Returns:
        True if successful, False otherwise.
    """

    img = img.copy()
    points = []
    values = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not np.isnan(img[i,j]):
                points.append([i,j])
                values.append(img[i,j])

    if interpolate_on_depth:
        for i,v in enumerate(values):
                values[i] = 1.0 / v

    grid_x, grid_y = np.mgrid[0:img.shape[0],0:img.shape[1]]
    interpolated_img = griddata(points, values, (grid_x, grid_y), method=method, fill_value=invalid_fill_val)

    if interpolate_on_depth:
        mask = interpolated_img > 0
        interpolated_img[mask] = 1.0 / interpolated_img[mask]

    tree = cKDTree(points)
    q = np.array((grid_x, grid_y)).reshape(2,-1).T
    k=1
    dist, _ = tree.query(q, k=k, workers=-1)
    if k>1:
        dist = dist.reshape((*interpolated_img.shape, -1))[:,:,-1].reshape(interpolated_img.shape)
    else:
        dist = dist.reshape(interpolated_img.shape)

    if max_neighbor_dist > 0:
        interpolated_img[dist > max_neighbor_dist] = invalid_fill_val

    return interpolated_img, dist



def show_model_outputs(model, dataloader, device, count=5, start_idx=4, stride_idx=4, interpolate_gt=True, axis_off=True, save_path=None, unnormalize=True):
    """Shows input, ground truth and model output in single row. Can be stack with increasing count parameter.

    Note:
        Results should look similar to evaluation of algorithms in http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo

    Args:
        model: The first parameter.
        dataloader: The second parameter.
        device: "cpu" or "cuda"
        count: Number of stacked figures.
        start_idx: Start index in the dataloader will be start_idx+stride_idx. Used for sampling different data.
        stride_idx: Defines steps between each sample. Used for sampling different data.
        interpolate_gt: Interpolation on ground truth. Recommended.
        axis_off: Turns off axis and makes it clear in the figure.
        save_path: Image save path with .svg extension.
        unnormalize: Unnormalizes input RGB data for visualization according to recommended torchvision.models normalization parameter.
    """

    interpolate_method="nearest"
    rainbow_cmap = copy.copy(matplotlib.cm.get_cmap("gist_rainbow_r"))
    rainbow_cmap.set_bad('black',1.)

    scale = 2
    fig, axs = plt.subplots(count,3, figsize=(10*scale,1.2*scale*count))

    model.eval()
    dataloader_iter = iter(dataloader)

    for i in range(start_idx):
        next(dataloader_iter)

    for i in range(count):
        for j in range(stride_idx):
            next(dataloader_iter)

        with torch.no_grad():
            l_,r_,gt = next(dataloader_iter)
            out = model(l_.to(device),r_.to(device))
            out_ = out.detach().cpu().numpy().copy()
            gt = gt.detach().cpu().numpy().copy()

        if unnormalize:
            unnormalizer_std = torchvision.transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225])
            unnormalizer_mean = torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
            l = unnormalizer_mean(unnormalizer_std(l_))
            r = unnormalizer_mean(unnormalizer_std(r_))
        else:
            l = l_
            r = r_

        gt = gt[0].squeeze(axis=0)
        out_ = out_[0].squeeze(axis=0)

        if interpolate_gt:
            gt,_ = interpolate_zerofilled_image(gt, method=interpolate_method)

        # Keep the same normalization for both images
        mask = np.bitwise_not( np.isnan(gt) )
        vmin = np.min([np.min(out_[mask]), np.min(gt[mask])])
        vmax = np.max([np.max(out_[mask]), np.max(gt[mask])])

        axs[i,0].grid(False)
        axs[i,0].imshow(l[0].squeeze(axis=0).moveaxis(0,2))

        axs[i,1].grid(False)
        axs[i,1].imshow(gt, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)

        axs[i,2].grid(False)
        axs[i,2].imshow(out_, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)

        if i==0:
            axs[i,0].set_title("Input (Image Left)").set_fontsize(8*scale)
            axs[i,1].set_title("Groundtruth").set_fontsize(8*scale)
            axs[i,2].set_title("Model Output").set_fontsize(8*scale)


        if axis_off:
            axs[i,0].set_axis_off()
            axs[i,1].set_axis_off()
            axs[i,2].set_axis_off()

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg")


def show_model_error(model, dataloader, device, count=1, interpolate_err=True, interpolate_gt=True, pixel_error=3, axis_off=True, save_path=None, unnormalize=True):
    """Shows input, ground truth, error map and model output in two rows. Can be generated multiple figures with setting count >1

    Note:
        Results should look similar to evaluation of algorithms in http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo

    Args:
        model: The first parameter.
        dataloader: The second parameter.
        device: "cpu" or "cuda"
        count: Number of stacked figures.
        interpolate_err: Interpolation on error map. Recommended.
        interpolate_gt: Interpolation on ground truth. Recommended.
        pixel_error: Yellow color on error map will define this amount of pixel errors. 3PE by default.
        axis_off: Turns off axis and makes it clear in the figure.
        save_path: Image save path with .svg extension.
        unnormalize: Unnormalizes input RGB data for visualization according to recommended torchvision.models normalization parameter.
    """

    interpolate_method="nearest"
    rainbow_cmap = copy.copy(matplotlib.cm.get_cmap("gist_rainbow_r"))
    rainbow_cmap.set_bad('black',1.)
    err_cmap = copy.copy(matplotlib.cm.get_cmap("RdYlBu_r"))
    err_cmap.set_bad('black',1.)

    model.eval()
    dataloader_iter = iter(dataloader)
    for i in range(count):
        with torch.no_grad():
            l_,r_,gt = next(dataloader_iter)
            out = model(l_.to(device),r_.to(device))
            out_ = out.detach().cpu().numpy().copy()
            gt = gt.detach().cpu().numpy().copy()

        if unnormalize:
            unnormalizer_std = torchvision.transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225])
            unnormalizer_mean = torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
            l = unnormalizer_mean(unnormalizer_std(l_))
            r = unnormalizer_mean(unnormalizer_std(r_))
        else:
            l = l_
            r = r_

        gt = gt[0].squeeze(axis=0)
        out_ = out_[0].squeeze(axis=0)

        if interpolate_err:
            err = np.absolute(out_ - gt)
            err,_ = interpolate_zerofilled_image(err, method=interpolate_method)

        if interpolate_gt:
            gt,_ = interpolate_zerofilled_image(gt, method=interpolate_method)

        # Keep the same normalization for both images
        mask = np.bitwise_not( np.isnan(gt) )
        vmin = np.min([np.min(out_[mask]), np.min(gt[mask])])
        vmax = np.max([np.max(out_[mask]), np.max(gt[mask])])

        scale = 2
        fig, axs = plt.subplots(2,2, figsize=(10*scale,3.5*scale))
        axs[0,0].grid(False)
        axs[0,0].imshow(l[0].squeeze(axis=0).moveaxis(0,2))
        axs[0,0].set_title("Input (Image Left)").set_fontsize(8*scale)

        axs[1,1].grid(False)
        axs[1,1].imshow(out_, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
        axs[1,1].set_title("Model Output").set_fontsize(8*scale)
        
        axs[1,0].grid(False)
        axs[1,0].imshow(err, cmap=err_cmap, vmin=0, vmax=pixel_error)
        axs[1,0].set_title("Error".format(pixel_error)).set_fontsize(8*scale)
        for _, spine in axs[1,0].spines.items():
            spine.set_visible(True)

        axs[0,1].grid(False)
        axs[0,1].imshow(gt, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
        axs[0,1].set_title("Groundtruth").set_fontsize(8*scale)

        if axis_off:
            axs[0,0].set_axis_off()
            axs[0,1].set_axis_off()
            axs[1,0].set_axis_off()
            axs[1,1].set_axis_off()

        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path + "/show_model_error{}.svg".format(i), format="svg")


def show_compare_model_error(model1, model2, name1, name2, dataloader, device, idx=0, interpolate_err=True, interpolate_gt=True, pixel_error=3, axis_off=True, save_path=None, unnormalize=True):
    """Shows input RGB, ground truth and compares models' outputs in two rows while showing error maps.

    Note:
        Results should look similar to evaluation of algorithms in http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo

    Args:
        model1, model2: First and second model to compare.
        name1, name2: Names of the parameter written in the plot
        dataloader: The second parameter.
        device: "cpu" or "cuda"
        idx: Defines which data will be used in the dataloader.
        interpolate_gt: Interpolation on ground truth. Recommended.
        axis_off: Turns off axis and makes it clear in the figure.
        save_path: Image save path with .svg extension.
        unnormalize: Unnormalizes input RGB data for visualization according to recommended torchvision.models normalization parameter.
    """

    interpolate_method="nearest"
    rainbow_cmap = copy.copy(matplotlib.cm.get_cmap("gist_rainbow_r"))
    rainbow_cmap.set_bad('black',1.)
    err_cmap = copy.copy(matplotlib.cm.get_cmap("RdYlBu_r"))
    err_cmap.set_bad('black',1.)

    dataloader_iter = iter(dataloader)
    for i in range(idx):
        next(dataloader_iter)

    model1.eval()
    model2.eval()
    with torch.no_grad():
        l_,r_,gt = next(dataloader_iter)
        gt = gt.detach().cpu().numpy().copy()
        out1 = model1(l_.to(device),r_.to(device))
        out1_ = out1.detach().cpu().numpy().copy()
        out2 = model2(l_.to(device),r_.to(device))
        out2_ = out2.detach().cpu().numpy().copy()

    if unnormalize:
        unnormalizer_std = torchvision.transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225])
        unnormalizer_mean = torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        l = unnormalizer_mean(unnormalizer_std(l_))
        r = unnormalizer_mean(unnormalizer_std(r_))
    else:
        l = l_
        r = r_

    gt = gt[0].squeeze(axis=0)
    out1_ = out1_[0].squeeze(axis=0)
    out2_ = out2_[0].squeeze(axis=0)

    if interpolate_err:
        err1 = np.absolute(out1_ - gt)
        err1,_ = interpolate_zerofilled_image(err1, method=interpolate_method)
        err2 = np.absolute(out2_ - gt)
        err2,_ = interpolate_zerofilled_image(err1, method=interpolate_method)

    if interpolate_gt:
        gt,_ = interpolate_zerofilled_image(gt, method=interpolate_method)

    # Keep the same normalization for both images
    mask = np.bitwise_not( np.isnan(gt) )
    vmin = np.min([np.min(out1_[mask]), np.min(out2_[mask]), np.min(gt[mask])])
    vmax = np.max([np.max(out1_[mask]), np.max(out2_[mask]), np.max(gt[mask])])

    scale = 2
    fig, axs = plt.subplots(3,2, figsize=(10*scale,5*scale))
    axs[0,0].grid(False)
    axs[0,0].imshow(l[0].squeeze(axis=0).moveaxis(0,2))
    axs[0,0].set_title("Input (Image Left)").set_fontsize(8*scale)

    axs[0,1].grid(False)
    axs[0,1].imshow(gt, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    axs[0,1].set_title("Groundtruth").set_fontsize(8*scale)

    axs[1,1].grid(False)
    axs[1,1].imshow(out1_, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    axs[1,1].set_title(name1 + " Output").set_fontsize(8*scale)
    
    axs[1,0].grid(False)
    axs[1,0].imshow(err1, cmap=err_cmap, vmin=0, vmax=pixel_error)
    axs[1,0].set_title(name1 + " Error".format(pixel_error)).set_fontsize(8*scale)
    for _, spine in axs[1,0].spines.items():
        spine.set_visible(True)

    axs[2,1].grid(False)
    axs[2,1].imshow(out2_, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    axs[2,1].set_title(name2 + " Output").set_fontsize(8*scale)
    
    axs[2,0].grid(False)
    axs[2,0].imshow(err2, cmap=err_cmap, vmin=0, vmax=pixel_error)
    axs[2,0].set_title(name2 + " Error".format(pixel_error)).set_fontsize(8*scale)
    for _, spine in axs[1,0].spines.items():
        spine.set_visible(True)

    if axis_off:
        axs[0,0].set_axis_off()
        axs[0,1].set_axis_off()
        axs[1,0].set_axis_off()
        axs[1,1].set_axis_off()
        axs[2,0].set_axis_off()
        axs[2,1].set_axis_off()

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, format="svg")


def imshow_pred_gt_disparity(pred_disp, gt_disp):

    fig, axs = plt.subplots(1,2, figsize=(17,6))

    axs[0].imshow(pred_disp.moveaxis(0,2), cmap="gray")
    axs[1].imshow(gt_disp.moveaxis(0,2), cmap="gray", interpolation="nearest")

    axs[0].set_title("Predicted Disparity")
    axs[0].grid(False)
    axs[1].set_title("True Disparity")
    axs[1].grid(False)

    fig.tight_layout()

    return fig, axs


#########################################################
##### UTILS #############################################
#########################################################

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


def get_device():
    """
    Returns CUDA device if possible, otherwise returns cpu

    Returns:
        device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

@torch.no_grad()
def evaluate_model(model, test_dataloader, device, criterion):
    """Evaluates model and computes EP3 and loss (according to the criterion). 
    Similar to eval_model in Experiment but can run without the whole Class.
    Only used for notebook cells.

    Args:
        model: PyTorch model to evaluate
        test_dataloader: Dataloader for feeding inputs to the model.
        device: "cpu" or "cuda"
        criterion: Loss criterion function or loss object

    Returns:
        None. Only prints mean 3PE and mean loss
    """

    three_PE = []
    loss = []
    model.eval()
    for left_img, right_img, disparity in tqdm(test_dataloader):
        pred_disp = model(left_img.to(device), right_img.to(device)).detach().cpu()

        # remove pixels where data is not present in the ground truth
        mask = torch.bitwise_and(disparity >= 0, disparity < model.maxdisp)      # use only non-negative disparities and that are smaller than maxdisp
        mask = torch.bitwise_and(mask , torch.bitwise_not(torch.isnan(disparity)))  # kitti zero values will get here as NaN

        three_PE.append(three_pixel_error(pred_disp[mask], disparity[mask]))
        loss.append(criterion(pred_disp[mask], disparity[mask]).numpy())

    mean_threePE = np.mean(np.array(three_PE))
    mean_loss = np.mean(np.array(loss))
    print("mean 3PE: {}%".format(round(mean_threePE*100, 2)))
    print("mean Loss: ", round(mean_loss,4))




#########################################################
##### NOT USED ##########################################
#########################################################

"""NOTE ABOUT THIS SECTION:
We tried interpolation on disparity ground truths and merging with pretrained model outputs to generate dense finetuning dataset for models.
But the results was not good.
"""

def compute_augmented_dataset(dataset, device):
    new_dataset = []
    for l,r,gt in tqdm(dataset):
        aug_gt, dist = interpolate_zerofilled_image(np.array(gt), method="cubic", max_neighbor_dist=3)
        new_dataset.append((l.copy(), r.copy(), np.array(aug_gt)))
    return new_dataset


class AugKittiDataset():    
    # first 150 samples are for training, last 50 are for evaluation
    def __init__(self, dataset_arr, transforms=None):
        self.name = "Kitti 2015 Dataset (Augmented)"
        self.dataset_arr = dataset_arr
        self.transforms = transforms
        
    def __len__(self):
        return len(self.dataset_arr)

    def __getitem__(self, idx):
        left, right, disparity = self.dataset_arr[idx]
        disparity = np.array(disparity)
        
        if(self.transforms is not None):
            left, right, disparity = self.transforms((left, right, disparity))

        return left, right, disparity