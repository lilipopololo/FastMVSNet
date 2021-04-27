import cv2
import torch
import os

from torch import nn
import numpy as np
import fastmvsnet.utils.io as io
from fastmvsnet.model import build_pointmvsnet as build_model, FastMVSNet
from torch.utils.data import Dataset, DataLoader


class testDataset(Dataset):
    # mean = torch.tensor([1.97145182, -1.52387525, 651.07223895])
    # std = torch.tensor([84.45612252, 93.22252387, 80.08551226])

    def __init__(self, height, width, rootdir, num_view, interval_scale=None, num_virtual_plane=None):
        self.rootdir = rootdir
        self.num_view = num_view
        self.interval_scale = interval_scale
        self.num_virtual_plane = num_virtual_plane
        self.pathlist = self._load_dataset(num_view)

        self.height = height
        self.width = width

    def __getitem__(self, index):
        path = self.pathlist[index]
        images = []
        cams = []
        data
        for view in range(self.num_view):
            try:
                image = cv2.imread(path["image_file"])
                cam = io.load_cam_dtu(open(path["cam_file"]),
                                      self.num_virtual_plane, self.interval_scale)
                images.append(image)
                cams.append(cam)
            except:
                print(path +"Wrong")
        img_list = np.stack(images, axis = 0)
        cams_list = np.stack(cams,axis = 0)
        img_list = torch.tensor(img_list).permute(0, 3, 1, 2).float()
        cams_list = torch.tensor(cams_list).float()
        return {
                    images:img_list,
                    cams:cams_list
                }

    def __len__(self):
        return len(self.paths)  # 只有1（实验只有一组，按照同种光照条件）

    def _load_dataset(self, num_view):
        pathlist = []
        path = {}
        image_file=[]
        cam_file=[]
        depth_file=[]
        image_folder = self.root_dir + "/Rectified/"
        cam_folder = self.root_dir + "/Cameras/"
        depth_folder = self.root_dir + "/Depths/"
        for viewnum in num_view:
            image_file.append(image_folder + "/{:02d}.jpg".format(viewnum))
            cam_file.append(cam_folder + "/{:02d}.jpg".format(viewnum))
            depth_file.append(depth_folder + "/{:02d}.jpg".format(viewnum))

        path["image_file"] = image_file
        path["cam_file"] = cam_file
        path["depth_file"] = depth_file
        pathlist.append(path)

        return pathlist



if __name__ == "main":
    root_path = "E:\dataset\dtu_training\dtu_training\mvs_training\dtu"
    dataset = testDataset()
    net = FastMVSNet(
        img_base_channels=8,
        vol_base_channels=8,
        flow_channels=8,
    )
    model = nn.DataParallel(net).cuda()
    model.load_state_dict("D:\srccode\FastMVSNet\outputs\pretrained.pth")
    with torch.no_grad():
        for i, data in enumerate(dataset):
            {k: v.cuda(non_blocking=True) for k, v in data.items() if isinstance(v, torch.Tensor)}
            preds = model(data, 1, inter_scales=4.24, isGN=True, isTest=True)
            init_depth_map = preds["coarse_depth_map"].cpu().numpy()[0, 0]
            init_prob_map = preds["coarse_prob_map"].cpu().numpy()[0, 0]
            io.write_pfm("./init_depth_map.pfm", init_depth_map)
            io.write_pfm("./init_prob_map.pfm", init_prob_map)
            interval_list = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
            interval_list = np.reshape(interval_list, [1, 1, -1])
            for i, k in enumerate(preds.keys()):
                if "flow" in k:
                    if "prob" in k:
                        out_flow_prob_map = preds[k][0].cpu().permute(1, 2, 0).numpy()
                        num_interval = out_flow_prob_map.shape[-1]
                        assert num_interval == interval_list.size
                        pred_interval = np.sum(out_flow_prob_map * interval_list, axis=-1) + 2.0
                        pred_floor = np.floor(pred_interval).astype(np.int)[..., np.newaxis]
                        pred_ceil = pred_floor + 1
                        pred_ceil = np.clip(pred_ceil, 0, num_interval - 1)
                        pred_floor = np.clip(pred_floor, 0, num_interval - 1)
                        prob_height, prob_width = pred_floor.shape[:2]
                        prob_height_ind = np.tile(np.reshape(np.arange(prob_height), [-1, 1, 1]), [1, prob_width, 1])
                        prob_width_ind = np.tile(np.reshape(np.arange(prob_width), [1, -1, 1]), [prob_height, 1, 1])

                        floor_prob = np.squeeze(out_flow_prob_map[prob_height_ind, prob_width_ind, pred_floor], -1)
                        ceil_prob = np.squeeze(out_flow_prob_map[prob_height_ind, prob_width_ind, pred_ceil], -1)
                        flow_prob = floor_prob + ceil_prob
                        io.write_pfm("./flow_prob.pfm",flow_prob)