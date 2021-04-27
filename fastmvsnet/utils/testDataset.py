import cv2
import torch
import os
import fastmvsnet.utils.io as io
from torch.utils.data import Dataset, DataLoader


class testDataset(Dataset):
    # mean = torch.tensor([1.97145182, -1.52387525, 651.07223895])
    # std = torch.tensor([84.45612252, 93.22252387, 80.08551226])

    def __init__(self, rootdir, num_view, interval_scale=None, num_virtual_plane=None):
        self.rootdir = rootdir
        self.num_view = num_view
        self.interval_scale = interval_scale
        self.num_virtual_plane = num_virtual_plane
        self.paths = self._load_dataset(num_view)

    def __getitem__(self, index):
        path = self.paths[index]
        for view in range(self.num_view):
            try:
                image = cv2.imread(path["image_file"])
                cam = io.load_cam_dtu(open(path["cam_file"]),
                                      self.num_virtual_plane, self.interval_scale)
            except:
                print(path +"Wrong")


    def __len__(self):
        return len(self.paths)  # 视角个数（实验只有一组，按照同种光照条件）==num_view

    def _load_dataset(self, num_view):
        paths = []
        path = {}

        image_folder = self.root_dir + "/Rectified/"
        cam_folder = self.root_dir + "/Cameras/"
        depth_folder = self.root_dir + "/Depths/"
        for viewnum in num_view:
            image_file = image_folder + "/{:02d}.jpg".format(viewnum)
            cam_file = cam_folder + "/{:02d}.jpg".format(viewnum)
            depth_file = depth_folder + "/{:02d}.jpg".format(viewnum)
            path["image_file"] = image_file
            path["cam_file"] = cam_file
            path["depth_file"] = depth_file
            paths.append(path)

        return paths



if __name__ == "main":
    root_path = "E:\dataset\dtu_training\dtu_training\mvs_training\dtu"
    dataset = testDataset()
    for i, (data, label) in enumerate(dataset):
        pass