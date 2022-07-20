import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from pathlib import Path
import numpy
from PIL import Image
from splendor.image import save_image
import splendor

from ltron.config import Config
from ltron.dataset.paths import get_dataset_info, get_dataset_paths

from ltron_torch.gym_tensor import (
    gym_space_to_tensors, default_tile_transform, default_image_transform, default_image_untransform)
import pdb
import json

class rolloutFramesConfig(Config):
    train_split = 'rollouts_frames'
    train_subset = None
    start = 0
    batch_size = 4
    epochs = 4
    checkpoint_frequency = 10000000
    test_frequency = 100
    visualization_frequency = 100
    end = None
    file_override = None
    loader_workers = 0
    dataset = "omr_clean"
    test_split = "rollouts_test"
    optimizer = "adamw"
    learning_rate = 3e-4
    test_subset = None

class rolloutFrames(Dataset):

    def id_mapping(self, arr, id_map):
        return numpy.vectorize(id_map.__getitem__)(arr)

    def color_mapping(self, arr, id_map):
        return numpy.vectorize(id_map.__getitem__)(arr)

    def __init__(self, dataset, split, subset, transform=None):

        self.transform = transform
        self.table = []
        # self.pos_snap = []
        # self.neg_snap = []
        # self.mask = []
        self.stack_label = []
        dataset_paths = get_dataset_paths(dataset, split, subset=subset)
        self.rollout_paths = dataset_paths[split]

    def __len__(self):
        return len(self.rollout_paths)

    def __getitem__(self, i):
        
        path = self.rollout_paths[i]
        rollout = numpy.load(path, allow_pickle=True)['rollout'].item()
        table = rollout['table_color_render'].astype(float)
        pos_snap_reduced = numpy.where(rollout['table_pos_snap_render'][:, :, 0] > 0, 1, 0)
        neg_snap_reduced = numpy.where(rollout['table_neg_snap_render'][:, :, 0] > 0, 1, 0)
        shape_ids = self.id_mapping(rollout['table_mask_render'], rollout['config']['shape'])
        color_ids = self.color_mapping(rollout['table_mask_render'], rollout['config']['color'])
        stacked_label = numpy.stack([shape_ids, pos_snap_reduced, neg_snap_reduced, color_ids], axis=2)
        # if numpy.unique(color_ids).shape[0] > 2:
        #     pdb.set_trace()
        # if numpy.sum(rollout['assembly']['class'] > 0) > len(numpy.unique(rollout['assembly']['class']))-1:
        # pdb.set_trace()
        # if "Wright Flyer" not in str(path) and "Metroliner" not in str(path) and "Darth Maul" not in str(path) and\
        #         "Rebel Blockade Runner" not in str(path) and "Imperial Star Destroyer" not in str(path) and 2020 in rollout['assembly']['class']:
        #     pdb.set_trace()
        # if 1627 in rollout['assembly']['class'] and "Darth Maul" not in str(path):
        #     pdb.set_trace()

        if self.transform is not None:
            table = self.transform(table)
            return table, stacked_label

        return table, stacked_label

def build_rolloutFrames_train_loader(config, batch_overload=None, shuffle=True):
    print('-'*80)
    print("Building single frame data loader")
    dataset = rolloutFrames(
            config.dataset,
            config.train_split,
            config.train_subset,
            default_image_transform,
    )

    loader = DataLoader(
            dataset,
            batch_size = batch_overload if batch_overload else config.batch_size,
            num_workers = config.loader_workers,
            shuffle=shuffle,
    )
    
    return loader

def build_rolloutFrames_test_loader(config, batch_overload=None):
    print('-'*80)
    print("Building single frame test data loader")
    dataset = rolloutFrames(config.dataset, config.test_split, config)


def main():
    config = rolloutFramesConfig.load_config("../../experiments/pretrainbackbone_resnet/settings.cfg")
    loader = build_rolloutFrames_train_loader(config, 1)
    counter = 1
    for table, label in loader:
        # print(table.type())
        # print(label.type())
        # pdb.set_trace()
        image = numpy.transpose(table[0].squeeze().detach().cpu().numpy(), [1,2,0])
        im = numpy.uint8(image * 255)
        im = default_image_untransform(table[0])
        # save_image(im, "test_dataset/test_im" + str(counter) + ".png")
        mask = label[0, :, :, 0].squeeze().detach().cpu().numpy()
        mask = numpy.uint8(mask * 255)
        # save_image(mask, "test_dataset/test_mask" + str(counter) + ".png")
        # print(numpy.where(label[0, :, :, 0] > 0))
        # print(numpy.where(label[0, :, :, 1] > 0))
        # print(table.shape)
        # print(label.shape)
        # print(numpy.unique(label[0, :, :, 0]))
        # print(numpy.unique(label[0, :, :, 1]))
        # print(numpy.unique(label[0, :, :, 2]))
        counter += 1
        if counter >= 100000:
            break


if __name__ == '__main__' :
    main()



