import sys
sys.path.insert(0, './renderer')
sys.path.insert(0, '../')

from collections import namedtuple
import render_car_instances as rci
import utils.utils as uts
import glob
import os

import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
%matplotlib inline

import ntpath
pattern = '*%s.%s' % ('_Camera_5', 'jpg')


m PIL import Image
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
import utils.utils as uts

splits = ['train']

for split in splits:
    Setting = namedtuple('Setting', ['data_dir'])
    setting = Setting('../apolloscape/full/%s/'%split)

    visualizer = rci.CarPoseVisualizer(setting, scale=0.2)
    visualizer.load_car_models()
    apollo_images_root = '/home/zhurui/Documents/baidu/personal-code/car-fitting/rui_modelfitting/dataset-api/apolloscape/full/%s/images'%split
    search_files = os.path.join(apollo_images_root, pattern)
    filenames = sorted(glob.glob(search_files))

    for file_idx, filename in enumerate(filenames):
        print '--- %d/%d...'%(file_idx, len(filenames))
        image_name = ntpath.basename(filename).replace('.jpg', '')
        folder_name = 'pose_maps_02'
        plot_path = filename.replace('images', folder_name).replace('.jpg', '_plot.jpg')
        image_vis, seg_array, depth, pose_map, image_rescaled, pose_list = visualizer.showAnn(image_name, if_visualize=file_idx<20, if_save=True, plot_path=plot_path)

        print pose_map.shape, pose_map.dtype

        image_rescaled = Image.fromarray(np.uint8(image_rescaled))
        image_rescaled_name = filename.replace('images', folder_name).replace('.jpg', '_rescaled.png')
        image_rescaled.save(image_rescaled_name)

        image_vis = Image.fromarray(np.uint8(image_vis))
        image_vis_name = filename.replace('images', folder_name).replace('.jpg', '_vis.png')
        image_vis.save(image_vis_name)

        seg = Image.fromarray(np.uint8(seg_array))
        seg_name = filename.replace('images', folder_name).replace('.jpg', '_seg.png')
        seg.save(seg_name)

        pose_dict_path = filename.replace('images', folder_name).replace('.jpg', '_posedict.npy')
        pose_list_array = np.asarray(pose_list)
        np.save(pose_dict_path, np.float32(pose_list_array))

''' Demo code for converting seg_map and pose_dict to posemap in Python (Tensorflow).
        pose_dict: [N, 6], N is the number of classes in the seg_map, including the background (denoted as 0 in seg_map).
        seg: [H, W, 1], H is the image height, W is the image width, and one channel encoding the instance label (0 for background).'''

'''
seg_one_hot = tf.one_hot(tf.reshape(seg, [-1]), depth=tf.shape(pose_dict)[0])
pose_map = tf.matmul(seg_one_hot, pose_dict)
pose_map = tf.reshape(pose_map, [H, W, 6])
'''
