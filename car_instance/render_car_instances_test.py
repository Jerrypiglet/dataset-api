"""
    Brief: Demo for render labelled car 3d poses to the image
    Author: wangpeng54@baidu.com
    Date: 2018/6/10
"""

import argparse
import cv2
import car_models
import data
import numpy as np
import json
import pickle as pkl
from off_utils import Mesh
import sys
sys.path.insert(0, '../')
import os

import renderer.render_egl as render
import utils.utils as uts
import utils.eval_utils as eval_uts
import logging

from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class CarPoseVisualizer(object):
    def __init__(self, args=None, scale=0.5, codes=None, linewidth=0.):
        """Initializer
        Input:
            scale: whether resize the image in case image is too large
            linewidth: 0 indicates a binary mask, while > 0 indicates
                       using a frame.
        """
        self.dataset = data.ApolloScape(args)
        self._data_config = self.dataset.get_3d_car_config()

        self.MAX_DEPTH = 1e4
        self.MAX_INST_NUM = 100
        h, w = self._data_config['image_size']
        self.image_size = np.uint32(uts.round_prop_to(
            np.float32([h * scale, w * scale])))
        self.scale = scale
        self.linewidth = linewidth
        self.colors = np.random.random((self.MAX_INST_NUM, 3)) * 255

        if codes is None:
            self.codes = codes

    def load_car_models(self, car_model_dir=None):
        """Load all the car models
        """
        self.car_models = OrderedDict([])
        self.car_models_all = []
        # car_model_dir = self._data_config['car_model_dir'] if car_model_dir==None else self._data_config['car_model_dir']
        car_model_dir = self._data_config['car_model_dir']
        logging.info('loading %d car models from %s...' % (len(car_models.models), car_model_dir))
        for model in car_models.models:
            car_model = '%s/%s.pkl' % (car_model_dir,
                                       model.name)
            with open(car_model) as f:
                self.car_models[model.name] = pkl.load(f)
                # fix the inconsistency between obj and pkl
                self.car_models[model.name]['vertices'][:, [0, 1]] *= -1
                self.car_models[model.name]['vertices'] = self.car_models[model.name]['vertices']
                self.car_models_all.append(self.car_models[model.name]['vertices'])
        alls = np.concatenate(self.car_models_all, axis=0)
        # print np.amax(alls, axis=0), np.amin(alls, axis=0)

    def load_car_models_off(self, off_path, postfix = ''):
        """Load all the car models
        """
        self.car_models = OrderedDict([])
        self.car_models_all = []
        logging.info('loading %d car models' % len(car_models.models))
        for model in car_models.models:
            # print 'Loading... ', model
            car_model = '%s/%s%s.off' % (off_path, model.name, postfix)
            off_mesh = Mesh.from_off(car_model)
            self.car_models[model.name] = {'vertices': off_mesh.vertices * 0.01, 'faces': off_mesh.faces+1} # Faces are one-based numbering for this renderer!

    def render_car(self, pose, car_name):
        """Render a car instance given pose and car_name
        """
        car = self.car_models[car_name]
        scale = np.ones((3, ))
        pose = np.array(pose)
        # print np.max(car['vertices'], axis=0), np.min(car['vertices'], axis=0)
        vert = uts.project(pose, scale, car['vertices']) # [*, 3]
        K = self.intrinsic
        intrinsic = np.float64([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        depth, mask = render.renderMesh_py(np.float64(vert),
                                           np.float64(car['faces']),
                                           intrinsic,
                                           self.image_size[0],
                                           self.image_size[1],
                                           np.float64(self.linewidth))
        return depth, mask, vert, K

    def compute_reproj_sim(self, car_names, out_file=None):
        """Compute the similarity matrix between every pair of cars.
        """
        if out_file is None:
            out_file = './sim_mat.txt'

        sim_mat = np.eye(len(self.car_model))
        for i in range(len(car_names)):
            for j in range(i, len(car_names)):
                name1 = car_names[i][0]
                name2 = car_names[j][0]
                ind_i = self.car_model.keys().index(name1)
                ind_j = self.car_model.keys().index(name2)
                sim_mat[ind_i, ind_j] = self.compute_reproj(name1, name2)
                sim_mat[ind_j, ind_i] = sim_mat[ind_i, ind_j]

        np.savetxt(out_file, sim_mat, fmt='%1.6f')

    def compute_reproj(self, car_name1, car_name2):
        """Compute reprojection error between two cars
        """
        sims = np.zeros(10)
        for i, rot in enumerate(np.linspace(0, np.pi, num=10)):
            pose = np.array([0, rot, 0, 0, 0,5.5])
            depth1, mask1 = self.render_car(pose, car_name1)
            depth2, mask2 = self.render_car(pose, car_name2)
            sims[i] = eval_uts.IOU(mask1, mask2)

        return np.mean(sims)

    def merge_inst(self,
                   depth_in,
                   inst_id,
                   shape_id,
                   total_mask,
                   total_shape_id_map,
                   total_depth,
                   total_pose_mask,
                   pose_list): # depth, i + 1, car_pose['car_id'] + 1, self.mask, self.shape_id_map, self.depth, self.pose_map, car_pose['pose'])
        """Merge the prediction of each car instance to a full image
        """

        render_depth = depth_in.copy()
        render_depth[render_depth <= 0] = np.inf
        depth_arr = np.concatenate([render_depth[None, :, :],
                                    total_depth[None, :, :]], axis=0)
        idx = np.argmin(depth_arr, axis=0)
        total_depth = np.amin(depth_arr, axis=0)
        total_mask[idx == 0] = inst_id
        total_shape_id_map[idx == 0] = shape_id
        for pose_dim_idx, pose_one_dim in enumerate(pose_list):
            total_pose_mask[idx == 0, pose_dim_idx] = pose_one_dim

        return total_mask, total_shape_id_map, total_depth, total_pose_mask # self.mask, self.shape_id_map, self.depth, self.pose_map

    def rescale(self, image, intrinsic):
        """resize the image and intrinsic given a relative scale
        """

        intrinsic_out = uts.intrinsic_vec_to_mat(intrinsic,
                                                 self.image_size)
        # if self.scale != 1.0:
        hs, ws = self.image_size
        image_out = cv2.resize(image.copy(), (ws, hs))
        # else:
        #     image_out = image
        return image_out, intrinsic_out

    def showId(self, image_name):
        car_pose_file = '%s/%s.json' % (
            self._data_config['pose_dir'], image_name)

        with open(car_pose_file) as f:
            car_poses = json.load(f)

        self.pose_list = []
        self.shape_id_list = []

        for i, car_pose in enumerate(car_poses):
            self.pose_list.append(car_pose['pose'])
            self.shape_id_list.append(car_pose['car_id'])
        assert len(self.shape_id_list) == len(self.pose_list), 'Shape and pose lists lengths should match!'

        return self.pose_list, self.shape_id_list

    def showAnn_test(self, image_name, if_result=False, if_visualize=False, if_save=False, plot_path='tmp', is_training=False):
        """Show the annotation of a pose file in an image
        Input:
            image_name: the name of image
        Output:
            depth: a rendered depth map of each car
            masks: an instance mask of the label
            image_vis: an image show the overlap of car model and image
        """

        image_file = '%s/%s.jpg' % (self._data_config['image_dir'], image_name)
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        # print 'Original and rescaled image size: ', image.shape, self.image_size
        intrinsic = self.dataset.get_intrinsic(image_name, 'Camera_5')
        image_rescaled, self.intrinsic = self.rescale(image, intrinsic)

        if is_training:
            car_pose_file = '%s/%s.json' % (
                self._data_config['pose_dir'] if not(if_result) else self._data_config['pose_dir_result'], image_name)

            with open(car_pose_file) as f:
                car_poses = json.load(f)

            self.depth = self.MAX_DEPTH * np.ones(self.image_size)
            self.mask = np.zeros(self.depth.shape)
            self.shape_id_map = np.zeros(self.depth.shape)
            self.pose_map = np.zeros((self.depth.shape[0], self.depth.shape[1], 6)) + np.inf
            self.shape_map = np.zeros((self.depth.shape[0], self.depth.shape[1], 10)) + np.inf

            self.pose_list = []
            self.rot_uvd_list = []
            self.bbox_list = []
            self.shape_id_list = []

            plt.figure(figsize=(20, 10))
            plt.imshow(image_rescaled)
            for i, car_pose in enumerate(car_poses):
                car_name = car_models.car_id2name[car_pose['car_id']].name
                # if if_result:
                #     car_pose['pose'][-1]  = 1./car_pose['pose'][-1]
                depth, mask, vert, K = self.render_car(car_pose['pose'], car_name)
                self.mask, self.shape_id_map, self.depth, self.pose_map = self.merge_inst(
                    depth, i + 1, car_pose['car_id'] + 1, self.mask, self.shape_id_map, self.depth, self.pose_map, car_pose['pose'])
                self.pose_list.append(car_pose['pose'])
                self.shape_id_list.append(car_pose['car_id'])

                scale = np.ones((3, ))
                car = self.car_models[car_name]
                pose = np.array(car_pose['pose'])
                print 'GT pose: ', pose[3:]
                vert = car['vertices']
                vert = np.zeros((1, 3))
                vert_transformed = uts.project(pose, scale, vert) # [*, 3]
                print 'Center transformed: ', vert_transformed

                vert_hom = np.hstack((vert_transformed, np.ones((vert.shape[0], 1))))
                K_hom = np.hstack((K, np.zeros((3, 1))))
                proj_uv_hom = np.matmul(K_hom, vert_hom.T)
                proj_uv = np.vstack((proj_uv_hom[0, :]/proj_uv_hom[2, :], proj_uv_hom[1, :]/proj_uv_hom[2, :]))
                u = proj_uv[0:1, :] # [1, 1]
                v = proj_uv[1:2, :]
                d = proj_uv_hom[2:3, :]

                rot_uvd = [car_pose['pose'][0], car_pose['pose'][1], car_pose['pose'][2], u[0, 0], v[0, 0], car_pose['pose'][5]]
                self.rot_uvd_list.append(rot_uvd)

                plt.scatter(u, v, linewidths=20)

                F1 = K_hom[0, 0]
                W = K_hom[0, 2]
                F2 = K_hom[1, 1]
                H = K_hom[1, 2]
                K_T = np.array([[1./F1, 0., -W/F1], [0, 1./F2, -H/F2], [0., 0., 1.]])
                # print K_T
                # print self.intrinsic
                # print F1, W, F2, H
                uvd = np.vstack((u*d, v*d, d))
                xyz = np.matmul(K_T, uvd)
                print 'xyz / pose recovered: ', xyz
                # print 'uvd:', rot_uvd

                # print car_pose['pose'].shape, vert_transformed.shape

                ## Get bbox from mask
                arr = np.expand_dims(np.int32(mask), -1)
                # number of highest label:
                labmax = 1
                # maximum and minimum positions along each axis (initialized to very low and high values)
                b_first = np.iinfo('int32').max * np.ones((3, labmax + 1), dtype='int32')
                b_last = np.iinfo('int32').max * np.ones((3, labmax + 1), dtype='int32')
                # run through all dimensions making 2D slices and marking all existing labels to b
                for dim in range(2):
                    # create a generic slice object to make the slices
                    sl = [slice(None), slice(None), slice(None)]
                    bf = b_first[dim]
                    bl = b_last[dim]
                    # go through all slices in this dimension
                    for k in range(arr.shape[dim]):
                        # create the slice object
                        sl[dim] = k
                        # update the last "seen" vector
                        bl[arr[sl].flatten()] = k
                        # if we have smaller values in "last" than in "first", update
                        bf[:] = np.clip(bf, None, bl)
                bbox = [b_first[1, 1], b_last[1, 1], b_first[0, 1], b_last[0, 1]] # [x_min, x_max, y_min, y_max]
                self.bbox_list.append(bbox)
                plt.imshow(mask)
                print mask.shape
                currentAxis = plt.gca()
                # print (bbox[0], bbox[2]), bbox[1]-bbox[0], bbox[3]-bbox[2]
                currentAxis.add_patch(Rectangle((bbox[0], bbox[2]), bbox[1]-bbox[0], bbox[3]-bbox[2], alpha=1, edgecolor='r', facecolor='none'))
                # plt.show()
                # break
            plt.show()

            self.depth[self.depth == self.MAX_DEPTH] = -1.0
            image = 0.5 * image_rescaled
            for i in range(len(car_poses)):
                frame = np.float32(self.mask == i + 1)
                frame = np.tile(frame[:, :, None], (1, 1, 3))
                image = image + frame * 0.5 * self.colors[i, :]

            if if_visualize:
                uts.plot_images({'image_vis': np.uint8(image),
                    'shape_id': self.shape_id_map, 'mask': self.mask, 'depth': self.depth}, np.asarray(self.rot_uvd_list), self.bbox_list, layout=[1, 4], fig_size=10, save_fig=if_save, fig_name=plot_path)

            return image, self.mask, self.shape_id_map, self.depth, self.pose_map, image_rescaled, self.pose_list, self.shape_id_list, self.rot_uvd_list, self.bbox_list
        else:
            return None, None, None, None, None, image_rescaled, None, None, None, None

    def showAnn(self, image_name, if_result=False, if_visualize=False, if_save=False, plot_path='tmp', is_training=False):
        """Show the annotation of a pose file in an image
        Input:
            image_name: the name of image
        Output:
            depth: a rendered depth map of each car
            masks: an instance mask of the label
            image_vis: an image show the overlap of car model and image
        """

        image_file = '%s/%s.jpg' % (self._data_config['image_dir'], image_name)
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)[:, :, ::-1]
        # print 'Original and rescaled image size: ', image.shape, self.image_size
        intrinsic = self.dataset.get_intrinsic(image_name, 'Camera_5')
        image_rescaled, self.intrinsic = self.rescale(image, intrinsic)

        if is_training:
            car_pose_file = '%s/%s.json' % (
                self._data_config['pose_dir'] if not(if_result) else self._data_config['pose_dir_result'], image_name)

            with open(car_pose_file) as f:
                car_poses = json.load(f)

            self.depth = self.MAX_DEPTH * np.ones(self.image_size)
            self.mask = np.zeros(self.depth.shape)
            self.shape_id_map = np.zeros(self.depth.shape)
            self.pose_map = np.zeros((self.depth.shape[0], self.depth.shape[1], 6)) + np.inf
            self.shape_map = np.zeros((self.depth.shape[0], self.depth.shape[1], 10)) + np.inf

            self.pose_list = []
            self.rot_uvd_list = []
            self.bbox_list = []
            self.shape_id_list = []

            plt.figure(figsize=(20, 10))
            plt.imshow(image_rescaled)
            for i, car_pose in enumerate(car_poses):
                car_name = car_models.car_id2name[car_pose['car_id']].name
                # if if_result:
                #     car_pose['pose'][-1]  = 1./car_pose['pose'][-1]
                depth, mask, vert, K = self.render_car(car_pose['pose'], car_name)
                self.mask, self.shape_id_map, self.depth, self.pose_map = self.merge_inst(
                    depth, i + 1, car_pose['car_id'] + 1, self.mask, self.shape_id_map, self.depth, self.pose_map, car_pose['pose'])
                self.pose_list.append(car_pose['pose'])
                self.shape_id_list.append(car_pose['car_id'])

                scale = np.ones((3, ))
                car = self.car_models[car_name]
                pose = np.array(car_pose['pose'])
                print 'GT pose: ', pose[3:]
                vert = car['vertices']
                vert = np.zeros((1, 3))
                vert_transformed = uts.project(pose, scale, vert) # [*, 3]
                print 'Center transformed: ', vert_transformed

                vert_hom = np.hstack((vert_transformed, np.ones((vert.shape[0], 1))))
                K_hom = np.hstack((K, np.zeros((3, 1))))
                proj_uv_hom = np.matmul(K_hom, vert_hom.T)
                proj_uv = np.vstack((proj_uv_hom[0, :]/proj_uv_hom[2, :], proj_uv_hom[1, :]/proj_uv_hom[2, :]))
                u = proj_uv[0:1, :] # [1, 1]
                v = proj_uv[1:2, :]
                d = proj_uv_hom[2:3, :]

                rot_uvd = [car_pose['pose'][0], car_pose['pose'][1], car_pose['pose'][2], u[0, 0], v[0, 0], car_pose['pose'][5]]
                self.rot_uvd_list.append(rot_uvd)

                plt.scatter(u, v, linewidths=20)

                F1 = K_hom[0, 0]
                W = K_hom[0, 2]
                F2 = K_hom[1, 1]
                H = K_hom[1, 2]
                K_T = np.array([[1./F1, 0., -W/F1], [0, 1./F2, -H/F2], [0., 0., 1.]])
                # print K_T
                # print self.intrinsic
                # print F1, W, F2, H
                uvd = np.vstack((u*d, v*d, d))
                xyz = np.matmul(K_T, uvd)
                print 'xyz / pose recovered: ', xyz
                # print 'uvd:', rot_uvd

                # print car_pose['pose'].shape, vert_transformed.shape

                ## Get bbox from mask
                arr = np.expand_dims(np.int32(mask), -1)
                # number of highest label:
                labmax = 1
                # maximum and minimum positions along each axis (initialized to very low and high values)
                b_first = np.iinfo('int32').max * np.ones((3, labmax + 1), dtype='int32')
                b_last = np.iinfo('int32').max * np.ones((3, labmax + 1), dtype='int32')
                # run through all dimensions making 2D slices and marking all existing labels to b
                for dim in range(2):
                    # create a generic slice object to make the slices
                    sl = [slice(None), slice(None), slice(None)]
                    bf = b_first[dim]
                    bl = b_last[dim]
                    # go through all slices in this dimension
                    for k in range(arr.shape[dim]):
                        # create the slice object
                        sl[dim] = k
                        # update the last "seen" vector
                        bl[arr[sl].flatten()] = k
                        # if we have smaller values in "last" than in "first", update
                        bf[:] = np.clip(bf, None, bl)
                bbox = [b_first[1, 1], b_last[1, 1], b_first[0, 1], b_last[0, 1]] # [x_min, x_max, y_min, y_max]
                self.bbox_list.append(bbox)
                plt.imshow(mask)
                print mask.shape
                currentAxis = plt.gca()
                # print (bbox[0], bbox[2]), bbox[1]-bbox[0], bbox[3]-bbox[2]
                currentAxis.add_patch(Rectangle((bbox[0], bbox[2]), bbox[1]-bbox[0], bbox[3]-bbox[2], alpha=1, edgecolor='r', facecolor='none'))
                # plt.show()
                # break
            plt.show()

            self.depth[self.depth == self.MAX_DEPTH] = -1.0
            image = 0.5 * image_rescaled
            for i in range(len(car_poses)):
                frame = np.float32(self.mask == i + 1)
                frame = np.tile(frame[:, :, None], (1, 1, 3))
                image = image + frame * 0.5 * self.colors[i, :]

            if if_visualize:
                uts.plot_images({'image_vis': np.uint8(image),
                    'shape_id': self.shape_id_map, 'mask': self.mask, 'depth': self.depth}, np.asarray(self.rot_uvd_list), self.bbox_list, layout=[1, 4], fig_size=10, save_fig=if_save, fig_name=plot_path)

            return image, self.mask, self.shape_id_map, self.depth, self.pose_map, image_rescaled, self.pose_list, self.shape_id_list, self.rot_uvd_list, self.bbox_list
        else:
            return None, None, None, None, None, image_rescaled, None, None, None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation self localization.')
    parser.add_argument('--image_name', default='180116_053947113_Camera_5',
                        help='the dir of ground truth')
    parser.add_argument('--data_dir', default='../apolloscape/3d_car_instance_sample/',
                        help='the dir of ground truth')
    args = parser.parse_args()
    assert args.image_name
    visualizer = CarPoseVisualizer(args, scale=0.1)
    visualizer.load_car_models()
    image_vis, mask, depth = visualizer.showAnn(args.image_name)
    print 'image_vis.shape, mask.shape, depth.shape: ', image_vis.shape, mask.shape, depth.shape

    import scipy.misc
    scipy.misc.imsave('test_image_vis.jpg', image_vis)
    scipy.misc.imsave('test_mask.jpg', mask*255)
    scipy.misc.imsave('test_depth.jpg', depth)

    print "Test images saved at test_{image_vis, mask, depth}.jpg --render_car_instances"
