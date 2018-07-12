"""
    Brief: Test renderer
    Author: wangpeng54@baidu.com
    Date: 2018/6/10
"""

import sys
import numpy as np
np.set_printoptions(threshold=np.nan)
import render_egl
import pickle as pkl
import utils.utils as uts

sys.path.insert(0, '../../')


def test_render():
    car_model = 'test.pkl'
    with open(car_model, 'r') as f:
        model = pkl.load(f)

    vertices = model['vertices']
    faces = model['faces']
    scale = np.float32([1, 1, 1])

    intrinsic = np.float64([250, 250, 160, 120])
    imgsize = [240, 320]

    intrinsic = np.float64([350, 350, 320, 92])
    imgsize = [180, 624]

    T = np.float32([0.0, 1.9, 0.0, -1.0, -1.1, 6.0])

    vertices_r = uts.project(T, scale, vertices)
    vertices_r[:, 1] = vertices_r[:, 1] - 0.3
    vertices_r[:, 2] = vertices_r[:, 2] + 3.0

    faces = np.float64(faces)
    depth, mask = render_egl.renderMesh_py(
        vertices_r, faces, intrinsic, imgsize[0], imgsize[1], 0.0)
    assert np.max(depth) > 0
    print 'passed.'
    print 'Shape of depth:', depth.shape, 'Shape of mask:', mask.shape
    import scipy.misc
    scipy.misc.imsave('depth_test.jpg', depth)
    scipy.misc.imsave('mask_test.jpg', mask*255)
    print 'Depth and mask image are saved to depth_test.jpg and mask_test.jpg.'
if __name__ == '__main__':
    test_render()
