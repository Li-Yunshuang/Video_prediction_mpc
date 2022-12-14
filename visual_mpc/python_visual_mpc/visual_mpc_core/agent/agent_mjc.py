""" This file defines an agent for the MuJoCo simulator environment. """
from copy import deepcopy
import copy
import numpy as np
import mujoco_py
from mujoco_py.mjtypes import *
import pdb
import cPickle
from PIL import Image
import matplotlib.pyplot as plt



import time
from python_visual_mpc.visual_mpc_core.infrastructure.trajectory import Trajectory
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
from mpl_toolkits.mplot3d import Axes3D
import xml.etree.cElementTree as ET
import xml.dom.minidom as minidom
import pdb

class AgentMuJoCo(object):
    """
    All communication between the algorithms and MuJoCo is done through
    this class.
    """
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self.T = self._hyperparams['T']

        self._setup_world()

    def _setup_world(self):
        """
        Helper method for handling setup of the MuJoCo world.
        Args:
            filename: Path to XML file containing the world information.
        """
        if "varying_mass" in self._hyperparams:
            self.create_xml()

        #######pdb.set_trace()

        self._model= mujoco_py.MjModel(self._hyperparams['filename'])
        self.model_nomarkers = mujoco_py.MjModel(self._hyperparams['filename_nomarkers'])

        gofast = True
        self._small_viewer = mujoco_py.MjViewer(visible=True,
                                                init_width=self._hyperparams['image_width'],
                                                init_height=self._hyperparams['image_height'],
                                                go_fast=gofast)
        self._small_viewer.start()
        self._small_viewer.cam.camid = 0
        if self._hyperparams['additional_viewer']:
            self._large_viewer = mujoco_py.MjViewer(visible=True, init_width=480,
                                                    init_height=480, go_fast=gofast)
            self._large_viewer.start()
            self._large_viewer.cam.camid = 0


    def create_xml(self):

        for i in range(self._hyperparams['num_objects']):
            xmldir = '/'.join(str.split(self._hyperparams['filename'], '/')[:-1])
            mass = np.random.uniform(.01, 1.)
            root = ET.Element("top")
            ET.SubElement(root, "inertial", pos="0 0 0", mass="{}".format(mass),
                          diaginertia="{0} {1} {2}".format(mass/2., mass/2., mass/2.))
            tree = ET.ElementTree(root)
            xml_str = minidom.parseString(ET.tostring(
                    tree.getroot(),
                    'utf-8')).toprettyxml(indent="    ")

            xml_str = xml_str.splitlines()[1:]
            xml_str = "\n".join(xml_str)

            with open(xmldir+"/mass{}.xml".format(i), "wb") as f:
                f.write(xml_str)

    def sample(self, policy, verbose=True, save=True, noisy=False):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.
        """
        if "varying_mass" in self._hyperparams:
            self._setup_world()

        traj_ok = False
        i_trial = 0
        imax = 100
        while not traj_ok and i_trial < imax:
            i_trial += 1
            traj_ok, traj = self.rollout(policy)
        print 'needed {} trials'.format(i_trial)

        tfinal = self._hyperparams['T'] -1
        if not self._hyperparams['data_collection']:
            if 'use_goalimage' in self._hyperparams:
                self.final_poscost, self.final_anglecost = self.eval_action(traj, tfinal, getanglecost=True)
            else:
                self.final_poscost = self.eval_action(traj, tfinal)

        if 'save_goal_image' in self._hyperparams:
            self.save_goal_image_conf(traj)

        if not 'novideo' in self._hyperparams:
            self.save_gif()

        if "varying_mass" in self._hyperparams:
            self._small_viewer.finish()
            self._large_viewer.finish()

        policy.finish()
        return traj


    def get_max_move_pose(self, traj):

        delta_move = np.zeros(self._hyperparams['num_objects'])
        for i in range(self._hyperparams['num_objects']):
            for t in range(self.T-1):
                delta_move[i] += np.linalg.norm(traj.Object_pose[t+1,i,:2] -traj.Object_pose[t,i,:2])

        imax = np.argmax(delta_move)
        traj.max_move_pose = traj.Object_pose[:,imax]

        return traj

    def rollout(self, policy):
        # Create new sample, populate first time step.
        self._init()
        traj = Trajectory(self._hyperparams)

        self._small_viewer.set_model(self.model_nomarkers)
        if self._hyperparams['additional_viewer']:
            self._large_viewer.set_model(self._model)
        self._small_viewer.cam.camid = 0
        self._large_viewer.cam.camid = 0

        # apply action of zero for the first few steps, to let the scene settle
        for t in range(self._hyperparams['skip_first']):
            for _ in range(self._hyperparams['substeps']):
                self._model.data.ctrl = np.array([0., 0.])
                self._model.step()

        self.large_images_traj = []
        self.large_images = []

        # Take the sample.
        for t in range(self.T):
            traj.X_full[t, :] = self._model.data.qpos[:2].squeeze()
            traj.Xdot_full[t, :] = self._model.data.qvel[:2].squeeze()
            traj.X_Xdot_full[t, :] = np.concatenate([traj.X_full[t, :], traj.Xdot_full[t, :]])
            for i in range(self._hyperparams['num_objects']):
                fullpose = self._model.data.qpos[i * 7 + 2:i * 7 + 9].squeeze()
                zangle = self.quat_to_zangle(fullpose[3:])
                traj.Object_pose[t, i, :] = np.concatenate([fullpose[:2], zangle])

            if not self._hyperparams['data_collection']:
                traj.score[t] = self.eval_action(traj, t)

            self._store_image(t, traj, policy)

            if self._hyperparams['data_collection'] or 'random_baseline' in self._hyperparams:
                mj_U, target_inc = policy.act(traj, t)
            else:
                mj_U, pos, ind, targets = policy.act(traj, t, init_model=self._model)

                traj.desig_pos[t, :] = self._model.data.site_xpos[0, :2]

                if self._hyperparams['add_traj']:  # whether to add visuals for trajectory
                    self.large_images_traj += self.add_traj_visual(self.large_images[t], pos, ind, targets)

            if 'poscontroller' in self._hyperparams.keys():
                traj.U[t, :] = target_inc
            else:
                traj.U[t, :] = mj_U

            accum_touch = np.zeros_like(self._model.data.sensordata)

            for _ in range(self._hyperparams['substeps']):
                accum_touch += self._model.data.sensordata

                if 'vellimit' in self._hyperparams:
                    # calculate constraint enforcing force..
                    c_force = self.enforce(self._model)
                    mj_U += c_force
                self._model.data.ctrl = mj_U
                self._model.step()  # simulate the model in mujoco

            if 'touch' in self._hyperparams:
                traj.touchdata[t, :] = accum_touch.squeeze()
                print 'accumulated force', t
                print accum_touch

        traj = self.get_max_move_pose(traj)

        # only save trajectories which displace objects above threshold
        if 'displacement_threshold' in self._hyperparams:
            assert self._hyperparams['data_collection']
            disp_per_object = np.zeros(self._hyperparams['num_objects'])
            for i in range(self._hyperparams['num_objects']):
                pos_old = traj.Object_pose[0, i, :2]
                pos_new = traj.Object_pose[t, i, :2]
                disp_per_object[i] = np.linalg.norm(pos_old - pos_new)

            if np.sum(disp_per_object) > self._hyperparams['displacement_threshold']:
                traj_ok = True
            else:
                traj_ok = False
        else:
            traj_ok = True

        return traj_ok, traj

    def save_goal_image_conf(self, traj):
        div = .05
        quantized = np.around(traj.score/div)
        best_score = np.min(quantized)
        for i in range(traj.score.shape[0]):
            if quantized[i] == best_score:
                first_best_index = i
                break

        print 'best_score', best_score
        print 'allscores', traj.score
        print 'goal index: ', first_best_index

        goalimage = traj._sample_images[first_best_index]
        goal_ballpos = np.concatenate([traj.X_full[first_best_index], np.zeros(2)])  #set velocity to zero

        goal_object_pose = traj.Object_pos[first_best_index]

        img = Image.fromarray(goalimage)

        dict = {}
        dict['goal_image'] = goalimage
        dict['goal_ballpos'] = goal_ballpos
        dict['goal_object_pose'] = goal_object_pose

        cPickle.dump(dict, open(self._hyperparams['save_goal_image'] + '.pkl', 'wb'))
        img.save(self._hyperparams['save_goal_image'] + '.png',)

    def eval_action(self, traj, t, getanglecost=False):
        if 'use_goalimage' not in self._hyperparams:
            goalpoint = np.array(self._hyperparams['goal_point'])
            refpoint = self._model.data.site_xpos[0,:2]
            return np.linalg.norm(goalpoint - refpoint)
        else:
            goalpos = self._hyperparams['goal_object_pose'][0][0:2]
            goal_quat= self._hyperparams['goal_object_pose'][0][3:]
            curr_pos = traj.Object_pos[t, 0, 0:2]

            goalangle = self.quat_to_zangle(goal_quat)
            currangle = traj.Object_pos[t, 0, 2]
            anglediff = self.calc_anglediff(goalangle, currangle)
            mult = 0.01 #0.1
            anglecost = np.abs(anglediff) / np.pi *180 * mult

            poscost = np.linalg.norm(goalpos - curr_pos)
            print 'angle diff cost :', anglecost
            print 'pos cost: ', poscost

            if getanglecost:
                return poscost, anglecost
            else: return poscost

    def zangle_to_quat(self, zangle):
        """
        :param zangle in rad
        :return: quaternion
        """
        return np.array([np.cos(zangle/2), 0, 0, np.sin(zangle/2) ])

    def quat_to_zangle(self, quat):
        """
        :param quat: quaternion with only
        :return: zangle in rad
        """
        theta = np.arctan2(2*quat[0]*quat[3], 1-2*quat[3]**2)
        return np.array([theta])

    def calc_anglediff(self, alpha, beta):
        delta = alpha - beta
        while delta > np.pi:
            delta -= 2*np.pi
        while delta < -np.pi:
            delta += 2*np.pi
        return delta

    def enforce(self, model):
        vel = model.data.qvel[:2].squeeze()
        des_vel = deepcopy(vel)
        vmax = self._hyperparams['vellimit']
        des_vel[des_vel> vmax] = vmax
        des_vel[des_vel<-vmax] = -vmax
        gain = 1000
        force = -(vel - des_vel) * gain
        # if np.any(force != 0):
            # print 'enforcing vel constraint', force
            # print 'vel ',vel
            # print 'des_vel ', des_vel
            # print 'velocity constraint violation', (vel - des_vel)
            # print 'correction force:', force
        return force


    def get_world_coord(self, proj_mat, depth_image, pix_pos):
        depth = depth_image[pix_pos[0], pix_pos[1]]
        pix_pos = pix_pos.astype(np.float32) / depth_image.shape[0]
        clipspace = pix_pos*2 -1
        depth = depth*2 -1

        clipspace = np.concatenate([clipspace, depth, np.array([1.]) ])

        res = np.linalg.inv(proj_mat).dot(clipspace)
        res[:3] = 1 - res[:3]

        return res[:3]

    def get_point_cloud(self, depth_image, proj_mat):

        height = depth_image.shape[0]
        point_cloud = np.zeros([height, height,3])
        for r in range(point_cloud.shape[0]):
            for c in range(point_cloud.shape[1]):
                pix_pos = np.array([r, c])
                point_cloud[r, c] = self.get_world_coord(proj_mat,depth_image, pix_pos)[:3]

        return point_cloud

    def plot_point_cloud(self, point_cloud):

        height = point_cloud.shape[0]

        point_cloud = point_cloud.reshape([height**2, 3])
        px = point_cloud[:, 0]
        py = point_cloud[:, 1]
        pz = point_cloud[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Axes3D.scatter(ax, px, py, pz)
        plt.show()


    def _store_image(self,t, traj, policy):
        """
        store image at time index t
        """
        if self._hyperparams['additional_viewer']:
            self._large_viewer.loop_once()

        img_string, width, height = self._large_viewer.get_image()
        largeimage = np.fromstring(img_string, dtype='uint8').reshape(
                (480, 480, self._hyperparams['image_channels']))[::-1, :, :]
        self.large_images.append(largeimage)

        # collect retina image
        if 'large_images_retina' in self._hyperparams:
            imheight = self._hyperparams['large_images_retina']
            traj.large_images_retina[t] = cv2.resize(largeimage, (imheight, imheight), interpolation=cv2.INTER_AREA)

            # save initial retina position:
            if t == 0:
                rh = self._hyperparams['retina_size']
                block_coord = traj.Object_pos[t, 0, :2]
                # angle = traj.Object_pos[t, 0, 2]
                # adjusted_blockcoord = block_coord+ .1 *np.array([np.cos(angle), np.sin(angle)])
                img_coord = self.mujoco_to_imagespace(block_coord, numpix=imheight)
                if img_coord[0] < rh/2:
                    img_coord[0] = rh/2
                if img_coord[1] < rh/2:
                    img_coord[1] = rh/2
                if img_coord[0] > imheight - rh/2 -1:
                    img_coord[0] = imheight - rh/2
                if img_coord[1] > imheight - rh/2 -1:
                    img_coord[1] = imheight - rh/2

                traj.initial_ret_pos = img_coord
                # ret = traj.large_images_retina[t][ img_coord[0]-rh/2:img_coord[0]+rh/2,
                #                                                            img_coord[1]-rh/2:img_coord[1] + rh/2]
                # Image.fromarray(ret).show()
                # pdb.set_trace()

        ######
        #small viewer:
        self.model_nomarkers.data.qpos = self._model.data.qpos
        self.model_nomarkers.data.qvel = self._model.data.qvel
        self.model_nomarkers.step()
        self._small_viewer.loop_once()

        img_string, width, height = self._small_viewer.get_image()
        img = np.fromstring(img_string, dtype='uint8').reshape((height, width, self._hyperparams['image_channels']))[::-1,:,:]

        traj._sample_images[t,:,:,:] = img

        if 'gen_point_cloud' in self._hyperparams:
            # getting depth values
            (img_string, width, height), proj_mat = self._small_viewer.get_depth()
            dimage = np.fromstring(img_string, dtype=np.float32).reshape(
                (width, height, 1))[::-1, :, :]
            Image.fromarray(np.squeeze(dimage*255.).astype(np.uint8)).show()
            pcl_dict = {}
            pcl = self.get_point_cloud(dimage, proj_mat)
            pcl_dict['pcl'] = pcl
            pcl_dict['image'] = img
            cPickle.dump(pcl_dict, open(self._hyperparams['pcldir']+'cloud.pkl' ,'wb'))
            # self.plot_point_cloud(pcl)
            #####pdb.set_trace()

        if 'store_video_prediction' in self._hyperparams:
            if t > 1:
                traj.final_predicted_images.append((policy.terminal_pred*255.).astype(np.uint8))

        if 'store_whole_pred' in self._hyperparams:
            if t > 1:
                traj.predicted_images = policy.best_gen_images
                traj.gtruth_images = policy.best_gtruth_images


    def add_traj_visual(self, img, traj, bestindices, targets):

        large_sample_images_traj = []
        fig = plt.figure(figsize=(6, 6), dpi=80)
        fig.add_subplot(111)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        num_samples = traj.shape[0]
        niter = traj.shape[1]

        for itr in range(niter):

            axes = plt.gca()
            plt.cla()
            axes.axis('off')
            plt.imshow(img, zorder=0)
            axes.autoscale(False)

            for smp in range(num_samples):  # for each trajectory

                x = traj[smp, itr, :, 1]
                y = traj[smp, itr, :, 0]

                if smp == bestindices[itr][0]:
                    plt.plot(x, y, zorder=1, marker='o', color='y')
                elif smp in bestindices[itr][1:]:
                    plt.plot(x, y, zorder=1, marker='o', color='r')
                else:
                    if smp % 5 == 0:
                        plt.plot(x, y, zorder=1, marker='o', color='b')

            fig.canvas.draw()  # draw the canvas, cache the renderer

            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            large_sample_images_traj.append(data)

        return large_sample_images_traj

    def save_gif(self):
        file_path = self._hyperparams['record']
        from python_visual_mpc.video_prediction.utils_vpred.create_gif_lib import npy_to_gif
        if 'random_baseline' in self._hyperparams:
            npy_to_gif(self.large_images, file_path)
        else:
            npy_to_gif(self.large_images_traj, file_path)

    def _init(self):
        """
        Set the world to a given model, and run kinematics.
        Args:
        """

        #create random starting poses for objects
        def create_pos():
            poses = []
            for i in range(self._hyperparams['num_objects']):
                pos = np.random.uniform(-.35, .35, 2)
                alpha = np.random.uniform(0, np.pi*2)
                ori = np.array([np.cos(alpha/2), 0, 0, np.sin(alpha/2) ])
                poses.append(np.concatenate((pos, np.array([0]), ori), axis= 0))
            return np.concatenate(poses)

        if self._hyperparams['x0'].shape[0] > 4: # if object pose explicit do not sample poses
            object_pos = self._hyperparams['x0'][4:]
        else:
            object_pos= create_pos()

        # Initialize world/run kinematics
        if 'randomize_ballinitpos' in self._hyperparams:
            x0 = np.random.uniform(-.35, .35, 2)
        else:
            x0 = self._hyperparams['x0']
        if 'goal_point' in self._hyperparams.keys():
            goal = np.append(self._hyperparams['goal_point'], [.1])   # goal point
            ref = np.append(object_pos[:2], [.1]) # reference point on the block
            self._model.data.qpos = np.concatenate((x0[:2], object_pos,goal, ref), 0)
        else:
            self._model.data.qpos = np.concatenate((x0[:2], object_pos), 0)

        self._model.data.qvel = np.zeros_like(self._model.data.qvel)

    def mujoco_to_imagespace(self, mujoco_coord, numpix=64, truncate=False):
        """
        convert form Mujoco-Coord to numpix x numpix image space:
        :param numpix: number of pixels of square image
        :param mujoco_coord:
        :return: pixel_coord
        """
        viewer_distance = .75  # distance from camera to the viewing plane
        window_height = 2 * np.tan(75 / 2 / 180. * np.pi) * viewer_distance  # window height in Mujoco coords
        pixelheight = window_height / numpix  # height of one pixel
        pixelwidth = pixelheight
        window_width = pixelwidth * numpix
        middle_pixel = numpix / 2
        pixel_coord = np.rint(np.array([-mujoco_coord[1], mujoco_coord[0]]) /
                              pixelwidth + np.array([middle_pixel, middle_pixel]))
        pixel_coord = pixel_coord.astype(int)

        return pixel_coord
