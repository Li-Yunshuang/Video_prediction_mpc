import os
import numpy as np
import tensorflow as tf
import imp
import sys
import cPickle
import pdb

import imp
import matplotlib.pyplot as plt
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from makegifs import comp_gif


from datetime import datetime

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 1

# How often to run a batch through the validation model.
VAL_INTERVAL = 10#200

# How often to save a model checkpointre
SAVE_INTERVAL = 20
from PIL import Image
from prediction_model_sawyer_action import Prediction_Model

FLAGS = flags.FLAGS
flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')
flags.DEFINE_string('visualize', '', 'model within hyperparameter folder from which to create gifs')
flags.DEFINE_integer('device', 0 ,'the value for CUDA_VISIBLE_DEVICES variable')
flags.DEFINE_string('pretrained', None, 'path to model file from which to resume training')
flags.DEFINE_bool('diffmotions', False, 'visualize several different motions for a single scene')

def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))

# def gen_actions(gt_images, gen_images):
#     videolist = []
#     videolist.append(gen_images)

#     vid_length = min([len(vid) for vid in videolist])
#     print 'smallest length of all videos', vid_length
#     # cut to the smallest video lenth
#     for i in range(len(video_batch)):
#         video_batch[i] = [np.expand_dims(videoframe, axis=0) for videoframe in video_batch[i]]  #grete a gif

#     for i in range(len(video_batch)):
#         video_batch[i] = np.concatenate(video_batch[i], axis= 0)

#     #videobatch is a list of [timelength, batchsize, 64, 64, 3]    14,32,64,64,3

#     fullframe_list = []
#     for t in range(vid_length):   #vid_lenth = 14
#         column_list = []
#         for exp in range(num_exp):
#             #column_images = [video[t,exp] for video in video_batch]   # 12 , 64, 64, 3  one sequence
#             for video in video_batch:
#                 column_images = [video[t,exp]]
#                 column_images = np.concatenate(column_images, axis=0)  #make column
#                 column_list.append(column_images)
#                 break

#         full_frame = np.concatenate(column_list, axis= 1)
#         if convert_from_float:
#             full_frame = np.uint8(255 * full_frame)
#         fullframe_list.append(full_frame)

#     return fullframe_list

def cal_best_actions(action, cost, num_exp = 1):

    rl = [5, 3, 5, 3.006981, 5, 3.4825,5,0.05,0.05]
    ll = [-2.5,-1.5,-2.5,-3.0,-2.5,0.0175,-2.5,0,0]
    #ul = [2.5,1.5,2.5,0.06981,2.5,3.5,2.5]

    actions = []
    sp = np.shape(action)
    print sp
    for i in range(sp[0]):
        choices = cost[i]
        minindex  = np.argmax(choices)
        print minindex
        
        act_choices = action[i]
        act_choices = act_choices[minindex]
        act = act_choices[0]
        
        # print act
        # for i in range(9):
        #     act[i] = ll[i] + rl[i]*act[i]
        # print act

        actions.append(act)

        file_path = './results/trial1.npy'
        np.save(file_path, actions)

    return actions




class Model(object):
    def __init__(self,
                 conf,
                 images=None,
                 actions=None,
                 states=None,
                 reuse_scope=None,
                 pix_distrib=None,
                 pix_distrib2=None,
                 inference = False):

        self.conf = conf
        #print images  #(batch_size use_len 64 64 3)  
        self.images_sel = images
        self.actions_sel = actions
        self.states_sel = states

        self.iter_num = tf.placeholder(tf.float32, [])
        summaries = []

        # Split into timesteps.
        if actions != None:
            actions = tf.split(axis=1, num_or_size_splits=actions.get_shape()[1], value=actions)
            actions = [tf.squeeze(act) for act in actions]

        if states != None:
            states = tf.split(axis=1, num_or_size_splits=states.get_shape()[1], value=states)
            states = [tf.squeeze(st) for st in states]
        images = tf.split(axis=1, num_or_size_splits=images.get_shape()[1], value=images)
        #print images is 12 (32,1,64,64,3)
        images = [tf.squeeze(img) for img in images]
        #print images is 12 (32,64,64,3))
        if pix_distrib != None:
            pix_distrib = tf.split(axis=1, num_or_size_splits=pix_distrib.get_shape()[1], value=pix_distrib)
            pix_distrib = [tf.squeeze(pix) for pix in pix_distrib]

        if pix_distrib2 != None:
            pix_distrib2 = tf.split(axis=1, num_or_size_splits=pix_distrib2.get_shape()[1], value=pix_distrib2)
            pix_distrib2= [tf.squeeze(pix) for pix in pix_distrib2]
        if reuse_scope is None:
            self.m = Prediction_Model(
                images,
                actions,
                states,
                iter_num=self.iter_num,
                pix_distributions1=pix_distrib,
                pix_distributions2=pix_distrib2,
                conf=conf)
            self.m.build()
        else:  # If it's a validation or test model.
            with tf.variable_scope(reuse_scope, reuse=tf.AUTO_REUSE):
                self.m = Prediction_Model(
                    images,
                    actions,
                    states,
                    iter_num=self.iter_num,
                    pix_distributions1=pix_distrib,
                    pix_distributions2=pix_distrib2,
                    conf= conf)
                self.m.build()

        self.lr = tf.placeholder_with_default(conf['learning_rate'], ())

        if not inference:
            # L2 loss, PSNR for eval.
            true_fft_list, pred_fft_list = [], []
            loss, psnr_all = 0.0, 0.0

            self.fft_weights = tf.placeholder(tf.float32, [64, 64])

            for i, x, gx in zip(
                    range(len(self.m.gen_images)), images[conf['context_frames']:],
                    self.m.gen_images[conf['context_frames'] - 1:]):
                recon_cost_mse = mean_squared_error(x, gx)  # MSE OF origin & generate 
                # psnr_i = peak_signal_to_noise_ratio(x, gx)
                # psnr_all += psnr_i
                recon_cost = recon_cost_mse

                loss += recon_cost

            if ('ignore_state_action' not in conf) and ('ignore_state' not in conf):
                for i, state, gen_state in zip(
                        range(len(self.m.gen_states)), states[conf['context_frames']:],
                        self.m.gen_states[conf['context_frames'] - 1:]):
                    state_cost = mean_squared_error(state, gen_state) * 1e-4 * conf['use_state']
                    loss += state_cost

            self.loss = loss = loss / np.float32(len(images) - conf['context_frames'])


def main(unused_argv, conf_script= None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.device)
    print 'using CUDA_VISIBLE_DEVICES=', FLAGS.device
    from tensorflow.python.client import device_lib
    print device_lib.list_local_devices()

    if conf_script == None: conf_file = FLAGS.hyper
    else: conf_file = conf_script

    if not os.path.exists(FLAGS.hyper):
        sys.exit("Experiment configuration not found")
    hyperparams = imp.load_source('hyperparams', conf_file)

    conf = hyperparams.configuration

    inference = False
    if FLAGS.visualize:
        print 'creating visualizations ...'
        conf['schedsamp_k'] = -1  # don't feed ground truth
        conf['data_dir'] = '/'.join(str.split(conf['data_dir'], '/')[:-1] + ['test'])
        conf['visualize'] = conf['output_dir'] + '/' + FLAGS.visualize
        conf['event_log_dir'] = '/tmp'
        conf.pop('use_len', None)
        conf['batch_size'] = 2

        conf['sequence_length'] = 45
        if FLAGS.diffmotions:
            inference = True
            conf['sequence_length'] = 45

    if 'sawyer' in conf:
        from read_tf_record_sawyer12 import build_tfrecord_input
    else:
        from read_tf_record import build_tfrecord_input

    # print 'Constructing for val_model'
    with tf.variable_scope('model', reuse=None) as training_scope:
        images_aux1, actions, states = build_tfrecord_input(conf, training=True)
        images = images_aux1
    #     print images
    #     model = Model(conf, images, actions, states, inference=inference)
    with tf.variable_scope('val_model', reuse=True):
        val_images_aux1, val_actions, val_states = build_tfrecord_input(conf, training=False)
        val_images = val_images_aux1
        print val_images
        val_model = Model(conf, val_images, val_actions, val_states,
                               training_scope, inference=inference)
        #val_model = Model(conf, val_images, val_actions, val_states,
        #                      inference=inference)
    print 'Constructing saver.'
    # Make saver.

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # remove all states from group of variables which shall be saved and restored:
    vars_no_state = filter_vars(vars)
    saver = tf.train.Saver(vars_no_state, max_to_keep=0)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.
    sess = tf.InteractiveSession(config= tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.summary.FileWriter(conf['output_dir'], graph=sess.graph, flush_secs=10)

    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())

    if conf['visualize']:
        print '-------------------------------------------------------------------'
        print 'verify current settings!! '
        for key in conf.keys():
            print key, ': ', conf[key]
        print '-------------------------------------------------------------------'
        saver.restore(sess, conf['visualize'])
        feed_dict = {val_model.lr: 0.0,
                     val_model.iter_num: 0 }

        file_path = conf['output_dir']
        
        ground_truth, gen_images, gen_masks, action, cost= sess.run([val_images,
                                                              val_model.m.gen_images,
                                                              val_model.m.gen_masks,
                                                              val_model.m.ranaction_all,
                                                              val_model.m.cost_all],
                                                             feed_dict)
        best_actions = cal_best_actions(action, cost)
        print best_actions


        dict = {}
        dict['gen_images'] = gen_images
        dict['ground_truth'] = ground_truth
        dict['gen_masks'] = gen_masks
        cPickle.dump(dict, open(file_path + '/pred.pkl', 'wb'))
        print 'written files to:' + file_path

            #comp_gif(conf, conf['output_dir'], append_masks=True, show_parts=True)
        comp_gif(conf, conf['output_dir'], append_masks=True)

        return

def create_one_hot(conf, desig_pix):
    one_hot = np.zeros((1, 1, 64, 64, 1), dtype=np.float32)
    # switch on pixels
    one_hot[0, 0, desig_pix[0], desig_pix[1]] = 1.
    one_hot = np.repeat(one_hot, conf['context_frames'], axis=1)
    app_zeros = np.zeros((1, conf['sequence_length']- conf['context_frames'], 64, 64, 1), dtype=np.float32)
    one_hot = np.concatenate([one_hot, app_zeros], axis=1)
    one_hot = np.repeat(one_hot, conf['batch_size'], axis=0)

    return one_hot


def filter_vars(vars):
    newlist = []
    for v in vars:
        if not '/state:' in v.name:
            newlist.append(v)
        else:
            print 'removed state variable from saving-list: ', v.name

    return newlist

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()

