# This file is used to construct and visualize critical point sets 
# and upper-bound sets based on a trained model and a sample.
# This file only support batch_size = 1 and is modified base on file
# "evalutate.py"


import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import errno
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util
import math
import matplotlib.pyplot as plt

# changed by wind:
# set batch_size = 1
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls_basic', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_class', type=int, default=40, help='Number of Classes [default: 7]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--num_feature', type=int, default=256, help='Point Number [32-2048] [default: 128]')
parser.add_argument('--dump_dir', default='dump_visual', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()

NUM_CLASSES = FLAGS.num_class
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FEATURE = FLAGS.num_feature
MODEL_PATH = 'log/' + FLAGS.model+'_'+str(NUM_FEATURE)+'_'+'model.ckpt'
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points, global_feature,global_matrix = MODEL.get_model(
                pointclouds_pl, is_training_pl,
                num_feature = NUM_FEATURE, num_class = NUM_CLASSES)
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'global_matrix': global_matrix,
           'global_feature': global_feature}
    
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
    
    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    for fn in range(len(TEST_FILES)):
        log_string('----'+str(fn)+'----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        print(file_size)

        # set by wind:
        # my code is based on batch_size = 1
        # set batch_size = 1 for this file
        for batch_idx in range(file_size):
#        for batch_idx in np.array([1,710,1805,385,703,1501])-1:
            print(batch_idx)
            start_idx = batch_idx
            end_idx = batch_idx + 1 
            cur_batch_size = 1

            no_influence_position = current_data[start_idx,0,:].copy()

            global_feature_list = []
            orgin_data = current_data[start_idx,:,:].copy()

            #-------------------------------------------------------------------
            # save origin data
            #-------------------------------------------------------------------
            fileName = 'dataAnalysis/figures/%d_orgin_points' % (start_idx)
            img_filename = fileName + '.jpg'
#            plyFileName = fileName + '.ply'
#            pc_util.write_ply(np.squeeze(orgin_data),plyFileName)
#             pc_util.pyplot_draw_point_cloud(np.squeeze( orgin_data ),'')
            output_img = pc_util.draw_point_cloud(np.squeeze(orgin_data))
            scipy.misc.imsave(img_filename, output_img)
            
            #-------------------------------------------------------------------
            # get global matrix
            #-------------------------------------------------------------------
            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :] ,
                                ops['labels_pl']: current_label[start_idx:end_idx],
                                ops['is_training_pl']: is_training}

            global_matrix = np.array(sess.run(ops['global_matrix'],
                                    feed_dict=feed_dict))
            
            global_matrix = global_matrix[global_matrix[:,0].argsort()]
#            print(global_matrix.shape)
#            plt.imshow(global_matrix, interpolation='nearest')
#            plt.show()
            fileName = 'dataAnalysis/figures/%d/%d_global_matrix' % (current_label[start_idx], batch_idx)
            img_filename = fileName + '.png'
            if not os.path.exists(os.path.dirname(img_filename)):
                try:
                    os.makedirs(os.path.dirname(img_filename))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            scipy.misc.imsave(img_filename, global_matrix)
#            #-------------------------------------------------------------------
#            # get critical points
#            #-------------------------------------------------------------------
#            for change_point in range(NUM_POINT):
#                current_data[start_idx, change_point, :] = no_influence_position.copy()
#            
#            for change_point in range(NUM_POINT):
#                current_data[start_idx, change_point, :] = orgin_data[change_point, :].copy()
#                # Aggregating BEG
#                for vote_idx in range(num_votes):
#                    if FLAGS.model == 'pointnet_cls_basic':
#                        rotated_data = current_data[start_idx:end_idx, :, :] # directional pointNet
#                    else:
#                        rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
#                                                      vote_idx/float(num_votes) * np.pi * 2) # pointNet
#                    feed_dict = {ops['pointclouds_pl']: rotated_data,
#                                ops['labels_pl']: current_label[start_idx:end_idx],
#                                ops['is_training_pl']: is_training}
#
#                    global_feature_val = sess.run(ops['global_feature'],
#                                            feed_dict=feed_dict)
#
#                    global_feature_list.append(global_feature_val)
#
#            critical_points = []
#            max_feature = np.zeros(global_feature_list[0].size) - 10
#            feature_points = np.zeros(global_feature_list[0].size)
#            for index in range(len(global_feature_list)):
#                #distance = math.sqrt(((global_feature_list[index] - global_feature_list[-1]) ** 2).sum())
#                #distance_list.append(distance)
#                top = global_feature_list[index]
#                feature_points = np.where(top > max_feature, index, feature_points)
#                max_feature = np.where(top > max_feature, top, max_feature)
#                
#            for index in feature_points[0]:
#                critical_points.append(orgin_data[int(index), :])
#            critical_points = list(set([tuple(t) for t in critical_points]))
#            
#            fileName = 'dataAnalysis/figures/%d_critical_points' % (start_idx)
#            img_filename = fileName + '.jpg'
#            plyFileName = fileName + '.ply'
#            pc_util.write_ply(np.squeeze( critical_points),plyFileName)
##             pc_util.pyplot_draw_point_cloud(np.squeeze( critical_points ),'')
#            output_img = pc_util.draw_point_cloud(np.squeeze( critical_points))
#            scipy.misc.imsave(img_filename, output_img)
            

#            #-------------------------------------------------------------------
#            # get upper-bound points
#            #-------------------------------------------------------------------
#            upper_bound_points = np.empty_like(orgin_data.shape)
#            upper_bound_points = orgin_data.copy()
#            current_data[start_idx,:,:] = orgin_data.copy()
#
#            search_step = 0.05
#            stand_feature = np.empty_like(global_feature_list[-1].shape)
#            max_position = [-0.5,-0.5,-0.5]
#            min_position = [0.5, 0.5, 0.5]
#
#            for point_index in range(NUM_POINT):
#                max_position = np.maximum(max_position, current_data[start_idx,point_index,:])
#                min_position = np.minimum(min_position, current_data[start_idx,point_index,:])
#            
#            print(max_position)
#            print(min_position)
#            for vote_idx in range(num_votes):
#                if FLAGS.model == 'pointnet_cls_basic':
#                    rotated_data = current_data[start_idx:end_idx, :, :] # directional pointNet
#                else:
#                    rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
#                                                  vote_idx/float(num_votes) * np.pi * 2) # pointNet
#
#                feed_dict = {ops['pointclouds_pl']: rotated_data,
#                            ops['labels_pl']: current_label[start_idx:end_idx],
#                            ops['is_training_pl']: is_training}
#                
#                
#                global_feature_val = sess.run(ops['global_feature'],feed_dict=feed_dict)
#                stand_feature = global_feature_val.copy()
#
#            change_point = 0
#            current_data[start_idx,:,:] = orgin_data.copy()
#            for point_index in range(NUM_POINT):
#                if not (point_index in feature_points[0]):
#                    change_point = point_index
#                    break
#             
#            for x in np.linspace(min_position[0], max_position[0], (max_position[0]-min_position[0])//search_step +1):
#                for y in np.linspace(min_position[1], max_position[1], (max_position[1]-min_position[1])//search_step +1):
#                    for z in np.linspace(min_position[2], max_position[2], (max_position[2]-min_position[2])//search_step +1):
#                        current_data[start_idx,change_point,:] = (x,y,z) #+ orgin_position
# 
#                        # Aggregating BEG
#                        for vote_idx in range(num_votes):
#                             
#                            if FLAGS.model == 'pointnet_cls_basic':
#                                rotated_data = current_data[start_idx:end_idx, :, :] # directional pointNet
#                            else:
#                                rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
#                                                              vote_idx/float(num_votes) * np.pi * 2) # pointNet
#                                 
#                            feed_dict = {ops['pointclouds_pl']: rotated_data,
#                                        ops['labels_pl']: current_label[start_idx:end_idx],
#                                        ops['is_training_pl']: is_training}
# 
#                            global_feature_val = sess.run(ops['global_feature'],feed_dict=feed_dict)
# 
#                            if (global_feature_val <= stand_feature).all():
#                                upper_bound_points = np.append(upper_bound_points, np.array([[x,y,z]]),axis = 0) 
#             
#            fileName = 'dataAnalysis/figures/%d_upper_bound_points' % (start_idx)
#            img_filename = fileName + '.jpg'
#            plyFileName = fileName + '.ply'
#            pc_util.write_ply(np.squeeze(upper_bound_points),plyFileName)
##             pc_util.pyplot_draw_point_cloud(np.squeeze( upper_bound_points ),'')
#            output_img = pc_util.draw_point_cloud(np.squeeze(upper_bound_points))
#            scipy.misc.imsave(img_filename, output_img)

            current_data[start_idx,:,:] = orgin_data.copy()
            print('------Finished!---------\n')
            
if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()