import tensorflow as tf
import sys
import cv2
import numpy as np
import os
from tensorflow.python.platform import gfile

# def _float_feature(value):
#     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
# TFRecord save dir

# Open the file
train_filename =  '/viscam/u/lys/visual_mpc/pushing_data/softmotion30/test/finetune1.tfrecords'
writer = tf.python_io.TFRecordWriter(train_filename)
sequence_length = 45
feature = {}
for i in range(1):
    # touch_path = '/viscam/u/lys/visual_mpc/pushing_data/softmotion30/raw_data_train/touch/'+str(i)
    # action_path = '/viscam/u/lys/visual_mpc/pushing_data/softmotion30/raw_data_train/actions/'+str(i)+'.npy'
    # pose_path = '/viscam/u/lys/visual_mpc/pushing_data/softmotion30/raw_data_train/endpose/'+str(i)+'.npy'   
    touch_path = '/viscam/u/lys/transfer/touch/'+str(i+360)
    action_path = '/viscam/u/lys/transfer/actions/'+str(i+360)+'.npy'
    pose_path = '/viscam/u/lys/transfer/endpose/'+str(i+360)+'.npy'   

    print i
    file_list = os.listdir(touch_path)
    print len(file_list)
    # if len(file_list) < sequence_length + 15:
    #     continue
    action_seq = np.load(action_path)
    endpose_seq = np.load(pose_path)

    num = 0
    for cnt in range(sequence_length): 
        img_path = os.path.join(touch_path, str(cnt)+'.png')
        print img_path

        image_aux1_name = str(num) + '/image_aux1/encoded'
        action_name = str(num) + '/action'
        endeffector_pos_name = str(num) + '/endeffector_pos'
    
        img = cv2.imread(img_path)
        img = cv2.resize(img,(64,64))
        #print img.shape  64,64,3
        img = img.astype(np.uint8) 

        #action = np.array([0.1+num*0.1,0.2+num*0.1,0.3+num*0.1,0.4+num*0.1])
        #endpose = np.array([0.1+num*0.1,0.2+num*0.1,0.3+num*0.1])

        if cnt < 44:
            action = action_seq[cnt]
            endpose = endpose_seq[cnt]

        else:
            action = action_seq[cnt-1]
            endpose = endpose_seq[cnt-1]

        # feature = {
        #         image_aux1_name: _bytes_feature(img.tostring()), #(tf.compat.as_bytes(img.tostring())), #_int64_feature(label),
        #         action_name:_float_feature(action.tolist()),#tf.train.Feature(float_list=tf.train.FloatList(value=[0.1,0.2,0.3])), #_bytes_feature(tf.compat.as_bytes(str([1, 1, 1, 1]))),
        #         endeffector_pos_name:_float_feature(endpose.tolist())#tf.train.Feature(float_list=tf.train.FloatList(value=[0.1,0.2,0.3,0.4]))#_bytes_feature(tf.compat.as_bytes(str([1, 1, 1]))),
        # }
        feature[str(num) + '/action'] = _float_feature(action.tolist())
        feature[str(num) + '/endeffector_pos'] = _float_feature(endpose.tolist())
        feature[str(num) + '/image_aux1/encoded'] = _bytes_feature(img.tostring())
        num = num +1
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

        

writer.close()
sys.stdout.flush()



# for i in range(2):
#     if not i % 1000:
#         print( 'Train data: {}/{}'.format(i, len(train_addrs)))
#         sys.stdout.flush()
        
#     img = load_image(train_addrs[i])
#     label = train_labels[i]
    
#     feature = {'train/label': _int64_feature(label),
#                'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    
#     example = tf.train.Example(features=tf.train.Features(feature=feature))

#     writer.write(example.SerializeToString())

# writer.close()
# sys.stdout.flush()
    
