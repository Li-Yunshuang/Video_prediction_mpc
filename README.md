# This is a quick guide for re-implement the video-MPC pipeline.
# Complementary document to the README file in this project.

1. "./pushing_data/softmotion30/python TFRead.py": Load dataset with images, actions and endposes for each frame. The length of actions for each frame is free to change.

2. "./pythhon_visual_mpc/video_prediction/python prediction_train_sawyer.py --hyper ../../tensorflow_data/sawyer/1stimg_bckgd_cdna/conf.py --visualize modelxxx": This is for training video prediction model with configuration file, options to visualize it or not (in gif format for certain model). 

3. "./python_visual_mpc/video_prediction/python predict_actions.py": Generate best actions for a test tatile images sequences. TODO: This hasn't been implemented well since when sampling random actions, it's not good to sample too many because the memory will run out.



