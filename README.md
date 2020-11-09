How_to_simulate_a_self_driving_car using Udacity's Unity Simulator
Download Simulator : https://github.com/udacity/self-driving-car-sim
Install necessary packages like tensorflow,numpy,pandas,flask,socketio etc.
To train the model
You'll need the data folder inside current working folder which contains the training images.
python model.py

This will generate a file model-<epoch>.h5 whenever the performance in the epoch is better than the previous best. 
Rename Best accuracy file to model.h5
and to run the client type python test.py model.h5
