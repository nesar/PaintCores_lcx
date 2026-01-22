# train_model.py

import numpy as np
import os
import glob
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
import tensorflow as tf

def load_training_data(dirIn, rnd_seed, nranks):
    # Load data from files
    mags = np.concatenate([np.load(os.path.join(dirIn, f'spec_no_dust{rnd_seed}_rank{rank}.npy')) for rank in range(nranks)], axis=0)
    redshift = np.concatenate([np.load(os.path.join(dirIn, f'redshift{rnd_seed}_rank{rank}.npy')) for rank in range(nranks)], axis=0)
    sfh = np.concatenate([np.load(os.path.join(dirIn, f'sfh{rnd_seed}_rank{rank}.npy')) for rank in range(nranks)], axis=0)
    # You can load other parameters if needed

    return mags, redshift, sfh

def preprocess_data(mags, redshift, sfh, wave_min=1000, wave_max=50000):
    # Filter wavelengths between wave_min and wave_max if necessary
    # Assuming mags already corresponds to the desired wavelength range

    # Remove invalid mags
    mag_mask = (np.min(mags, axis=1) > 1e-30) & (np.max(mags, axis=1) < 1e-1)
    mags = mags[mag_mask]
    redshift = redshift[mag_mask]
    sfh = sfh[mag_mask]

    # Prepare input and output data
    # Input: concatenate redshift and SFH
    sps_inputs = np.concatenate((redshift[:, np.newaxis], sfh), axis=1)
    # Output: mags

    # Scale input and output data
    scaler_input = MinMaxScaler()
    scaler_output = StandardScaler()

    sps_inputs_scaled = scaler_input.fit_transform(sps_inputs)
    mags_scaled = scaler_output.fit_transform(mags)

    return sps_inputs_scaled, mags_scaled, scaler_input, scaler_output

def build_model(input_shape, output_shape, hidden_dims):
    model = Sequential()
    model.add(Dense(hidden_dims[0], activation='relu', kernel_initializer='he_normal', input_shape=(input_shape,)))

    for hidden_dim in hidden_dims[1:]:
        model.add(Dense(hidden_dim, activation='relu', kernel_initializer='he_normal'))

    model.add(Dense(output_shape, activation='linear'))

    model.compile(optimizer='adam', loss='mse')

    return model

def train_model(model, train_data, train_target, val_data, val_target, learning_rate, decay_rate, num_epochs, batch_size):
    K.set_value(model.optimizer.lr, learning_rate)
    K.set_value(model.optimizer.decay, decay_rate)

    history = model.fit(train_data, train_target,
                        validation_data=(val_data, val_target),
                        epochs=num_epochs,
                        batch_size=batch_size,
                        verbose=1)
    return model, history

def save_model_and_scalers(model, scaler_input, scaler_output, model_path, scaler_input_path, scaler_output_path):
    model.save(model_path)
    with open(scaler_input_path, 'wb') as f:
        pickle.dump(scaler_input, f)
    with open(scaler_output_path, 'wb') as f:
        pickle.dump(scaler_output, f)

# def main():
#     # Set parameters
#     ifCentral = False
#     if ifCentral:
#         rnd_seed = 42
#         dirIn = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_centrals_100k_z10/'    
#     else:
#         rnd_seed = 14
#         dirIn = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_noncentrals_100k_z10/'

        

#     nranks = 16  # Adjust as needed

#     # Load data
#     mags, redshift, sfh = load_training_data(dirIn, rnd_seed, nranks)

#     # Preprocess data
#     sps_inputs_scaled, mags_scaled, scaler_input, scaler_output = preprocess_data(mags, redshift, sfh)

#     # Split data into training and validation sets
#     train_data, val_data, train_target, val_target = train_test_split(sps_inputs_scaled, mags_scaled, test_size=0.1, random_state=42)

#     # Define model
#     input_shape = train_data.shape[1]
#     output_shape = train_target.shape[1]
#     hidden_dims = [512, 512, 1024, 1024, 2048, 2048]

#     model = build_model(input_shape, output_shape, hidden_dims)

#     # Train model
#     learning_rate = 1e-5
#     decay_rate = 1.0
#     num_epochs = 1000
#     batch_size = 256

#     model, history = train_model(model, train_data, train_target, val_data, val_target,
#                                  learning_rate, decay_rate, num_epochs, batch_size)

#     # Save model and scalers
#     if ifCentral:
#         model_path = './paint_models/spec_mlp_eline_damp_central_z10.h5'
#         scaler_input_path = './paint_models/eline_input_scale_damp_central_z10.pkl'
#         scaler_output_path = './paint_models/eline_output_scale_damp_central_z10.pkl'
#     else:
#         model_path = './paint_models/spec_mlp_eline_damp_noncentral_z10.h5'
#         scaler_input_path = './paint_models/eline_input_scale_damp_noncentral_z10.pkl'
#         scaler_output_path = './paint_models/eline_output_scale_damp_noncentral_z10.pkl'

#     save_model_and_scalers(model, scaler_input, scaler_output, model_path, scaler_input_path, scaler_output_path)
    
    
def main():
    # Set the NUMEXPR_MAX_THREADS environment variable
    import os
    os.environ['NUMEXPR_MAX_THREADS'] = '64'  # Adjust as needed
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "" #### Training on CPU: If the above adjustments don’t resolve the issue, you can force TensorFlow to use the CPU:

    # Allow GPU memory growth
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # Set parameters
    ifCentral = False
    if ifCentral:
        rnd_seed = 42
        dirIn = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_centrals_100k_z10/'    
    else:
        rnd_seed = 14
        dirIn = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_noncentrals_100k_z10/'
        
    ifCombined = True 
    if ifCombined:
        # rnd_seed = 42
        dirIn1 = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_centrals_100k_z10/'
        dirIn2 = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_noncentrals_100k_z10/'

    nranks = 16  # Adjust as needed

    if not ifCombined: 
        # Load data
        mags, redshift, sfh = load_training_data(dirIn, rnd_seed, nranks)
    
    if ifCombined:
        rnd_seed = 42
        mags1, redshift1, sfh1 = load_training_data(dirIn1, rnd_seed, nranks)
        rnd_seed = 14
        mags2, redshift2, sfh2 = load_training_data(dirIn2, rnd_seed, nranks)
        
        print(mags1.shape, redshift1.shape, sfh1.shape)
        print(mags2.shape, redshift2.shape, sfh2.shape)
        
        mags = np.append(mags1, mags2, axis=0)
        redshift = np.append(redshift1, redshift2)
        sfh = np.append(sfh1, sfh2, axis=0)
        
        print(mags.shape, redshift.shape, sfh.shape)
        
        

    # Preprocess data
    sps_inputs_scaled, mags_scaled, scaler_input, scaler_output = preprocess_data(mags, redshift, sfh)

    # Split data into training and validation sets
    train_data, val_data, train_target, val_target = train_test_split(sps_inputs_scaled, mags_scaled, test_size=0.1, random_state=42)

    # Define model
    input_shape = train_data.shape[1]
    output_shape = train_target.shape[1]
    hidden_dims = [256, 256, 512, 512, 512, 512]  # Reduced layer sizes

    model = build_model(input_shape, output_shape, hidden_dims)

    # Train model
    learning_rate = 1e-5
    decay_rate = 1.0
    num_epochs = 1000
    batch_size = 64  # Reduced batch size

    model, history = train_model(model, train_data, train_target, val_data, val_target,
                                 learning_rate, decay_rate, num_epochs, batch_size)

    # Save model and scalers
    if ifCentral:
        model_path = './trained_painting_NNs/spec_mlp_eline_damp_central_z10.h5'
        scaler_input_path = './trained_painting_NNs/eline_input_scale_damp_central_z10.pkl'
        scaler_output_path = './trained_painting_NNs/eline_output_scale_damp_central_z10.pkl'
    else:
        model_path = './trained_painting_NNs/spec_mlp_eline_damp_noncentral_z10.h5'
        scaler_input_path = './trained_painting_NNs/eline_input_scale_damp_noncentral_z10.pkl'
        scaler_output_path = './trained_painting_NNs/eline_output_scale_damp_noncentral_z10.pkl'
        
    if ifCombined:
        model_path = './trained_painting_NNs/spec_mlp_eline_damp_central_noncentral_z10.h5'
        scaler_input_path = './trained_painting_NNs/eline_input_scale_damp_central_noncentral_z10.pkl'
        scaler_output_path = './trained_painting_NNs/eline_output_scale_damp_central_noncentral_z10.pkl'  

    save_model_and_scalers(model, scaler_input, scaler_output, model_path, scaler_input_path, scaler_output_path)

if __name__ == '__main__':
    main()