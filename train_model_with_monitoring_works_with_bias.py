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
import matplotlib.pyplot as plt

def load_training_data(dirIn, rnd_seed, nranks):
    mags = np.concatenate([np.load(os.path.join(dirIn, f'spec_no_dust{rnd_seed}_rank{rank}.npy')) for rank in range(nranks)], axis=0)
    redshift = np.concatenate([np.load(os.path.join(dirIn, f'redshift{rnd_seed}_rank{rank}.npy')) for rank in range(nranks)], axis=0)
    sfh = np.concatenate([np.load(os.path.join(dirIn, f'sfh{rnd_seed}_rank{rank}.npy')) for rank in range(nranks)], axis=0)
    return mags, redshift, sfh

def preprocess_data(mags, redshift, sfh, wave_min=1000, wave_max=50000):
    mag_mask = (np.min(mags, axis=1) > 1e-30) & (np.max(mags, axis=1) < 1e-1)
    mags = mags[mag_mask]
    redshift = redshift[mag_mask]
    sfh = sfh[mag_mask]
    sps_inputs = np.concatenate((redshift[:, np.newaxis], sfh), axis=1)
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

def plot_training_history(history):
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('plots/losses.png')
    plt.show()

def validate_random_points(model, val_data, val_target, scaler_output, n_points=3):
    np.random.seed(10)
    random_indices = np.random.choice(val_data.shape[0], n_points, replace=False)
    for idx in random_indices:
        true_target = scaler_output.inverse_transform(val_target[idx:idx+1])[0]
        model_output = scaler_output.inverse_transform(model.predict(val_data[idx:idx+1]))[0]
        ratio = model_output / true_target

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(true_target, label='True Target')
        plt.plot(model_output, label='Model Output')
        plt.yscale('log')
        plt.xlabel('Wavelength Index')
        plt.ylabel('Flux')
        plt.legend()
        plt.title(f'Validation Point {idx}: True vs Model Output')

        plt.subplot(2, 1, 2)
        plt.plot(ratio, label='Ratio (Model/True)')
        plt.axhline(y=1.0, color='r', linestyle='--', label='Ideal Ratio')
        plt.xlabel('Wavelength Index')
        plt.ylabel('Ratio')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/validation_'+str(idx)+'.png')
        # plt.show()

def main():
    import os
    os.environ['NUMEXPR_MAX_THREADS'] = '64'
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    ifCentral = False
    if ifCentral:
        rnd_seed = 42
        dirIn = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_centrals_100k_z10/'    
    else:
        rnd_seed = 14
        dirIn = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_noncentrals_100k_z10/'
    ifCombined = True 
    if ifCombined:
        dirIn1 = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_centrals_100k_z10/'
        dirIn2 = '/lcrc/project/cosmo_ai/nramachandra/Projects/SPHEREx/MAH/TrainingData/Data/Damp_red_noncentrals_100k_z10/'
    nranks = 16
    if not ifCombined: 
        mags, redshift, sfh = load_training_data(dirIn, rnd_seed, nranks)
    if ifCombined:
        rnd_seed = 42
        mags1, redshift1, sfh1 = load_training_data(dirIn1, rnd_seed, nranks)
        rnd_seed = 14
        mags2, redshift2, sfh2 = load_training_data(dirIn2, rnd_seed, nranks)
        mags = np.append(mags1, mags2, axis=0)
        redshift = np.append(redshift1, redshift2)
        sfh = np.append(sfh1, sfh2, axis=0)
    sps_inputs_scaled, mags_scaled, scaler_input, scaler_output = preprocess_data(mags, redshift, sfh)
    train_data, val_data, train_target, val_target = train_test_split(sps_inputs_scaled, mags_scaled, test_size=0.1, random_state=42)
    input_shape = train_data.shape[1]
    output_shape = train_target.shape[1]
    hidden_dims = [256, 512, 1024, 2048, 1024, 512, 512]
    model = build_model(input_shape, output_shape, hidden_dims)
    learning_rate = 1e-3
    decay_rate = 1.0
    num_epochs = 200
    batch_size = 64 # 64
    model, history = train_model(model, train_data, train_target, val_data, val_target,
                                 learning_rate, decay_rate, num_epochs, batch_size)
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
    plot_training_history(history)
    validate_random_points(model, val_data, val_target, scaler_output)

if __name__ == '__main__':
    main()