import os
import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf
import segmentation_models as sm
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_json
from models.models import unet

sm.set_framework('tf.keras')
sm.framework()

# convert all the audio to magnitude and phase spectograms and save it as matrix
def numpy_audio_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft):
    """This function takes as input a numpy audi of size (nb_frame,frame_length), and return
    a numpy containing the matrix spectrogram for amplitude in dB and phase. It will have the size
    (nb_frame,dim_square_spec,dim_square_spec)"""

    def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio):
        """This function takes an audio and convert into spectrogram,
        it returns the magnitude in dB and the phase"""

        
        stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
        stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)

        stftaudio_magnitude_db = librosa.amplitude_to_db(
            stftaudio_magnitude, ref=np.max)

        return stftaudio_magnitude_db, stftaudio_phase

    # we extract the magnitude vectors from the 256-point STFT vectors and 
    # take the first 129-point by removing the symmetric half.

    nb_audio = numpy_audio.shape[0]
    # dim_square_spec = 256/2
    m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
    m_phase = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=complex)

    for i in range(nb_audio):
        m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(
            n_fft, hop_length_fft, numpy_audio[i])

    return m_mag_db, m_phase


def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):
    """This function take an audio and split into several frame
    in a numpy matrix of size (nb_frame,frame_length)"""

    sequence_sample_length = sound_data.shape[0]
    # Creating several audio frames using sliding windows
    sound_data_list = [sound_data[start:start + frame_length] for start in range(
    0, sequence_sample_length - frame_length + 1, hop_length_frame)]  # get sliding windows
    # Combining all the frames to single matrix
    sound_data_array = np.vstack(sound_data_list)
    return sound_data_array


# Function to convert audio to frames matrix and combining all the frames matrix to single audio matrix
def audio_files_to_numpy(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):
    """This function take audio files of a directory and merge them
    in a numpy matrix of size (nb_frame,frame_length) for a sliding window of size hop_length_frame"""

    def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):
        """This function take an audio and split into several frame
        in a numpy matrix of size (nb_frame,frame_length)"""

        sequence_sample_length = sound_data.shape[0]
        # Creating several audio frames using sliding windows
        sound_data_list = [sound_data[start:start + frame_length] for start in range(
        0, sequence_sample_length - frame_length + 1, hop_length_frame)]  # get sliding windows
        # Combining all the frames to single matrix
        sound_data_array = np.vstack(sound_data_list)
        return sound_data_array

    list_sound_array = []
    count = 0
    for file in list_audio_files:
        # open the audio file
        try:
            y, sr = librosa.load(os.path.join(audio_dir, file), sr=sample_rate)
            # Getting duration of audio file
            total_duration = librosa.get_duration(y=y, sr=sr)
        except ZeroDivisionError:
            count += 1

            # Check if the duration is atleast the minimum duration
        if (total_duration >= min_duration):
            list_sound_array.append(audio_to_audio_frame_stack(
                y, frame_length, hop_length_frame))
        else:
            print(
                f"The following file {os.path.join(audio_dir,file)} is below the min duration")

    return np.vstack(list_sound_array)
    
	
# Data Preparation
def create_data(noise_dir, voice_dir,path_save_spectrogram, sample_rate,
min_duration, frame_length, hop_length_frame, hop_length_frame_noise, nb_samples, n_fft, hop_length_fft):
    """This function will randomly blend some clean voices from voice_dir with some noises from noise_dir
    and save the spectrograms of noisy voice, noise and clean voices to disk as well as complex phase,
    time series and sounds. This aims at preparing datasets for denoising training. It takes as inputs
    parameters defined in args module"""
    # function to convert the audio to magnitude and phase spectograms
    # convert all the audio to magnitude and phase spectograms and save it as matrix
    def numpy_audio_to_matrix_spectrogram(numpy_audio, dim_square_spec, n_fft, hop_length_fft):
        """This function takes as input a numpy audi of size (nb_frame,frame_length), and return
        a numpy containing the matrix spectrogram for amplitude in dB and phase. It will have the size
        (nb_frame,dim_square_spec,dim_square_spec)"""

        def audio_to_magnitude_db_and_phase(n_fft, hop_length_fft, audio):
            """This function takes an audio and convert into spectrogram,
            it returns the magnitude in dB and the phase"""

            
            stftaudio = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length_fft)
            stftaudio_magnitude, stftaudio_phase = librosa.magphase(stftaudio)

            stftaudio_magnitude_db = librosa.amplitude_to_db(
                stftaudio_magnitude, ref=np.max)

            return stftaudio_magnitude_db, stftaudio_phase
        
        # we extract the magnitude vectors from the 256-point STFT vectors and 
        # take the first 129-point by removing the symmetric half.

        nb_audio = numpy_audio.shape[0]
        # dim_square_spec = 256/2
        m_mag_db = np.zeros((nb_audio, dim_square_spec, dim_square_spec))
        m_phase = np.zeros((nb_audio, dim_square_spec, dim_square_spec), dtype=complex)

        for i in range(nb_audio):
            m_mag_db[i, :, :], m_phase[i, :, :] = audio_to_magnitude_db_and_phase(
                n_fft, hop_length_fft, numpy_audio[i])

        return m_mag_db, m_phase

    # Function to convert audio to frames matrix and combining all the frames matrix to single audio matrix
    def audio_files_to_numpy(audio_dir, list_audio_files, sample_rate, frame_length, hop_length_frame, min_duration):
        """This function take audio files of a directory and merge them
        in a numpy matrix of size (nb_frame,frame_length) for a sliding window of size hop_length_frame"""
        
        def audio_to_audio_frame_stack(sound_data, frame_length, hop_length_frame):
            """This function take an audio and split into several frame
            in a numpy matrix of size (nb_frame,frame_length)"""

            sequence_sample_length = sound_data.shape[0]
            # Creating several audio frames using sliding windows
            sound_data_list = [sound_data[start:start + frame_length] for start in range(
            0, sequence_sample_length - frame_length + 1, hop_length_frame)]  # get sliding windows
            # Combining all the frames to single matrix
            sound_data_array = np.vstack(sound_data_list)
            return sound_data_array

        list_sound_array = []

        count = 0
        for file in list_audio_files:
            # open the audio file
            try:
                y, sr = librosa.load(os.path.join(audio_dir, file), sr=sample_rate)
                # Getting duration of audio file
                total_duration = librosa.get_duration(y=y, sr=sr)
            except ZeroDivisionError:
                count += 1

                # Check if the duration is atleast the minimum duration
            if (total_duration >= min_duration):
                list_sound_array.append(audio_to_audio_frame_stack(
                    y, frame_length, hop_length_frame))
            else:
                print(
                    f"The following file {os.path.join(audio_dir,file)} is below the min duration")

        return np.vstack(list_sound_array)
    
    # Function to Blend a noise to clean speech audio
    def blend_noise_randomly(voice, noise, nb_samples, frame_length):
        """This function takes as input numpy arrays representing frames
        of voice sounds, noise sounds and the number of frames to be created
        and return numpy arrays with voice randomly blend with noise"""

        prod_voice = np.zeros((nb_samples, frame_length))
        prod_noise = np.zeros((nb_samples, frame_length))
        prod_noisy_voice = np.zeros((nb_samples, frame_length))

        for i in range(nb_samples):
            id_voice = np.random.randint(0, voice.shape[0])
            id_noise = np.random.randint(0, noise.shape[0])
            level_noise = np.random.uniform(0.2, 0.8)
            prod_voice[i, :] = voice[id_voice, :]
            prod_noise[i, :] = level_noise * noise[id_noise, :]
            prod_noisy_voice[i, :] = prod_voice[i, :] + prod_noise[i, :]

        return prod_voice, prod_noise, prod_noisy_voice
        


    list_noise_files = os.listdir(noise_dir)
    list_voice_files = os.listdir(voice_dir)

    def remove_ds_store(lst):
        """remove mac specific file if present"""
        if '.DS_Store' in lst:
            lst.remove('.DS_Store')

        return lst

    list_noise_files = remove_ds_store(list_noise_files)
    list_voice_files = remove_ds_store(list_voice_files)


    # Extracting noise and voice from folder and convert to numpy
    noise = audio_files_to_numpy(noise_dir, list_noise_files, sample_rate,
                                     frame_length, hop_length_frame_noise, min_duration)

    voice = audio_files_to_numpy(voice_dir, list_voice_files,
                                     sample_rate, frame_length, hop_length_frame, min_duration)

    # Blend some clean voices with random selected noises (and a random level of noise)
    prod_voice, prod_noise, prod_noisy_voice = blend_noise_randomly(
            voice, noise, nb_samples, frame_length)


    # Squared spectrogram dimensions
    dim_square_spec = int(n_fft / 2) + 1

    # Create Amplitude and phase of the sounds
    m_amp_db_voice,  m_pha_voice = numpy_audio_to_matrix_spectrogram(
            prod_voice, dim_square_spec, n_fft, hop_length_fft)
    m_amp_db_noise,  m_pha_noise = numpy_audio_to_matrix_spectrogram(
            prod_noise, dim_square_spec, n_fft, hop_length_fft)
    m_amp_db_noisy_voice,  m_pha_noisy_voice = numpy_audio_to_matrix_spectrogram(
            prod_noisy_voice, dim_square_spec, n_fft, hop_length_fft)

    np.save(f'{path_save_spectrogram}voice_amp_db', m_amp_db_voice)
    np.save(f'{path_save_spectrogram}noise_amp_db', m_amp_db_noise)      
    np.save(f'{path_save_spectrogram}noisy_voice_amp_db', m_amp_db_noisy_voice)
	
	

# functions for scaling the audio files
def scaled_in(matrix_spec):
    "global scaling apply to noisy voice spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec + 46)/50
    return matrix_spec
def scaled_ou(matrix_spec):
    "global scaling apply to noise models spectrograms (scale between -1 and 1)"
    matrix_spec = (matrix_spec -6 )/82
    return matrix_spec



def training_unet(path_save_spectrogram, weights_path, epochs, batch_size):
    """ This function will read noisy voice and clean voice spectrograms created by data_creation mode,
    and train a Unet model on this dataset for epochs and batch_size specified. It saves best models to disk regularly.
    """
    #load noisy voice & clean voice spectrograms created by data_creation mode
    X_in = np.load(path_save_spectrogram +'noisy_voice_amp_db'+".npy")
    X_ou = np.load(path_save_spectrogram +'voice_amp_db'+".npy")
    #Model of noise to predict
    X_ou = X_in - X_ou

    #Check distribution
    print(stats.describe(X_in.reshape(-1,1)))
    print(stats.describe(X_ou.reshape(-1,1)))

    #to scale between -1 and 1
    X_in = scaled_in(X_in)
    X_ou = scaled_ou(X_ou)

    #Check shape of spectrograms
    print(X_in.shape)
    print(X_ou.shape)
    #Check new distribution
    print(stats.describe(X_in.reshape(-1,1)))
    print(stats.describe(X_ou.reshape(-1,1)))


    #Reshape for training
    X_in = X_in[:,:,:]
    X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
    X_ou = X_ou[:,:,:]
    X_ou = X_ou.reshape(X_ou.shape[0],X_ou.shape[1],X_ou.shape[2],1)
    # print(X_in.shape)
    # print(X_out.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_in, X_ou, test_size=0.10, random_state=42)

    generator_nn=unet()

    #Save best models to disk during training
    checkpoint = ModelCheckpoint(weights_path+'/model_unet_best.keras', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')

    generator_nn.summary()

    #Training
    history = generator_nn.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[checkpoint], verbose=1, validation_data=(X_test, y_test))
    model_in_json = generator_nn.to_json()

    #Saving Model
    with open(weights_path+'model_unet.json','w') as json_file:
      json_file.write(model_in_json)

    #Plot training and validation loss (log scale)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.yscale('log')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def training_unetRes(path_save_spectrogram, weights_path, epochs, batch_size):
    """ This function will read noisy voice and clean voice spectrograms created by data_creation mode,
    and train a Unet model on this dataset for epochs and batch_size specified.
    """
    #load noisy voice & clean voice spectrograms created by data_creation mode
    X_in = np.load(path_save_spectrogram +'noisy_voice_amp_db'+".npy")
    X_ou = np.load(path_save_spectrogram +'voice_amp_db'+".npy")
    #Model of noise to predict
    X_ou = X_in - X_ou

    #Check distribution
    print(stats.describe(X_in.reshape(-1,1)))
    print(stats.describe(X_ou.reshape(-1,1)))

    #to scale between -1 and 1
    X_in = scaled_in(X_in)
    X_ou = scaled_ou(X_ou)

    #Check shape of spectrograms
    print(X_in.shape)
    print(X_ou.shape)
    #Check new distribution
    print(stats.describe(X_in.reshape(-1,1)))
    print(stats.describe(X_ou.reshape(-1,1)))


    #Reshape for training
    X_in = X_in[:,:,:]
    X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
    X_ou = X_ou[:,:,:]
    X_ou = X_ou.reshape(X_ou.shape[0],X_ou.shape[1],X_ou.shape[2],1)

    # X_train, X_test, y_train, y_test = train_test_split(X_in, X_ou, test_size=0.10, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(X_in, X_ou, test_size=0.10, random_state=42)

    import segmentation_models as sm
    from segmentation_models import Unet
    from keras.layers import Input, Conv2D
    from keras.models import Model

    # define number of channels
    N = x_train.shape[-1]

    base_model = Unet(backbone_name='resnet101', encoder_weights='imagenet')

    inp = Input(shape=(None, None, N))
    l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    out = base_model(l1)

    model = Model(inp, out, name=base_model.name)
    BACKBONE = 'resnet101'
    preprocess_input = sm.get_preprocessing(BACKBONE)

    # preprocess input
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)

    # define model
    model.compile(
    'Adam',
    loss = tf.keras.losses.MeanSquaredError(),
     metrics = ['mae']
    )

    # fitting model
    checkpoint = ModelCheckpoint(weights_path+'/model_ResNet.keras', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=[checkpoint]
       )
    #Saving model in Json file
    model_in_json = model.to_json()
    with open('model_ResNet.json','w') as json_file:
      json_file.write(model_in_json)
    #Plot training and validation loss (log scale)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.yscale('log')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


from utils.helper import matrix_spectrogram_to_numpy_audio
from utils.helper import inv_scaled_ou

def prediction(weights_path, audio_dir_prediction, dir_save_prediction, audio_input_prediction, audio_output_prediction, config):
    """ This function takes as input pretrained weights, noisy voice sound to denoise, predict
    the denoise sound and save it to disk.
    """

    # load json and create model
    json_file = open('Best_json_Unet.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model/
    loaded_model.load_weights('Best_weight_Unet.keras')
    print("Loaded model from disk")

    # Extracting noise and voice from folder and convert to numpy
    audio = audio_files_to_numpy(audio_dir_prediction, audio_input_prediction, config.sample_rate,
                                 config.frame_length, config.hop_length_frame, config.min_duration)

    #Dimensions of squared spectrogram
    dim_square_spec = int(config.n_fft / 2) + 1
    print(dim_square_spec)

    # Create Amplitude and phase of the sounds
    m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(
        audio, dim_square_spec, config.n_fft, config.hop_length_fft)

    #global scaling to have distribution -1/1
    X_in = scaled_in(m_amp_db_audio)
    #Reshape for prediction
    X_in = X_in.reshape(X_in.shape[0],X_in.shape[1],X_in.shape[2],1)
    #Prediction using loaded network
    X_pred = loaded_model.predict(X_in)
    #Rescale back the noise model
    inv_sca_X_pred = inv_scaled_ou(X_pred)
    #Remove noise model from noisy speech
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:,:,:,0]
    #Reconstruct audio from denoised spectrogram and phase
    print(X_denoise.shape)
    print(m_pha_audio.shape)
    print(config.frame_length)
    print(config.hop_length_fft)
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, config.frame_length, config.hop_length_fft)
    #Number of frames
    nb_samples = audio_denoise_recons.shape[0]
    #Save all frames in one file
    denoise_long = audio_denoise_recons.reshape(1, nb_samples * config.frame_length)*10
    # librosa.output.write_wav(dir_save_prediction + audio_output_prediction, denoise_long[0, :], 1000)
    sf.write(dir_save_prediction + audio_output_prediction, denoise_long[0, :], 8000, 'PCM_24')
    # wavfile.write(dir_save_prediction + audio_output_prediction, 1000, denoise_long[0,:])


def prediction_deeplab(weights_path, audio_dir_prediction, dir_save_prediction, audio_input_prediction,
audio_output_prediction, config):
    """ This function takes as input pretrained weights, noisy voice sound to denoise, predict
    the denoise sound and save it to disk.
    """

    # load json and create model
    json_file = open('model_depplab.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model/
    loaded_model.load_weights('model_deep_encoded.keras')
    print("Loaded model from disk")

    # Extracting noise and voice from folder and convert to numpy
    audio = audio_files_to_numpy(audio_dir_prediction, audio_input_prediction, config.sample_rate,
                                 config.frame_length, config.hop_length_frame, config.min_duration)

    #Dimensions of squared spectrogram
    dim_square_spec = int(config.n_fft / 2) + 1
    print(dim_square_spec)

    # Create Amplitude and phase of the sounds
    m_amp_db_audio,  m_pha_audio = numpy_audio_to_matrix_spectrogram(
        audio, dim_square_spec, config.n_fft, config.hop_length_fft)

    #global scaling to have distribution -1/1
    X_in = scaled_in(m_amp_db_audio)
    X_in = np.dstack([X_in] * 3)
    #Reshape for prediction
    X_in = X_in.reshape(X_in.shape[0],128,128,3)
    #Prediction using loaded network
    X_pred = loaded_model.predict(X_in)
    #Rescale back the noise model
    inv_sca_X_pred = inv_scaled_ou(X_pred)
    #Remove noise model from noisy speech
    X_denoise = m_amp_db_audio - inv_sca_X_pred[:,:,:,0]
    #Reconstruct audio from denoised spectrogram and phase
    print(X_denoise.shape)
    print(m_pha_audio.shape)
    print(config.frame_length)
    print(config.hop_length_fft)
    audio_denoise_recons = matrix_spectrogram_to_numpy_audio(X_denoise, m_pha_audio, config.frame_length, config.hop_length_fft)
    #Number of frames
    nb_samples = audio_denoise_recons.shape[0]
    #Save all frames in one file
    denoise_long = audio_denoise_recons.reshape(1, nb_samples * config.frame_length)*10
    # librosa.output.write_wav(dir_save_prediction + audio_output_prediction, denoise_long[0, :], 1000)
    sf.write(dir_save_prediction + audio_output_prediction, denoise_long[0, :], 8000, 'PCM_24')
    # wavfile.write(dir_save_prediction + audio_output_prediction, 1000, denoise_long[0,:])