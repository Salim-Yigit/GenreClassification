import numpy as np 
import pandas as pd 
import matplotlib.pylab as plt 
from scipy.io import wavfile  
import os   
from scipy import signal 
import csv 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler

def load_from_folder(folder_path,train=True,train_size=20,test_size=10):  
    songs = [] 
    cnt = 0 
    if train: 
        for filename in os.listdir(folder_path): 
            if cnt < train_size : 
                sample_rate , song = wavfile.read(os.path.join(folder_path,filename)) 
                if song is not None:
                    songs.append(song) 
                    cnt = cnt + 1 
    else : 
        for i, filename in enumerate(os.listdir(folder_path)): 
            if i > train_size and cnt < test_size: 
               sample_rate , song = wavfile.read(os.path.join(folder_path,filename))   
               if song is not None : 
                   songs.append(song)   
                   cnt = cnt  + 1 
    return songs 

def FFT(x):
    """
    A recursive implementation of 
    the 1D Cooley-Tukey FFT, the 
    input should have a length of 
    power of 2. 
    """
    N = len(x)
    
    if N == 1:
        return x
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = \
          np.exp(-2j*np.pi*np.arange(N)/ N)
        
        X = np.concatenate(\
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])
        return X 


def stft(signal, window_size, overlap_ratio=0.2,window_type='hamming'):
    hop_length = int(window_size * (1 - overlap_ratio))

    # Compute the number of frames
    num_frames = 1 + (len(signal) - window_size) // hop_length

    # Create an empty matrix to store the STFT coefficients
    stft_matrix = np.zeros((window_size, num_frames), dtype=np.complex128)

    for frame_idx in range(num_frames):
        # Calculate the start and end indices for the current frame
        start = frame_idx * hop_length
        end = start + window_size

        # Extract the current frame from the signal
        frame = signal[start:end]

        # Apply the window function to the frame 
        if(window_type == 'hanning') : 
            windowed_frame = frame * np.hanning(window_size) 
        elif(window_type == 'bartlett') : 
            windowed_frame = frame * np.bartlett(window_size) 
        else : 
            windowed_frame = frame * np.hamming(window_size) 

        # Perform the FFT on the windowed frame
        fft_frame = np.fft.fft(windowed_frame)

        # Store the FFT coefficients in the STFT matrix
        stft_matrix[:, frame_idx] = fft_frame

    return stft_matrix


def plot_spectrogram(magnitude_spectrogram, hop_size, sample_rate):
    """
    Plot the magnitude spectrogram.

    Args:
        magnitude_spectrogram (numpy.ndarray): The magnitude spectrogram.
        hop_size (int): The hop size used in the STFT.
        sample_rate (int): The sample rate of the original signal.
    """
    # Calculate the time and frequency axes for the spectrogram
    num_frames, window_size = magnitude_spectrogram.shape
    duration = num_frames * hop_size / sample_rate
    frequencies = np.arange(window_size) * sample_rate / window_size
    times = np.arange(num_frames) * hop_size / sample_rate

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.imshow(magnitude_spectrogram.T, aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Magnitude Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show() 

def calculate_features(stft_matrix):
    num_windows = stft_matrix.shape[1]
    num_bins = stft_matrix.shape[0]
    freq_axis = np.arange(num_bins)
    features = []

    for window_idx in range(num_windows):
        stft_window = stft_matrix[:, window_idx]
        power = np.sum(np.abs(stft_window) ** 2) / stft_window.size
        freq_avg = np.average(freq_axis, weights=np.abs(stft_window))
        ampl_avg = np.average(np.abs(stft_window))
        features.append([power, freq_avg, ampl_avg])
    return features 

def process_genres(data,genre,train=True,window_type='hamming',train_size=90,test_size=10) : 
    if train : 
        num =  train_size
    else : 
        num = test_size - 1 
    features_genre = []
    for i in range(num): 
        print(i)
        stft_matrix = stft(data[i],1024,0.2,window_type=window_type) 
        features = calculate_features(stft_matrix=stft_matrix) 
        array = np.array(features)  
        mean_values = np.mean(array, axis=0)
        std_values = np.std(array, axis=0)
        median_values = np.median(array, axis=0) 

        
        features_dict = {
        'Power_mean': mean_values[0],
        'Amplitude_mean': mean_values[1],
        'Weighted_Mean_Frequency_mean': mean_values[2], 
        'Power_std': std_values[0],
        'Amplitude_std': std_values[1],
        'Weighted_Mean_Frequency_std': std_values[2],  
        'Power_median': median_values[0],
        'Amplitude_median': median_values[1],
        'Weighted_Mean_Frequency_median': median_values[2], 
        'Genre': genre
        } 
        features_genre.append(features_dict)
    if train: 
        csv_file = f'{genre}_{window_type}_train.csv' 
    else : 
        csv_file = f'{genre}_{window_type}_test.csv'

    # Get the keys from the first entry in the features list
    keys = list(features_genre[0].keys())

    # Save the features to the CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(features_genre) 

def concant_files(file1,file2,file3,file4,file5,file6,file7,file8,file9,file10,train=True,window_type='hamming'): 
    df_1 = pd.read_csv(file1) 
    df_2 = pd.read_csv(file2) 
    df_3 = pd.read_csv(file3) 
    df_4 = pd.read_csv(file4) 
    df_5 = pd.read_csv(file5) 
    df_6 = pd.read_csv(file6) 
    df_7 = pd.read_csv(file7) 
    df_8 = pd.read_csv(file8) 
    df_9 = pd.read_csv(file9) 
    df_10 = pd.read_csv(file10)

    df_all = pd.concat([df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10]) 

    if train: 
        df_all.to_csv(f'{window_type}_training.csv',index=False) 
    else : 
        df_all.to_csv(f'{window_type}_test.csv',index=False)

def get_ready_training(filename) : 
    df_train = pd.read_csv(filename) 
    df_train['Genre'] = df_train['Genre'].map({'blues' :0, 'metal' :1, 'disco' :2,
                                               'hiphop':3, 'classical':4,
                                               'country':5,'jazz':6,'pop':7,
                                               'reggae':8,'rock':9}).astype(int) 
    
    x = df_train.drop('Genre',axis=1)
    y = df_train['Genre'] 

    return x,y

def prediction(y_test,y_pred): 
    cnt = 0
    for i in range(len(y_test)): 
        #print('Actual class:',y_test[i],' Predicted class:',y_pred[i])
        if(y_test[i] == y_pred[i]): 
            cnt = cnt + 1 
    df = pd.DataFrame({'Actual': pd.Series(dtype='int'),
                   'Predicted': pd.Series(dtype='int')})
    
    df['Actual'] = y_test 
    df['Predicted'] = y_pred
    df['Actual'] = df['Actual'].map({0:'blues' , 1:'metal' , 2:'disco' ,
                                               3:'hiphop', 4:'classical',
                                               5:'country',6:'jazz',7:'pop',
                                               8:'reggae',9:'rock'})   
    
    df['Predicted'] = df['Predicted'].map({0:'blues' , 1:'metal' , 2:'disco' ,
                                               3:'hiphop', 4:'classical',
                                               5:'country',6:'jazz',7:'pop',
                                               8:'reggae',9:'rock'}) 
    
    print(df.head(30))
    score = (cnt / len(y_test)) * 100
    print('Yüzdelik doğruluk oranı:', score)


def fill_zeros_with_average(song_list):
    filled_list = []
    for song in song_list:
        if isinstance(song, np.ndarray) and np.count_nonzero(song) < song.size:
            average = np.mean(song[np.nonzero(song)])  # Sıfır olmayan değerlerin ortalaması
            filled_song = np.where(song == 0, average, song)  # Sıfır olan değerleri ortalama ile doldurma
            filled_list.append(filled_song)
        else:
            filled_list.append(song)
    return filled_list

if __name__ == '__main__': 
    # samplerate, data = wavfile.read('Data/genres_original/blues/blues.00000.wav')  
    
    blues_train = load_from_folder('../Data/genres_original/blues',train=True,train_size=90,test_size=10)  
    blues_test = load_from_folder('../Data/genres_original/blues',train=False,train_size=90,test_size=10)
    print('blues okey')
    
    classical_train = load_from_folder('../Data/genres_original/classical',train=True,train_size=90,test_size=10)  
    classical_test = load_from_folder('../Data/genres_original/classical',train=False,train_size=90,test_size=10) 

    disco_train = load_from_folder('../Data/genres_original/disco',train=True,train_size=90,test_size=10)  
    disco_test = load_from_folder('../Data/genres_original/disco',train=False,train_size=90,test_size=10) 

    hiphop_train = load_from_folder('../Data/genres_original/hiphop',train=True,train_size=90,test_size=10)  
    hiphop_test = load_from_folder('../Data/genres_original/hiphop',train=False,train_size=90,test_size=10) 

    metal_train = load_from_folder('../Data/genres_original/metal',train=True,train_size=90,test_size=10)  
    metal_test = load_from_folder('../Data/genres_original/metal',train=False,train_size=90,test_size=10) 

    country_train = load_from_folder('../Data/genres_original/country',train=True,train_size=90,test_size=10)  
    country_test = load_from_folder('../Data/genres_original/country',train=False,train_size=90,test_size=10)  
    
    jazz_train = load_from_folder('../Data/genres_original/jazz',train=True,train_size=89,test_size=10)  
    jazz_test = load_from_folder('../Data/genres_original/jazz',train=False,train_size=89,test_size=10) 

    pop_train = load_from_folder('../Data/genres_original/pop',train=True,train_size=90,test_size=10)  
    pop_test = load_from_folder('../Data/genres_original/pop',train=False,train_size=90,test_size=10) 

    reggae_train = load_from_folder('../Data/genres_original/reggae',train=True,train_size=90,test_size=10)  
    reggae_test = load_from_folder('../Data/genres_original/reggae',train=False,train_size=90,test_size=10) 

    rock_train = load_from_folder('../Data/genres_original/rock',train=True,train_size=90,test_size=10)  
    rock_test = load_from_folder('../Data/genres_original/rock',train=False,train_size=90,test_size=10) 
    
    print('read correctly')
    hiphop_train = fill_zeros_with_average(hiphop_train) 
    classical_train = fill_zeros_with_average(classical_train)
    print(hiphop_train[43])
    # HANNING WINDOW TYPE
    process_genres(blues_train,'blues',window_type='hanning',train_size=90,test_size=10) 
    process_genres(metal_train,'metal',window_type='hanning',train_size=90,test_size=10)  
    process_genres(disco_train,'disco',window_type='hanning',train_size=90,test_size=10)  
    process_genres(hiphop_train,'hiphop',window_type='hanning',train_size=85,test_size=10)  
    process_genres(classical_train,'classical',window_type='hanning',train_size=90,test_size=10)  
    process_genres(country_train,'country',window_type='hanning',train_size=90,test_size=10)
    process_genres(jazz_train,'jazz',window_type='hanning',train_size=89,test_size=10)
    process_genres(pop_train,'pop',window_type='hanning',train_size=90,test_size=10) 
    process_genres(reggae_train,'reggae',window_type='hanning',train_size=90,test_size=10)
    process_genres(rock_train,'rock',window_type='hanning',train_size=90,test_size=10)

    concant_files('blues_hanning_train.csv','metal_hanning_train.csv','disco_hanning_train.csv',
                  'hiphop_hanning_train.csv','classical_hanning_train.csv'
                  ,'country_hanning_train.csv','jazz_hanning_train.csv'
                  ,'pop_hanning_train.csv','reggae_hanning_train.csv','rock_hanning_train.csv',window_type='hanning')  

    process_genres(blues_test,'blues',train=False,window_type='hanning',train_size=90,test_size=10) 
    process_genres(metal_test,'metal',train=False,window_type='hanning',train_size=90,test_size=10)  
    process_genres(disco_test,'disco',train=False,window_type='hanning',train_size=90,test_size=10)  
    process_genres(hiphop_test,'hiphop',train=False,window_type='hanning',train_size=85,test_size=10)  
    process_genres(classical_test,'classical',train=False,window_type='hanning',train_size=90,test_size=10)  
    process_genres(country_test,'country',train=False,window_type='hanning',train_size=90,test_size=10)
    process_genres(jazz_test,'jazz',train=False,window_type='hanning',train_size=89,test_size=10)
    process_genres(pop_test,'pop',train=False,window_type='hanning',train_size=90,test_size=10) 
    process_genres(reggae_test,'reggae',train=False,window_type='hanning',train_size=90,test_size=10)
    process_genres(rock_test,'rock',train=False,window_type='hanning',train_size=90,test_size=10)

    concant_files('blues_hanning_test.csv','metal_hanning_test.csv','disco_hanning_test.csv',
                  'hiphop_hanning_test.csv','classical_hanning_test.csv'
                  ,'country_hanning_test.csv','jazz_hanning_test.csv'
                  ,'pop_hanning_test.csv','reggae_hanning_test.csv','rock_hanning_test.csv',train=False,window_type='hanning')
    
    # BARTLETT WINDOW TYPE   
    process_genres(blues_train,'blues',window_type='bartlett',train_size=90,test_size=10) 
    process_genres(metal_train,'metal',window_type='bartlett',train_size=90,test_size=10)  
    process_genres(disco_train,'disco',window_type='bartlett',train_size=90,test_size=10)  
    process_genres(hiphop_train,'hiphop',window_type='bartlett',train_size=90,test_size=10)  
    process_genres(classical_train,'classical',window_type='bartlett',train_size=90,test_size=10)  
    process_genres(country_train,'country',window_type='bartlett',train_size=90,test_size=10)
    process_genres(jazz_train,'jazz',window_type='bartlett',train_size=89,test_size=10)
    process_genres(pop_train,'pop',window_type='bartlett',train_size=90,test_size=10) 
    process_genres(reggae_train,'reggae',window_type='bartlett',train_size=90,test_size=10)
    process_genres(rock_train,'rock',window_type='bartlett',train_size=90,test_size=10)

    concant_files('blues_bartlett_train.csv','metal_bartlett_train.csv','disco_bartlett_train.csv',
                  'hiphop_bartlett_train.csv','classical_bartlett_train.csv'
                  ,'country_bartlett_train.csv','jazz_bartlett_train.csv'
                  ,'pop_bartlett_train.csv','reggae_bartlett_train.csv','rock_bartlett_train.csv',window_type='bartlett')  

    process_genres(blues_test,'blues',train=False,window_type='bartlett',train_size=90,test_size=10) 
    process_genres(metal_test,'metal',train=False,window_type='bartlett',train_size=90,test_size=10)  
    process_genres(disco_test,'disco',train=False,window_type='bartlett',train_size=90,test_size=10)  
    process_genres(hiphop_test,'hiphop',train=False,window_type='bartlett',train_size=90,test_size=10)  
    process_genres(classical_test,'classical',train=False,window_type='bartlett',train_size=90,test_size=10)  
    process_genres(country_test,'country',train=False,window_type='bartlett',train_size=90,test_size=10)
    process_genres(jazz_test,'jazz',train=False,window_type='bartlett',train_size=89,test_size=10)
    process_genres(pop_test,'pop',train=False,window_type='bartlett',train_size=90,test_size=10) 
    process_genres(reggae_test,'reggae',train=False,window_type='bartlett',train_size=90,test_size=10)
    process_genres(rock_test,'rock',train=False,window_type='bartlett',train_size=90,test_size=10)

    concant_files('blues_bartlett_test.csv','metal_bartlett_test.csv','disco_bartlett_test.csv',
                  'hiphop_bartlett_test.csv','classical_bartlett_test.csv'
                  ,'country_bartlett_test.csv','jazz_bartlett_test.csv'
                  ,'pop_bartlett_test.csv','reggae_bartlett_test.csv','rock_bartlett_test.csv',train=False,window_type='bartlett')
 
    # HAMMING WINDOW TYPE
    process_genres(blues_train,'blues',window_type='hamming',train_size=90,test_size=10) 
    process_genres(metal_train,'metal',window_type='hamming',train_size=90,test_size=10)  
    process_genres(disco_train,'disco',window_type='hamming',train_size=90,test_size=10)  
    process_genres(hiphop_train,'hiphop',window_type='hamming',train_size=90,test_size=10)  
    process_genres(classical_train,'classical',window_type='hamming',train_size=90,test_size=10)  
    process_genres(country_train,'country',window_type='hamming',train_size=90,test_size=10)
    process_genres(jazz_train,'jazz',window_type='hamming',train_size=89,test_size=10)
    process_genres(pop_train,'pop',window_type='hamming',train_size=90,test_size=10) 
    process_genres(reggae_train,'reggae',window_type='hamming',train_size=90,test_size=10)
    process_genres(rock_train,'rock',window_type='hamming',train_size=90,test_size=10)

    concant_files('blues_hamming_train.csv','metal_hamming_train.csv','disco_hamming_train.csv',
                  'hiphop_hamming_train.csv','classical_hamming_train.csv'
                  ,'country_hamming_train.csv','jazz_hamming_train.csv'
                  ,'pop_hamming_train.csv','reggae_hamming_train.csv','rock_hamming_train.csv',window_type='hamming')  

    process_genres(blues_test,'blues',train=False,window_type='hamming',train_size=90,test_size=10) 
    process_genres(metal_test,'metal',train=False,window_type='hamming',train_size=90,test_size=10)  
    process_genres(disco_test,'disco',train=False,window_type='hamming',train_size=90,test_size=10)  
    process_genres(hiphop_test,'hiphop',train=False,window_type='hamming',train_size=90,test_size=10)  
    process_genres(classical_test,'classical',train=False,window_type='hamming',train_size=90,test_size=10)  
    process_genres(country_test,'country',train=False,window_type='hamming',train_size=90,test_size=10)
    process_genres(jazz_test,'jazz',train=False,window_type='hamming',train_size=89,test_size=10)
    process_genres(pop_test,'pop',train=False,window_type='hamming',train_size=90,test_size=10) 
    process_genres(reggae_test,'reggae',train=False,window_type='hamming',train_size=90,test_size=10)
    process_genres(rock_test,'rock',train=False,window_type='hamming',train_size=90,test_size=10)

    concant_files('blues_hamming_test.csv','metal_hamming_test.csv','disco_hamming_test.csv',
                  'hiphop_hamming_test.csv','classical_hamming_test.csv'
                  ,'country_hamming_test.csv','jazz_hamming_test.csv'
                  ,'pop_hamming_test.csv','reggae_hamming_test.csv','rock_hamming_test.csv',train=False,window_type='hamming')

    
    scaler = StandardScaler()
    #TRAIN- PREDICT HAMMING
    X_train, y_train = get_ready_training('hamming_training.csv')
    X_test, y_test = get_ready_training('hamming_training.csv')  
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.fit_transform(X_test)

    print('Hanning:')
    k = [1,3,5]
    for i in k : 
        knn_model = KNeighborsClassifier(i) 
        knn_model.fit(X_train,y_train) 
        print('Accuracy for k=',i)
        y_pred = knn_model.predict(X_test)   
        prediction(y_test=y_test,y_pred=y_pred) 

    # BARTLETT TRAIN-TEST
    X_train, y_train = get_ready_training('bartlett_training.csv')
    X_test, y_test = get_ready_training('bartlett_training.csv')  
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.fit_transform(X_test)

    print('Bartlett:')
    k = [1,3,5]
    for i in k : 
        knn_model = KNeighborsClassifier(i) 
        knn_model.fit(X_train,y_train) 
        print('Accuracy for k=',i)
        y_pred = knn_model.predict(X_test)   
        prediction(y_test=y_test,y_pred=y_pred) 

    # HANNING TRAIN-TEST 
    X_train, y_train = get_ready_training('hamming_training.csv')
    X_test, y_test = get_ready_training('hamming_training.csv')  
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.fit_transform(X_test)

    print('Hamming:')
    k = [1,3,5]
    for i in k : 
        knn_model = KNeighborsClassifier(i) 
        knn_model.fit(X_train,y_train) 
        print('Accuracy for k=',i)
        y_pred = knn_model.predict(X_test)   
        prediction(y_test=y_test,y_pred=y_pred)