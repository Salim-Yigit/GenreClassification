# Musical gnere classification with KNN 
GTZAN Dataset was used. You can download from [Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).    
Kod folder includes features of songs, this features extracted with FFT and STFT.  
In code file you can see FFT fucntion, this function was written by me. It is too slow numpy fft function.  
You can use numpy.fft.fft() function if you want.  
Also, after you downloaded GTZAN Dataset, you should create Data folder and genres original folder inside of the Data folder.  
Dataset was downloaded as this file structure.  I didn't push Data folder because it is too large and I couldn't upload it.    
  
  


FOLDER STRUCTURE in the below :   
![image](https://github.com/Salim-Yigit/GenreClassification/assets/94362868/687e042b-b1b4-45af-b299-75a77c19f0c6)

Finally, you should move Kod folder then you can run code with this command.==> "python v2.py"

## FEATURE EXTRACTOIN 
For feature extraction two functions were implemented. These are FFT and STFT functions.  
FFT function is fast Fourier Transform which does same numpy.fft.ftt() function, but it is slower.   
STFT function Short Time Fourier Transform.  
It takes three argument:signal,window_size,window_type,overlap_ratio.    
window_size = 1024     
window_type = hamming    
overlap_ratio = 0.2 by default.    
You can change if you want.  
Feature extraction performing for each genre, then informations are saved csv files.  
For example, blues_hamming_train.csv. Then all files were merged and got one files which named {window_type}_train.csv.(hamming_train.csv)  
This process is repeated for test data. Also in this process three window_type is used.  
These are Hanning,Bartlett and Hamming.  
In the end, we have 3 train files and 3 test files. 
### FEATURES  
- Frequency power 
- Amplitude of frequency 
- Weighted frequency average by amplitude.   are used as features.   
Then to distinguish genres, statistical functions are used. These are mean,standard variation and median.   
Therefore for every music 9 features are extracted.  



## MODEL  
This project was homework for Digital Signal Processing lecture.In homework,instructors said just use KNN.  
Therefore KNN algorithm just used but any classification algorithms can be used.  
n_neighbors parameter were set 1,3,5 sequantially and scores were listed.   
You can see in below.  

![image](https://github.com/Salim-Yigit/GenreClassification/assets/94362868/cbeb9a0a-0d3f-4be3-8d1f-33c621415d2f) 

## REQUIREMENTS    
matplotlib==3.6.0    
numpy==1.22.3    
pandas==1.5.1    
scikit-learn==1.2.0    
scipy==1.10.0  
sklearn==0.0.post1  

The versions of libraries which are used in the project. 
You can setup your enviroment with this versions,if code doesn't work on your enviroment.  
There is a requirements.txt file to make your job easier.  

