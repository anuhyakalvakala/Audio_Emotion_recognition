# Audio_Emotion_recognition

Emotion Recognition from the Speech
Anuhya Kalvakala (anuhya@uwm.edu)

Questions:

We as humans have this natural instinct of recognizing the emotion of the other person who talks to you respond according to it with this when Alexa, Google assistant recognize your voice and respond to you accordingly in searching, playing songs etc.., with this we have taken the dataset with some audio files where we can predict the emotion in their speech.

Dataset:

Link: https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view

The dataset is a Ryerson audio-visual database of emotional speech and song with 24 actor’s voices with 1140 audio files in the dataset. The dataset has audio files in the format of the “.wav “. These are all the unstructured data we have for the analysis.

Analysis:

Preparation:

Soundwaves are digitized by sampling them at the discrete intervals called the sampling rate. Each sample has an amplitude wave at a particular time interval, the depth of the bit determines the details of that sample also range of the signal.

  ![image](https://user-images.githubusercontent.com/96926526/225809884-037396b9-c7d0-465f-994c-a42ca96e1e35.png)

(Reference: Rechnernetze (click here))


Talking about the signal it is measured in amplitude and time where we have three formats such as WAV (wave format audio file), mp3 (MPEG-1 Audio layer 3), WMA (Windows Media Audio). The representation one of the audio files is represented in Amplitude and time as below.

<img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225809924-61c9e2c1-29f6-4ecf-b832-5e90e3199f3e.png">

 

To depict the above audio file, we have used the python libraries such as Librosa and matplotlib.

When the audio files are loaded and checked  for their shape  and type the output will be returned as 

Type of the loaded data <class 'numpy.ndarray'>, Type of sampling output     <class 'int'>, Shape of the audio file after dividing into bits (73574,), Sampling rate 22050 ,The audio file timeseries representation in array format [ 9.0421668e-07 -1.0006800e-06 -3.1308584e-08 ...    0.0000000e+00 0.0000000e+00 0.0000000e+00] 

Sample rate is measured in Hz or KHz , where you can minimize or increase sampling as per the computational power. The signal strength is represented in the form of a spectrogram over different frequencies in the wave. It is represented in a colored heatmap.by using Librosa library and specshow we displayed the below spectrogram for above example audio file.

 <img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225809965-684dec49-b8a8-4fc2-8aaf-96255cd4b52e.png">


As shown in above spectrogram we can observe that most of the signal is at the bottom frequencies for the more clarified view we can apply logarithmic frequencies in y axis which is shown as below.

 <img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225809997-d95562e5-9070-4096-9a5b-d02fbe769469.png">


Feature Extraction:

We have extracted the features from the audio files such as MFCC(Mel-frequency cepstral coefficient), Chroma, Spectral centroid, Zero Crossing Rate, Spectral bandwidth, Spectral Roll off, RMSE

Spectral centroid: It indicates the frequency the energy of a spectrum is centered, below is the frames for the above example

 
<img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810022-527825f1-01be-4f60-992f-24d6b687bbb0.png">


Spectral Roll off: It measures the Shape of signal and sees where the signals reach base zero.

 <img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810042-c43fa57a-e1c8-4469-a6a3-e3b5cc80a693.png">


Spectral bandwidth: It measures width at one traversal of the wave on wavelength of the axis. where we measured bandwidths at 2,3,4 below

 <img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810056-ccb0c4a3-ddc4-4d32-8bcd-867b72180fb8.png">

Zero Crossing Rate: It is used to measure smoothness of the signal by calculating the no of zeros crossed.

 <img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810071-84de9f5c-e76a-4b34-8382-fbb2f9a9ae30.png">


The total  count is 15268 for the above example.


MFCC(Mel-Frequency Cepstral Coefficients): This defines the overall shape off the spectral envelope which is of 20 features. This feature is used for human voices

 <img width="404" alt="image" src="https://user-images.githubusercontent.com/96926526/225810088-94c31e57-dc78-46d8-98d1-b72421dfe2ce.png">

Chroma Feature: It is a 12-feature vector indicating energy of each pitch in the signal which measures the similarity.

 <img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810104-08b44e0e-84f9-41f1-9675-a5e4cae0575d.png">


These all the above features we have extracted from the audio files into the csv from audio files for the further analysis. 

The emotions in the features are also appended in the  with the labels where the filename values at the third place after ‘-‘ are assigned and appended into column label.
'01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'

For further analysis we have developed histograms of different columns from the excel sheet created with the features that are extracted from the audio data by calculating the means of the values calculated for the audio file and are appended into the csv file with columns as 'chroma_stft', 'rmse', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20', 'label'
       

By seeing above histograms, we can use normalization and say that the Spectral_centroid, chroma  is normally distributed whereas left skewed data are MFCC1,RSME,Zerocrossing rate and right skewed are spectral_rolloff and spectral_bandwidth.

To check outliers few, we have created some boxplots and fond in some in the column Rolloff and as they all are close to 3000 range changed them to 3000 and corrected it.

 
<img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810284-5dffb096-c976-401c-9065-a928ed967d7c.png">


To see how the data is distributed among the all the emotions we have plotted a graph.

 <img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810299-f07b3390-a537-41dd-9ea4-6febf23e5ff4.png">



The labels are encoded into numerical for the further analysis using LabelEncoder() where the labels are converted into no’s as below. 

array([7, 7, 6, 2, 2, 6, 3, 0, 0, 3, 4, 4, 1, 5, 1, 7, 7, 0, 3, 3, 0, 2, 6, 6, 2, 7, 7, 3, 0, 0, 3, 6, 2, 2, 6, 4, 4, 1, 1, 5])

Standard scalar function is used to standardize the values in the dataset for further usage.

Then further 20% of the data is used for the testing and 80% is used for training the data 



Conclusion:

Neural Networks:

Neural networks, also known as artificial neural networks (ANNs) or simulated neural networks (SNNs), are a subset of machine learning and are at the heart of deep learning algorithms. 
The model we used to train, and test is as below using neural networks.


 <img width="331" alt="image" src="https://user-images.githubusercontent.com/96926526/225810342-efcd939a-d033-43ac-b869-4f47d4b48f3c.png">



When the data was done using neural networks, we got an accuracy as 70% ,with ruc_auc score is 0.9341288816732738 we can also see that the data in validation set has more loss after 60 epoch which is caused due to less amount of test data that is been represented in training and the low data, we had we can increase our training examples to better our accuracy.

  
<img width="226" alt="image" src="https://user-images.githubusercontent.com/96926526/225810605-fff5bca3-25d5-473e-8c10-f86b103c78f7.png">
<img width="214" alt="image" src="https://user-images.githubusercontent.com/96926526/225810615-a947f719-0a7f-4d7c-93ad-87d61ff99e22.png">



Classification Report:

 
<img width="420" alt="image" src="https://user-images.githubusercontent.com/96926526/225810629-879768a4-ad23-458e-bc20-2d791c103cfd.png">


The confusion matrix for the neural networks 

 
 <img width="311" alt="image" src="https://user-images.githubusercontent.com/96926526/225810652-f64bce9d-495e-49e6-9e91-131182b27534.png">
<img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810670-4fe1ec12-83b9-4c2c-b87c-6c940e0c3e88.png">





MLP Classifier: MLP Classifier implements a multi-layer perceptron (MLP) algorithm that trains using Backpropagation. Currently, MLP Classifier supports only the Cross-Entropy loss function, which allows probability estimates by running the predict_proba method.

MLP classifier is used we got a 72%

Classification Report

 
<img width="429" alt="image" src="https://user-images.githubusercontent.com/96926526/225810684-6a85b35d-f982-4ec9-a74e-7a9d90465a25.png">



Confusion matrix for MLP Classifier

 
<img width="330" alt="image" src="https://user-images.githubusercontent.com/96926526/225810693-8c2fd461-a849-49b6-b98e-3b6a4238d653.png">

<img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810712-e5b5c381-197e-47e6-9335-c9c45ff1079d.png">


Roc Curve for each label:


From the below graph we can see the true positives are more for label neutral and fearful

 <img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810751-e0ce2d5e-decc-4f12-8bf6-1a1576bbdf08.png">


Logistic Regression: 

Logistic regression is a statistical model that uses Logistic function to model the conditional probability.

Logistic  regression is giving accuracy of  50%. 

Classification Report:

 
<img width="410" alt="image" src="https://user-images.githubusercontent.com/96926526/225810789-ba0e5dfe-643c-47dd-81dd-f8cf4e5ba0d9.png">


Confusion Matrix:

 <img width="434" alt="image" src="https://user-images.githubusercontent.com/96926526/225810805-dde65870-383c-4879-8c9d-24b0f23b78ec.png">


<img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810818-4b5318c3-22c8-48bf-8555-b9c0f33b9584.png">


Roc Curve for each label:

From the below graph we can see the true positives are more for label neutral and calm
<img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810833-d061ab53-34b2-41d6-8cbc-bcac24995794.png">

 
K-means:

The K-means clustering algorithm is used to find groups which have not been explicitly labeled in the data. This can be used to confirm business assumptions about what types of groups exist or to identify unknown groups in complex data sets

The accuracy is 13% with the k-means with 8 clusters.

Classification matrix:

<img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810850-7acce23a-f2ca-4c65-8e93-dff95e84d94e.png">

<img width="468" alt="image" src="https://user-images.githubusercontent.com/96926526/225810867-7286ee22-e2e3-49e6-9c29-e1bb48a57f54.png">


 
 
Comparing all the models  MLP classifier and neural networks works best with 72 % and 70 % of accuracy when built on 70% data and tested on 20% of the data. As we see we didn’t have the great accuracy but increasing the training data would increase the accuracy and give the excellent results to the data. We can also convert of audio data into image data using spectrogram and make those data work with CNN and further analyze on them .

