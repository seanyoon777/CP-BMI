import numpy as np 
import scipy.io as sio 
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_all_data (data_path): 

    big_X_train, big_y_train, big_X_test, big_y_test = [None]*9, [None]*9, [None]*9, [None]*9
    for subject in range (0,9):
        path = data_path+'/'
        big_X_train[subject], big_y_train[subject] = get_data(subject+1, True ,path)
        big_X_test[subject], big_y_test[subject] = get_data(subject+1, False ,path)
    
    return big_X_train, big_y_train, big_X_test, big_y_test

def get_data(subject,training,path, highpass = False, include_artifacts=False):
	'''	Loads the dataset 2a of the BCI Competition IV
	available on http://bnci-horizon-2020.eu/database/data-sets

	Keyword arguments:
	subject -- number of subject in [1, .. ,9]
	training -- if True, load training data
				if False, load testing data
	
	Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
			class_return 	numpy matrix 	size = NO_valid_trial
	'''
	NO_channels = 22
	NO_tests = 6*48 	
	Window_Length = 7*250 

	class_return = np.zeros(NO_tests)
	data_return = np.zeros((NO_tests,NO_channels,Window_Length))

	NO_valid_trial = 0
	if training:
		a = sio.loadmat(path+'A0'+str(subject)+'T.mat')
	else:
		a = sio.loadmat(path+'A0'+str(subject)+'E.mat')
	a_data = a['data']
	for ii in range(0,a_data.size):
		a_data1 = a_data[0,ii]
		a_data2= [a_data1[0,0]]
		a_data3= a_data2[0]
		a_X 		= a_data3[0]
		a_trial 	= a_data3[1]
		a_y 		= a_data3[2]
		a_fs 		= a_data3[3]
		a_classes 	= a_data3[4]
		a_artifacts = a_data3[5]
		a_gender 	= a_data3[6]
		a_age 		= a_data3[7]

		if not include_artifacts: 
			for trial in range(0,a_trial.size):
					if(a_artifacts[trial]==0):
						data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
						class_return[NO_valid_trial] = int(a_y[trial])
						NO_valid_trial +=1
	
		if include_artifacts: 
			for trial in range(0, a_trial.size): 
				if(a_artifacts[trial] == 1): 
					class_return[NO_valid_trial] = 0
				else: 
					class_return[NO_valid_trial] = int(a_y[trial])
				data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
				NO_valid_trial += 1

	return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]

# def get_data(subject,training,path, highpass = False):
# 	'''	Loads the dataset 2a of the BCI Competition IV
# 	available on http://bnci-horizon-2020.eu/database/data-sets

# 	Keyword arguments:
# 	subject -- number of subject in [1, .. ,9]
# 	training -- if True, load training data
# 				if False, load testing data
	
# 	Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
# 			class_return 	numpy matrix 	size = NO_valid_trial
# 	'''
# 	NO_channels = 22
# 	NO_tests = 6*48 	
# 	Window_Length = 7*250 

# 	class_return = np.zeros(NO_tests)
# 	data_return = np.zeros((NO_tests,NO_channels,Window_Length))

# 	NO_valid_trial = 0
# 	if training:
# 		a = sio.loadmat(path+'A0'+str(subject)+'T.mat')
# 	else:
# 		a = sio.loadmat(path+'A0'+str(subject)+'E.mat')
# 	a_data = a['data']
# 	for ii in range(0,a_data.size):
# 		a_data1 = a_data[0,ii]
# 		a_data2= [a_data1[0,0]]
# 		a_data3= a_data2[0]
# 		a_X 		= a_data3[0]
# 		a_trial 	= a_data3[1]
# 		a_y 		= a_data3[2]
# 		a_fs 		= a_data3[3]
# 		a_classes 	= a_data3[4]
# 		a_artifacts = a_data3[5]
# 		a_gender 	= a_data3[6]
# 		a_age 		= a_data3[7]

# 		for trial in range(0,a_trial.size):
# 			if(a_artifacts[trial]==0):
# 				data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
# 				class_return[NO_valid_trial] = int(a_y[trial])
# 				NO_valid_trial +=1


# 	return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]


def prepare_features(X, y, include_artifacts=False):
	fs = 250 
	t1 = int(1.5*fs)
	t2 = int(6*fs)
	T = t2-t1

	N_tr,N_ch,_ =X.shape 
	X = X[:,:,t1:t2].reshape(N_tr,1,N_ch,T)
	if not include_artifacts: 
		y = (y-1).astype(int)
	else: 
		y = y.astype(int)
	y = to_categorical(y)

	return X, y