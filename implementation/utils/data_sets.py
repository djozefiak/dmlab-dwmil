import numpy as np

def import_dataset(name, data_name='data', class_name='class'):
	path_data = f'../data_sets/{name}_{data_name}.csv'
	path_class = f'../data_sets/{name}_{class_name}.csv'
	result_data = np.genfromtxt(path_data, delimiter=',')
	result_class = np.genfromtxt(path_class, delimiter=',')
	return result_data, result_class

def load_dataset(name):
	loaded_data = np.load(f'../data_sets/{name}.npz')
	result_data = loaded_data['data']
	result_class = loaded_data['label']
	return result_data, result_class
