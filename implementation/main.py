from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from algorithms.dwmil import DWMIL
from algorithms.lpn import LearnppNIE
from utils.check_measure import prequential_measure
from utils.data_sets import import_dataset, load_dataset

chunk_size = 1000

def run_dwmil(data_set, data_type, data, label):
	data_num = data.shape[0]
	model_dwmil = DWMIL(data_num=data_num, chunk_size=chunk_size)
	pred_dwmil = np.array([])

	print(f'Starting DWMIL on {data_set}_{data_type} ...')

	for i in tqdm(range(data_num), unit='it', ascii=True):
		pred_dwmil = np.append(pred_dwmil, model_dwmil.update(data[i], label[i]))

	print('DWMIL finished!')

	return prequential_measure(pred_dwmil, label)

def run_lpn(data_set, data_type, data, label):
	data_num = data.shape[0]
	model_lpn = LearnppNIE(data_num=data_num, chunk_size=chunk_size)
	pred_lpn = np.array([])

	print(f'Starting LearnppNIE on {data_set}_{data_type} ...')

	for i in tqdm(range(data_num), unit='it', ascii=True):
		pred_lpn = np.append(pred_lpn, model_lpn.update(data[i], label[i]))

	print('LearnppNIE finished!')

	return prequential_measure(pred_lpn, label)

def print_result(data_set, data_type, result, method):
	print(f'Results for {method} on {data_set}_{data_type} ...')
	for key, value in result.items():
		print(f'{key} = {round(value[-1], 4)}')

def write_result(data_set, data_type, result, method):
	file_name = f'{data_set}_{data_type}_{method}.txt'
	f = open('../results/' + file_name, 'w')
	f.write('metric,value' + '\n')
	for key, value in result.items():
		f.write(f'{key},{round(value[-1], 4)}' + '\n')
	f.close()

def run_dataset(data_set, data_type):
	data, label = load_dataset(f'{data_set}_{data_type}')
	
	result_dwmil = run_dwmil(data_set, data_type, data, label)
	write_result(data_set, data_type, result=result_dwmil, method='DWMIL')
	
	result_lpn = run_lpn(data_set, data_type, data, label)
	write_result(data_set, data_type, result=result_lpn, method='LearnppNIE')

data_type = 'abrupt'
data_sets = ['moving_gaussian', 'sea', 'hyper_plane', 'checkerboard', 'electricity', 'weather']

parser = ArgumentParser()
parser.add_argument('-d', '--data-set', action='store', help='name of the data set, omit for every available')
args = parser.parse_args()

if args.data_set:
	if args.data_set in data_sets:
		run_dataset(args.data_set, data_type)
		exit(0)
	else:
		print('Invalid data set name! Available data sets: ' + str(data_sets))
		exit(-1)

for data_set in data_sets:
	print('No data set name provided, running on all data sets!')
	run_dataset(data_set, data_type)
