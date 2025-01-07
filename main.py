#main
import os
import argparse
import torch
import torchvision
import torchvision.transforms as tf
from train import train_model
from model import R_Net, D_Net, R_Loss, D_Loss, R_WLoss, D_WLoss, Accuracy , Dataset


def main(args):

	torch.manual_seed(0)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
	# Uncomment this if HTTP error happened

	# new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
	# torchvision.datasets.MNIST.resources = [
	#    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
	#    for url, md5 in torchvision.datasets.MNIST.resources
	# ]
	
	train_raw_dataset = torchvision.datasets.MNIST(root='./mnist', 
									train=True, 
									download=True,
									transform=tf.Compose([tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]))
	
	valid_raw_dataset = torchvision.datasets.MNIST(root='./mnist', 
									train=False, 
									download=True, 
									transform=tf.Compose([tf.ToTensor(), tf.Normalize((0.1307,), (0.3081,))]))
	# Train and validate only on pictures of 1
	train_dataset = Dataset(train_raw_dataset, [1])
	valid_dataset = Dataset(valid_raw_dataset, [1])
	
	if args.gpu and torch.cuda.is_available():
		device = torch.device('cuda:0')
		print(f'Using GPU {torch.cuda.get_device_name()}')
		print(torch.cuda.get_device_properties(device))
	else:
		device = torch.device('cpu')
		print('Using CPU')
	
	if args.load_path:
		r_net_path = os.path.join(args.load_path, args.r_load_path)
		d_net_path = os.path.join(args.load_path, args.d_load_path)
		r_net = torch.load(r_net_path).to(device)
		print(f'Loaded R_Net from {r_net_path}')
		d_net = torch.load(d_net_path).to(device)
		print(f'Loaded D_Net from {d_net_path}')
	else:
		r_net = R_Net(in_channels = 1, std = args.std, skip = args.res, cat = args.cat).to(device)
		d_net = D_Net(in_resolution = (28, 28), in_channels = 1).to(device)
		print('Created models')
	
	# TRAINING PARAMETERS

	save_path = (args.save_path, args.r_save_path, args.d_save_path)
	optim_r_params = {'alpha' : 0.9, 'weight_decay' : 1e-9}
	optim_d_params = {'alpha' : 0.9, 'weight_decay' : 1e-9}

	model = train_model(r_net, d_net, train_dataset, valid_dataset, R_Loss, D_Loss, Accuracy,  optimizer_class=torch.optim.RMSprop,
					device=device, batch_size=args.batch_size, optim_r_params=optim_r_params, optim_d_params=optim_d_params,
					learning_rate=args.lr, rec_loss_bound=args.rec_bound,
					save_step=args.sstep, num_workers=args.nw, save_path=save_path, lambd=args.lambd)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', action='store_false', help='Turns off gpu training')
	parser.add_argument('--save_path', '-sp', type=str, default='.', help='Path to a folder where metrics and models will be saved')
	parser.add_argument('--d_save_path', '-dsp', type=str, default='d_net.pth', help='Name of .pth file for d_net to be saved')
	parser.add_argument('--r_save_path', '-rsp', type=str, default='r_net.pth', help='Name of .pth file for r_net to be saved')
	parser.add_argument('--load_path', '-lp', default=None, help='Path to a folder from which models will be loaded')
	parser.add_argument('--d_load_path', '-dlp', type=str, default='d_net.pth', help='Name of .pth file for d_net to be loaded')
	parser.add_argument('--r_load_path', '-rlp', type=str, default='r_net.pth', help='Name of .pth file for r_net to be loaded')
	parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loader')
	parser.add_argument('--rec_bound', type=float, default=0.1, help='Upper bound of reconstruction loss')
	parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate')
	parser.add_argument('--nw', type=int, default=1, help='num_workers for DataLoader')
	parser.add_argument('--sstep', type=int, default=5, help='Step in epochs for saving models')
	parser.add_argument('--std', type=float, default=0.155, help='Standart deviation for noise in R_Net')
	parser.add_argument('--lambd', type=float, default=0.2, help='Lambda parameter for LR loss')
	parser.add_argument('--cat', action='store_true', help='Turns on skip connections with concatenation')
	parser.add_argument('--res', action='store_true', help='Turns on residual connections')
	args = parser.parse_args()

	main(args)
