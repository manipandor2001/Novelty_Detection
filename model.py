#module feature selection
import torch
import torch.nn.functional as F
import numpy as np 
from sklearn.metrics import accuracy_score , roc_curve, f1_score



class Dataset(torch.utils.data.Dataset):

	def __init__(self, dataset: torch.utils.data.Dataset, labels: list):
		self.dataset = dataset
		self.labels = labels
		self.indexes = self._extract_indexes()

	def _extract_indexes(self):

		indexes = []

		for label in self.labels:
			for i, sample in enumerate(self.dataset):
				if sample[1] == label:
					indexes.append(i)
		print(f'Created dataset of length {len(indexes)} for labels: {self.labels}')
		return indexes

	def __len__(self):
		return len(self.indexes)
	
	def __getitem__(self, idx: int):
		return self.dataset[self.indexes[idx]]

class ModuleList(torch.nn.ModuleList):

	def forward(self, x:torch.Tensor, res_connections:list=[], cat:bool=False):
		connections = []
		for i, module in enumerate(self):
			if i % 3 == 0 and not bool(res_connections):
				x = module(x)
				connections.append(x)
			elif i % 3 == 0 and bool(res_connections):
				x = module(torch.cat((x, res_connections[i//3]), dim = 1)) if cat else module(x + res_connections[i//3])
			else:
				x = module(x)
		return x, connections

class R_Net(torch.nn.Module):

	def __init__(self, activation = torch.nn.LeakyReLU, in_channels:int = 1, n_channels:int = 64,
					kernel_size:int = 5, std:float = 1.0, skip:bool = False, cat:bool = False):

		super(R_Net, self).__init__()

		self.activation = activation
		self.in_channels = in_channels
		self.n_c = n_channels
		self.k_size = kernel_size
		self.std = std
		self.cat = cat
		self.skip = skip if self.cat is False else False

		if self.skip:
			print('Turn on res connections')
		elif self.cat:
			print('Turn on concat skip')
		else:
			print('No skip connections')

		self.Encoder = ModuleList([torch.nn.Conv2d(self.in_channels, self.n_c, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c),
											self.activation(),
											torch.nn.Conv2d(self.n_c, self.n_c*2, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c*2),
											self.activation(),
											torch.nn.Conv2d(self.n_c*2, self.n_c*4, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c*4),
											self.activation(),
											torch.nn.Conv2d(self.n_c*4, self.n_c*8, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.n_c*8),
											self.activation()])

		self.n_c = self.n_c * 2 if self.cat else self.n_c
		
		self.bfw = FeatureSelectionModule(73728 , torch.nn.Tanh())
		self.Decoder = ModuleList([torch.nn.ConvTranspose2d(self.n_c*8, n_channels*4, self.k_size, bias = False),
											torch.nn.BatchNorm2d(n_channels*4),
											self.activation(),
											torch.nn.ConvTranspose2d(self.n_c*4, n_channels*2, self.k_size, bias = False),
											torch.nn.BatchNorm2d(n_channels*2),
											self.activation(),
											torch.nn.ConvTranspose2d(self.n_c*2, n_channels, self.k_size, bias = False),
											torch.nn.BatchNorm2d(n_channels),
											self.activation(),
											torch.nn.ConvTranspose2d(self.n_c, self.in_channels, self.k_size, bias = False),
											torch.nn.BatchNorm2d(self.in_channels),
											self.activation()])

	def forward(self, x:torch.Tensor, noise:bool = True):

		x_hat = self.add_noise(x) if noise else x
		z, res_connections = self.Encoder.forward(x_hat)
		z_hat = self.bfw(z)
		res_connections.reverse()

		if self.skip or self.cat:
			x_out, _ = self.Decoder.forward(z, res_connections, self.cat)
		else:
			x_out, _ = self.Decoder.forward(z)

		return x_out

	def add_noise(self, x):

		noise = torch.randn_like(x) * self.std
		x_hat = x + noise

		return x_hat

class D_Net(torch.nn.Module):

	def __init__(self, in_resolution:tuple, activation = torch.nn.LeakyReLU, in_channels:int = 1, n_channels:int = 64, kernel_size:int = 3):

		super(D_Net, self).__init__()

		self.activation = activation
		self.in_resolution = in_resolution
		self.in_channels = in_channels
		self.n_c = n_channels
		self.k_size = kernel_size

		self.cnn = torch.nn.Sequential(torch.nn.Conv2d(self.in_channels, self.n_c, self.k_size, bias = False),
										torch.nn.BatchNorm2d(self.n_c),
										self.activation(),
										torch.nn.Conv2d(self.n_c, self.n_c*2, self.k_size, bias = False),
										torch.nn.BatchNorm2d(self.n_c*2),
										self.activation(),
										torch.nn.Conv2d(self.n_c*2, self.n_c*4, self.k_size, bias = False),
										torch.nn.BatchNorm2d(self.n_c*4),
										self.activation(),
										torch.nn.Conv2d(self.n_c*4, self.n_c*8, self.k_size, bias = False),
										torch.nn.BatchNorm2d(self.n_c*8),
										self.activation())

		# Compute output dimension after conv part of D network

		self.out_dim = self._compute_out_dim()

		self.fc = torch.nn.Linear(self.out_dim, 1)

	def _compute_out_dim(self):
		
		test_x = torch.Tensor(1, self.in_channels, self.in_resolution[0], self.in_resolution[1])
		for p in self.cnn.parameters():
			p.requires_grad = False
		test_x = self.cnn(test_x)
		out_dim = torch.prod(torch.tensor(test_x.shape[1:])).item()
		for p in self.cnn.parameters():
			p.requires_grad = True

		return out_dim

	def forward(self, x:torch.Tensor):

		x = self.cnn(x)

		x = torch.flatten(x, start_dim = 1)

		out = self.fc(x)

		return out

def R_Loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, lambd: float) -> dict:

	pred = d_net(x_fake)
	y = torch.ones_like(pred)

	rec_loss = F.mse_loss(x_fake, x_real)
	gen_loss = F.binary_cross_entropy_with_logits(pred, y) # generator loss

	L_r = gen_loss + lambd * rec_loss

	return {'rec_loss' : rec_loss, 'gen_loss' : gen_loss, 'L_r' : L_r}

def Accuracy(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor) -> tuple:
    # Predict for real and fake samples
    pred_real = d_net(x_real)
    pred_fake = d_net(x_fake.detach())

    # True labels for real and fake samples
    y_real = torch.ones_like(pred_real)
    y_fake = torch.zeros_like(pred_fake)

    # Combine labels and predictions
    y = torch.cat((y_real, y_fake), dim=0).cpu().detach().numpy()
    pred = torch.cat((pred_real, pred_fake), dim=0)
    pred = torch.sigmoid(pred).cpu().detach().numpy()  # Apply sigmoid activation

    # Explore thresholds from 0 to 1
    thresholds = np.linspace(0, 1, 100)
    best_accuracy = 0
    best_threshold = 0
    best_f1 = 0

    for threshold in thresholds:
        pred_binary = (pred > threshold).astype(int)  # Apply threshold to predictions
        acc = accuracy_score(y, pred_binary, normalize = True)
        f1 = f1_score(y, pred_binary)

        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold
            best_f1 = f1

    # Return the threshold, accuracy, and F1 score at the optimal threshold
    return best_accuracy
def D_Loss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:

	pred_real = d_net(x_real)
	pred_fake = d_net(x_fake.detach())
	
	y_real = torch.ones_like(pred_real)
	y_fake = torch.zeros_like(pred_fake)

	real_loss = F.binary_cross_entropy_with_logits(pred_real, y_real)
	fake_loss = F.binary_cross_entropy_with_logits(pred_fake, y_fake)

	return real_loss + fake_loss


# Wasserstein GAN loss (https://arxiv.org/abs/1701.07875)

def R_WLoss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor, lambd: float) -> dict:

	pred = torch.sigmoid(d_net(x_fake))

	rec_loss = F.mse_loss(x_fake, x_real)
	gen_loss = -torch.mean(pred) # Wasserstein G loss: - E[ D(G(x)) ]

	L_r = gen_loss + lambd * rec_loss

	return {'rec_loss' : rec_loss, 'gen_loss' : gen_loss, 'L_r' : L_r}

def D_WLoss(d_net: torch.nn.Module, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:

	pred_real = torch.sigmoid(d_net(x_real))
	pred_fake = torch.sigmoid(d_net(x_fake.detach()))
	
	dis_loss = -torch.mean(pred_real) + torch.mean(pred_fake) # Wasserstein D loss: -E[D(x_real)] + E[D(x_fake)]

	return dis_loss

class FeatureSelectionModule(torch.nn.Module):
    def __init__(self, input_l,  activation):
        """
        Module for Batch-Wise Attenuation and Feature Mask Normalization.
        """
        super(FeatureSelectionModule, self).__init__()

        self.input_l = input_l
        self.hidden_layer = input_l // 2
        self.activation =  activation



        self.fc1 = torch.nn.Linear(input_l , self.hidden_layer)



        self.fc2 = torch.nn.Linear(self.hidden_layer, input_l)
    def forward(self, z):
      z_flat_1 = z.view(z.size(0), -1)  # Shape: (B, D*H*W)

      z_flat = F.relu(self.fc1(z_flat_1))
      z_flat = self.fc2(z_flat)
      z_flat = self.activation(z_flat)

      z_avg = torch.mean(z_flat, dim=0)  # Shape: (D,)

      m = F.softmax(z_avg, dim=0)  # Shape: (D,)

      # Perform operations
      m_expanded = m.unsqueeze(0).expand(z_flat.shape[0], -1)
      z_hardman = z_flat_1 * m_expanded
      z_reshape = z_hardman.view(z.shape)
 
      # Explicitly detach temporary tensors after use
     # del z_flat_1, z_flat, z_avg, m_expanded, z_hardman
     # torch.cuda.empty_cache()  # Optional: Frees up GPU memory

      return z_reshape
          
