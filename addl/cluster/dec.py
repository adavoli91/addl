import sys
import os
import torch
import copy
from sklearn.cluster import KMeans

main_dir = os.path.dirname(__file__).split('cluster')[0]
sys.path.append(main_dir)

from reconstruct.autoencoder import Autoencoder
from reconstruct.autoencoder import TrainModel as TrainAutoencoder
# from utils.dataset_dataloader import *

class DEC(torch.nn.Module):
    def __init__(self, initial_centroids: torch.Tensor, encoder: torch.nn.Module) -> None:
        '''
        Class to implement DEC.

        Args:
            initial_centroids: Initial cluster centroids.
            encoder: Encoder.

        Returns: None.
        '''
        super().__init__()
        self.centroids = torch.nn.Parameter(initial_centroids)
        self.encoder = encoder

    def forward(self, x: torch.Tensor):
        '''
        Function to produce soft assignments q and target distribution p.

        Args:
            x: Encoded input.

        Returns:
            q: Soft assignment for each point to clusters.
            p: Target distribution.
        '''
        # distance from centroids
        dist_from_centroids = x.unsqueeze(dim = 1) - self.centroids
        # distribution q
        q_num = 1/(1 + torch.linalg.norm(dist_from_centroids, dim = -1)**2)
        q = q_num/q_num.sum(dim = -1).unsqueeze(dim = -1)
        # distribution p
        p_f = q.sum(dim = 0)
        p_num = q**2/p_f
        p = p_num/p_num.sum(dim = -1).unsqueeze(dim = -1)
        #
        return q, p
    
class TrainModel:
    def __init__(self, n_clust: int, dataloader_train: torch.utils.data.DataLoader, dataloader_valid: torch.utils.data.DataLoader,
                 n_feat_num: int, list_num_vals_cat: list, seed: int = 123):
        '''
        Class to train DEC. Notice: the function `train_autoencoder` should be executed prior to `train_dec`.

        Args:
            n_clust: Number of clusters.
            dataloader_train: Training dataloader.
            dataloader_valid: Validation dataloader.
            n_feat_num: Number of numerical features.
            list_num_vals_cat: List containing the number of different values for each categorical variable.
            seed: Random seed.

        Returns:
            None.
        '''
        self.n_clust = n_clust
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.n_feat_num = n_feat_num
        self.list_num_vals_cat = list_num_vals_cat
        self.seed = seed

    def train_autoencoder(self, dict_params: dict) -> None:
        '''
        Function to train the autoencoder.

        Args:
            dict_params: Dictionary containing the relevant parameters for training the autoencoder.

        Returns: None.
        '''
        dict_params_model = dict_params['model']
        #
        if hasattr(self, 'autoenc'):
            del autoenc
            del self.autoenc
        autoenc = Autoencoder(n_feat_num = self.n_feat_num, list_neurons = dict_params_model['list_neurons'], list_num_vals_cat = self.list_num_vals_cat,
                              dim_embed = dict_params_model['dim_embed'], dropout = dict_params_model['dropout'])
        trainer = TrainAutoencoder(model = autoenc, dict_params = dict_params, dataloader_train = self.dataloader_train, dataloader_valid = self.dataloader_valid)
        print('****************** Training of the autoencoder ******************')
        autoenc, _, _ = trainer.train_model()
        self.autoenc = autoenc

    def _get_initial_centroids(self) -> None:
        '''
        Function to get the initial centroids with K-means.

        Args: None.

        Returns: None.
        '''
        dataset_train = self.dataloader_train.dataset
        autoenc = self.autoenc.eval()
        # perform initial k-means
        with torch.no_grad():
            x_enc = autoenc.encoder(dataset_train.X)
        self.autoenc.train()
        k_means = KMeans(n_clusters = self.n_clust, random_state = self.seed)
        k_means.fit(x_enc)
        centroids = torch.tensor(k_means.cluster_centers_).float()
        self.k_means = k_means
        #
        if hasattr(self, 'dec'):
            del self.dec
        self.dec = DEC(initial_centroids = centroids, encoder = self.autoenc.encoder)

    def train_dec(self, dict_params: dict) -> None:
        '''
        Function to train DEC.

        Args:
            dict_params: Dictionary containing the relevant parameters for training DEC.

        Returns: None.
        '''
        # get initial centroids
        self._get_initial_centroids()
        #
        dataset_train = self.dataloader_train.dataset
        dict_params_training = dict_params['training']
        #
        dec = self.dec
        optimizer = torch.optim.Adam(params = self.dec.parameters(), lr = dict_params_training['lr'])
        #
        print()
        print('****************** Training of DEC ******************')
        for n_iter in range(dict_params_training['n_iterations']):
            X_enc = self.dec.encoder(dataset_train.X)
            #
            q, p = dec(X_enc)
            # update target distribution
            if n_iter%dict_params_training['n_iters_update_target'] == 0:
                p_target = p.detach()
            # KL divergence
            loss_kl = torch.nn.KLDivLoss(reduction =  'batchmean')
            loss = loss_kl(input = q.log(), target = p_target)
            # cluster assignment
            clust_assign = q.argmax(dim = -1)
            #
            if n_iter > 0: 
                frac_changed = (clust_assign != clust_assign_old).sum()/clust_assign.shape[0]
                print(f'Iteration {n_iter}: fraction of points that changed cluster assignment = {frac_changed*100:.1f}%')
                if (frac_changed <= dict_params_training['thresh_stop_training']) and (n_iter > 10):
                    break
            # loss backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #
            clust_assign_old = clust_assign
        return dec