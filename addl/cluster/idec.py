import numpy as np
import torch
import copy
import os
import sys
from sklearn.cluster import KMeans
from typing import Tuple
#
main_dir = os.path.dirname(__file__).split('cluster')[0]
sys.path.append(main_dir)
sys.path.append(os.path.join(main_dir, 'cluster'))
#
from reconstruct.autoencoder import Autoencoder
from reconstruct.autoencoder import TrainModel as TrainAutoencoder
from dec import DEC
    
class IDEC(torch.nn.Module):
    def __init__(self, initial_centroids: torch.Tensor, autoencoder: torch.nn.Module):
        '''
        Class to implement IDEC.

        Args:
            initial_centroids: Initial cluster centroids.
            autoencoder: Pre-trained autoencoder.

        Returns: None.
        '''
        super().__init__()
        self.centroids = torch.nn.Parameter(initial_centroids)
        self.autoencoder = autoencoder
        self.dec = DEC(initial_centroids = initial_centroids, encoder = autoencoder.encoder)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list, torch.Tensor, torch.Tensor]:
        '''
        Function to produce soft assignments q and target distribution p.

        Args:
            x: Input tensor: all the numerical features are assumed to be in the first `n_feat_num` columns, and categorical ones are
               assumed to be expressed as indices of some encoder (e.g., OrdinalEncoder).

        Returns:
            x_hat_num: Reconstructed input corresponding to numerical variables.
            x_hat_cat: Reconstructed input corresponding to categorical variables: each element of the list is the decoded representation of a single
                       categorical feature.
            q: Soft assignment for each point to clusters.
            p: Target distribution.
        '''
        autoencoder = self.autoencoder
        
        # autoencder
        x_enc = autoencoder.encoder(x)
        x_hat_num, x_hat_cat = autoencoder(x)
        
        # DEC
        q, p = self.dec(x_enc)

        return x_hat_num, x_hat_cat, q, p

class TrainModel:
    def __init__(self, n_clust: int, dict_params: dict, dataloader_train: torch.utils.data.DataLoader, dataloader_valid: torch.utils.data.DataLoader,
                 n_feat_num: int, list_num_vals_cat: list, seed: int = 123):
        '''
        Class to train IDEC. Notice: the function `train_autoencoder` should be executed prior to `train_idec`.

        Args:
            n_clust: Number of clusters.
            dict_params: Dictionary containing the relevant parameters for training.
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
        self.dict_params_training = dict_params['training']
        self.path_artifacts = dict_params['training']['path_artifacts']

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
        if hasattr(self, 'idec'):
            del self.idec
        self.idec = IDEC(initial_centroids = centroids, autoencoder = self.autoenc)

    def _loss_num(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Function to compute the reconstruction loss for numerical variables.

        Args:
            input: Input (reconstructed) tensor.
            target: Target (ground truth) tensor.

        Returns:
            loss: Value of the loss function.
        '''
        loss_func = torch.nn.MSELoss()
        loss = loss_func(input = input, target = target)
        return loss

    def _loss_cat(self, input: list, target: torch.Tensor) -> torch.Tensor:
        '''
        Function to compute the reconstruction loss for categorical variables.

        Args:
            input: List of input (reconstructed) tensors.
            target: Target (ground truth) tensor.

        Returns:
            loss: Value of the loss function.
        '''
        loss_tot = 0
        for i in range(len(input)):
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(input = input[i], target = target[:, i].long())
            loss_tot = loss_tot + loss
        return loss_tot/len(input)

    def loss_ae(self, x_hat_num: torch.Tensor, x_hat_cat, target: torch.Tensor) -> torch.Tensor:
        '''
        Function to compute the reconstruction loss for both numerical and categorical variables.

        Args:
            x_hat_num: Reconstructed input corresponding to numerical variables.
            x_hat_cat: List of reconstructed inputs corresponding to categorical variables.
            target: Target (ground truth) tensor.

        Returns:
            loss: Value of the loss function.
        '''
        rel_weight_losses = self.dict_params_training['rel_weight_losses']
        loss = 0

        # loss from numerical features
        if x_hat_num.shape[1] > 0:
            loss = loss + rel_weight_losses*self._loss_num(input = x_hat_num, target = target[:, :self.n_feat_num])
            
        # loss from categorical features
        if len(x_hat_cat) > 0:
            loss = loss + (1 - rel_weight_losses)*self._loss_cat(input = x_hat_cat, target = target[:, self.n_feat_num:])

        return loss

    def _model_on_batch(self, batch: tuple, training: bool) -> float:
        '''
        Function to perform training on a single batch of data.
        
        Args:
            batch: Batch of data to use for training/evaluation.
            training: Whether to perform training (if not, evaluation is understood).
            loss_epoch: Loss of the current epoch.
            
        Returns:
            loss: Value of the loss function.
        '''
        idec = self.idec
        device = next(idec.parameters()).device
        gamma = self.dict_params_training['weight_loss_clust']
        #
        if training == True:
            self.optimizer.zero_grad()
        #
        X = batch
        X = X.to(device)
        X_hat_num, X_hat_cat, q, p = idec(X)
        p = p.detach()
        # reconstruction loss
        loss_ae = self.loss_ae(x_hat_num = X_hat_num, x_hat_cat = X_hat_cat, target = X)
        # clustering loss
        loss_clust_func = torch.nn.KLDivLoss(reduction = 'batchmean')
        loss_clust = loss_clust_func(input = q.log(), target = p)
        # total loss
        loss = loss_ae + gamma*loss_clust
        #
        if training == True:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(idec.parameters(), 1.0)
            self.optimizer.step()
        #
        return loss.item()
    
    def _train(self) -> float:
        '''
        Function to train the model on a single epoch.
        
        Args: None.
            
        Returns:
            loss: Value of the training loss function per batch.
        '''
        self.idec.train()
        loss_epoch = 0
        for batch in self.dataloader_train:
            loss_epoch += self._model_on_batch(batch = batch, training = True)
        return loss_epoch/len(self.dataloader_train)

    def _eval(self) -> float:
        '''
        Function to evaluate the model on the validation set on a single epoch.
        
        Args: None.
            
        Returns:
            loss: Value of the validation loss function per batch.
        '''
        self.idec.eval()
        loss_epoch = 0
        with torch.no_grad():
            for batch in self.dataloader_valid:
                loss_epoch += self._model_on_batch(batch = batch, training = False)
        return loss_epoch/len(self.dataloader_valid)

    def train_idec(self, dict_params: dict) -> None:
        '''
        Function to train IDEC.
        
        Args:
            dict_params: Dictionary containing the relevant parameters for training IDEC.
            
        Returns:
            idec: Trained IDEC.
            list_loss_train: List of training loss function across the epochs.
            list_loss_valid: List of validation loss function across the epochs.
        '''
        print()
        print('****************** Training of IDEC ******************')
        # get initial centroids
        self._get_initial_centroids()
        #
        dataset_train = self.dataloader_train.dataset
        dict_params_training = dict_params['training']
        #
        idec = self.idec
        self.optimizer = torch.optim.Adam(params = idec.parameters(), lr = dict_params_training['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer, factor = 0.5)
        n_epochs = dict_params_training['n_epochs']
        list_loss_train, list_loss_valid = [], []
        path_artifacts = self.path_artifacts
        #
        counter_patience = 0
        for epoch in range(1, n_epochs + 1):
            loss_train = self._train()
            loss_valid = self._eval()
            #
            if (len(list_loss_valid) > 0) and (loss_valid >= np.min(list_loss_valid)*(1 - dict_params_training['min_delta_loss_perc'])):
                counter_patience += 1
            if (len(list_loss_valid) == 0) or ((len(list_loss_valid) > 0) and (loss_valid < np.min(list_loss_valid))):
                counter_patience = 0
                best_weights = copy.deepcopy(idec.state_dict())
                if path_artifacts is not None:
                    torch.save(idec.state_dict(), path_artifacts)
            if counter_patience >= dict_params_training['patience']:
                print(f'Training stopped at epoch {epoch}. Restoring weights from epoch {np.argmin(list_loss_valid) + 1}.')
                break
            #
            print(f'Epoch {epoch}: training loss = {loss_train:.4f}, validation loss = {loss_valid:.4f}, learning rate = {self.optimizer.param_groups[0]["lr"]}, patience counter = {counter_patience}.')
            self.scheduler.step(loss_valid)
            #
            list_loss_train.append(loss_train)
            list_loss_valid.append(loss_valid)
        #
        if path_artifacts is not None:
            idec.load_state_dict(torch.load(path_artifacts))
        idec.load_state_dict(best_weights)
        return idec, list_loss_train, list_loss_valid