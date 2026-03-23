import torch
import copy
from sklearn.cluster import KMeans
import torch
import copy
from typing import Tuple

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
        x_hat_num, x_hat_cat = autoencoder(x)
        
        # DEC
        q, p = self.dec(x)

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
            del self.autoenc
        autoenc = Autoencoder(n_feat_num = self.n_feat_num, list_neurons = dict_params_model['list_neurons'], list_num_vals_cat = self.list_num_vals_cat,
                              dim_embed = dict_params_model['dim_embed'], dropout = dict_params_model['dropout'])
        trainer = TrainAutoencoder(model = autoenc, dict_params = dict_params, dataloader_train = self.dataloader_train, dataloader_valid = self.dataloader_valid)
        print('****************** Training of the autoencoder ******************')
        autoenc, _, _ = trainer.train_model()
        self.autoenc = autoenc
        self.autoenc_original_weights = copy.deepcopy(autoenc.state_dict())

    def _get_initial_centroids(self) -> None:
        '''
        Function to get the initial centroids with K-means.

        Args: None.

        Returns: None.
        '''
        dataset_train = self.dataloader_train.dataset.X
        # reshuffle data
        dataset_train = dataset_train[torch.randperm(dataset_train.shape[0])]
        #
        autoenc = self.autoenc.eval()
        # perform initial k-means
        with torch.no_grad():
            x_enc = autoenc.encoder(dataset_train)
        self.autoenc.train()
        k_means = KMeans(n_clusters = self.n_clust, random_state = self.seed)
        k_means.fit(x_enc)
        centroids = torch.tensor(k_means.cluster_centers_).float()
        self.x_enc_original = copy.deepcopy(x_enc)
        self.k_means = k_means
        #
        if hasattr(self, 'dec'):
            del self.dec
            # reset autoencoder weights
            self.autoenc.load_state_dict(self.autoenc_original_weights)
        #
        self.dataset_train = dataset_train
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
    
    def train_model(self, dict_params: dict) -> torch.nn.Module:
        '''
        Function to train IDEC.

        Args:
            dict_params: Dictionary containing the relevant parameters for training IDEC.

        Returns:
            dec: Trained IDEC.
        '''
        self._get_initial_centroids()
        #
        idec = self.idec
        device = next(idec.parameters()).device
        dict_params_training = dict_params['training']
        #
        device = next(idec.parameters()).device
        gamma = dict_params_training['weight_loss_clust']
        batch_size = dict_params_training['batch_size']
        optimizer = torch.optim.Adam(params = idec.parameters(), lr = dict_params_training['lr'])
        #
        print()
        print('****************** Training of IDEC ******************')
        for n_iter in range(dict_params_training['n_iterations']):
            X = self.dataset_train
            # update target distribution
            if n_iter%dict_params_training['n_iters_update_target'] == 0:
                idec.eval()
                with torch.no_grad():
                    _, _, q, p = idec(X)
                idec.train()
                p_target = p.detach()
                #
                clust_assign = q.argmax(dim = -1)
                # check the fraction of points which changed cluster assignment
                if n_iter > 0:
                    frac_changed = (clust_assign != clust_assign_old).sum()/clust_assign.shape[0]
                    print(f'Iteration {n_iter}: fraction of points that changed cluster assignment = {frac_changed*100:.1f}%')
                    if (frac_changed <= dict_params_training['thresh_stop_training']) and (n_iter > 10):
                        break
                #
                clust_assign_old = clust_assign
            #
            idx = 0
            while idx < X.shape[0]:
                batch = X[idx: idx + batch_size].to(device)
                X_hat_num, X_hat_cat, q, _ = idec(batch)
                # reconstruction loss
                loss_ae = self.loss_ae(x_hat_num = X_hat_num, x_hat_cat = X_hat_cat, target = batch)
                # clustering loss
                loss_clust_func = torch.nn.KLDivLoss(reduction = 'batchmean')
                loss_clust = loss_clust_func(input = q.log(), target = p_target[idx: idx + batch_size])
                # total loss
                loss = loss_ae + gamma*loss_clust
                # loss backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #
                idx += batch_size
        return idec