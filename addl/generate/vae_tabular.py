import numpy as np
import torch
import copy
from typing import Tuple

class Encoder(torch.nn.Module):
    def __init__(self, n_feat_num: int = 0, list_neurons: list = [], list_num_vals_cat: list = [], dim_embed: int = 10,
                 dropout: float = 0) -> None:
        '''
        Class to implement the VAE encoder.

        Args:
            n_feat_num: Number of numerical features.
            list_neurons: List of numbers of neurons used during encoding.
            list_num_vals_cat: List containing, for each categorical variables, the number of categories.
            dim_embed: Embedding dimension for categorical variables.
            dropout: Dropout probability.

        Returns: None.
        '''
        super().__init__()
        #
        if (n_feat_num == 0) and (len(list_num_vals_cat) == 0):
            raise(Exception('At least one numerical or categorical feature is needed.'))
        #
        self.n_feat_num = n_feat_num
        self.dim_embed = dim_embed
        ## embedding layers for categorical variables
        n_feat_cat = 0
        if len(list_num_vals_cat) > 0:
            list_layers_embed = []
            for n_cat in list_num_vals_cat:
                list_layers_embed.append(torch.nn.Embedding(num_embeddings = n_cat + 1, embedding_dim = dim_embed))
                n_feat_cat += dim_embed
            #
            self.list_layers_embed = torch.nn.ModuleList(list_layers_embed)

        ## layer for input normalization
        self.layer_bn_input = torch.nn.BatchNorm1d(num_features = n_feat_num + n_feat_cat)

        ## encoder layers
        list_neurons = copy.deepcopy(list_neurons)
        list_neurons[-1] *= 2
        list_neurons = [n_feat_num + n_feat_cat] + list_neurons
        list_layers_enc, list_layers_bn = [], []
        for i, (neur_in, neur_out) in enumerate(list(zip(list_neurons[:-1], list_neurons[1:]))):
            list_layers_enc.append(torch.nn.Linear(in_features = neur_in, out_features = neur_out))
            if i < len(list_neurons) - 2:
                list_layers_bn.append(torch.nn.BatchNorm1d(num_features = neur_out))
        self.list_layers_enc = torch.nn.ModuleList(list_layers_enc)
        self.list_layers_bn = torch.nn.ModuleList(list_layers_bn)

        ## relu
        self.relu = torch.nn.ReLU()

        ## dropout
        self.dropout = torch.nn.Dropout(p = dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Function to encode input data.

        Args:
            x: Input tensor: all the categorical variables are assumed to be after all the numerical ones.

        Returns:
            mu: Mean values of features in latent space.
            log_sigma: Variances in latent space.
        '''
        x_num = x[:, :self.n_feat_num]

        ## embed categorical variables
        if hasattr(self, 'list_layers_embed'):
            x_cat = x[:, -len(self.list_layers_embed):]
            #
            embedding = []
            for i in range(x_cat.shape[1]):
                embedding.append(self.list_layers_embed[i](x_cat[:, i].int()))
            #
            embedding = torch.cat(embedding, dim = -1)
            # concatenate to numerical features
            x_in = torch.cat([x_num, embedding], axis = 1)
        else:
            x_in = x_num

        ## normalize features
        x_enc = self.layer_bn_input(x_in)

        ## encode input
        for i in range(len(self.list_layers_bn)):
            x_enc = self.list_layers_enc[i](x_enc)
            # batch normalization
            x_enc = self.list_layers_bn[i](x_enc)
            # activation function
            x_enc = self.relu(x_enc)
            # dropout
            x_enc = self.dropout(x_enc)
        #
        x_enc = self.list_layers_enc[-1](x_enc)
        # split in mean and variance
        mu, log_sigma = x_enc.chunk(chunks = 2, dim = -1)

        return mu, log_sigma

class Decoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, dropout: float = 0):
        '''
        Class to implement the VAE decoder.

        Args:
            encoder: VAE encoder.
            dropout: Dropout probability.

        Returns: None.
        '''
        super().__init__()
        self.dim_embed = encoder.dim_embed
        self.n_feat_num = encoder.n_feat_num
        
        ## decoder layers
        list_layers_dec, list_layers_bn = [], []
        for i, layer in enumerate(reversed(encoder.list_layers_enc)):
            if i == 0:
                list_layers_dec.append(torch.nn.Linear(in_features = int(layer.out_features/2), out_features = layer.in_features))
            else:
                list_layers_dec.append(torch.nn.Linear(in_features = layer.out_features, out_features = layer.in_features))
            #
            if i < len(encoder.list_layers_enc) - 1:
                list_layers_bn.append(torch.nn.BatchNorm1d(num_features = layer.in_features))
        self.list_layers_dec = torch.nn.ModuleList(list_layers_dec)
        self.list_layers_bn = torch.nn.ModuleList(list_layers_bn)

        ## layers to 'invert' embeddings
        if hasattr(encoder, 'list_layers_embed'):
            list_layer_embed_inv = []
            for layer in encoder.list_layers_embed:
                list_layer_embed_inv.append(torch.nn.Linear(in_features = layer.embedding_dim,
                                                            out_features = layer.num_embeddings))
            self.list_layer_embed_inv = torch.nn.ModuleList(list_layer_embed_inv)

        ## relu
        self.relu = torch.nn.ReLU()

        ## dropout
        self.dropout = torch.nn.Dropout(p = dropout)

    def forward(self, sampled_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Function to decode data sampled in the latent space.

        Args:
            sampled_points: Tensor with points sampled in the latent space.

        Returns:
            x_num: Portion of decoded data corresponding to numerical variables.
            x_cat: Portion of decoded data corresponding to categorical variables.
        '''
        dim_embed = self.dim_embed

        ## decode input
        x_dec = sampled_points
        for i in range(len(self.list_layers_bn)):
            x_dec = self.list_layers_dec[i](x_dec)
            # batch normalization
            x_dec = self.list_layers_bn[i](x_dec)
            # activation function
            x_dec = self.relu(x_dec)
            # dropout
            x_dec = self.dropout(x_dec)
        #
        x_dec = self.list_layers_dec[-1](x_dec)

        ## turn embedded categorical variables into logits
        x_num = x_dec[:, :self.n_feat_num]
        x_cat_embed = x_dec[:, self.n_feat_num:]
        x_cat = []
        if hasattr(self, 'list_layer_embed_inv'):
            for i, layer in enumerate(self.list_layer_embed_inv):
                x_cat.append(layer(x_cat_embed[:, i*dim_embed: (i+1)*dim_embed]))
            x_cat = torch.cat(x_cat, dim = -1)
            #
            return x_num, x_cat
        else:
            return x_num, None
    
class VAE(torch.nn.Module):
    def __init__(self, n_feat_num: int = 0, list_neurons: list = [], list_num_vals_cat: list = [], dim_embed: int = 10,
                 dropout: float = 0) -> None:
        '''
        Class to implement the VAE.

        Args:
            n_feat_num: Number of numerical features.
            list_neurons: List of numbers of neurons used during encoding.
            list_num_vals_cat: List containing, for each categorical variables, the number of categories.
            dim_embed: Embedding dimension for categorical variables.
            dropout: Dropout probability.

        Returns: None.
        '''
        super().__init__()
        #
        if (n_feat_num == 0) and (len(list_num_vals_cat) == 0):
            raise(Exception('At least one numerical or categorical feature is needed.'))
        #
        self.encoder = Encoder(n_feat_num = n_feat_num, list_neurons = list_neurons, list_num_vals_cat = list_num_vals_cat,
                               dim_embed = dim_embed, dropout = dropout)
        self.decoder = Decoder(encoder = self.encoder)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Function to apply the VAE to input data

        Args:
            x: Input tensor: all the categorical variables are assumed to be after all the numerical ones.

        Returns:
            mu: Mean values of features in latent space.
            log_sigma: Variances in latent space.
            x_hat_num: Portion of decoded data corresponding to numerical variables.
            x_hat_cat: Portion of decoded data corresponding to categorical variables.
        '''
        ## encode input
        mu, log_sigma = self.encoder(x = x)
        
        ## sample points
        z = torch.randn_like(mu)
        sampled_points = mu + torch.exp(log_sigma/2)*z

        ## decode sample data
        x_hat_num, x_hat_cat = self.decoder(sampled_points = sampled_points)
        #
        return mu, log_sigma, x_hat_num, x_hat_cat
    
class TrainModel:
    def __init__(self, model: torch.nn.Module, dict_params: dict, dataloader_train: torch.utils.data.DataLoader,
                 dataloader_valid: torch.utils.data.DataLoader) -> None:
        '''
        Class to train a VAE.

        Args:
            model: Model to be trained.
            dict_params: Dictionary containing the relevant parameters for training.
            dataloader_train: Training dataloader.
            dataloader_valid: Validation dataloader.

        Returns:
            None.
        '''
        self.model = model
        self.dict_params_training = dict_params['training']
        self.optimizer = torch.optim.AdamW(params = model.parameters(), lr = dict_params['training']['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer, factor = 0.5)
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.path_artifacts = dict_params['training']['path_artifacts']

    def loss_func_reconstr(self, decoder: torch.nn.Module, x_true: torch.Tensor, x_hat_num: torch.Tensor,
                           x_hat_cat: torch.Tensor) -> torch.Tensor:
        '''
        Function to compute the reconstruction loss.

        Args:
            decoder: VAE decoder.
            x_true: Input tensor to the VAE.
            x_hat_num: Portion of reconstructed tensor associated to numerical features.
            x_hat_cat: Portion of reconstructed tensor associated to categorical features.

        Returns:
            loss: Reconstruction loss.
        '''
        ## reconstruction loss from numerical variables
        x_true_num = x_true[:, :decoder.n_feat_num]
        loss = torch.nn.MSELoss()(x_hat_num, x_true_num)

        ## reconstruction loss from categorical variables
        x_true_cat = x_true[:, decoder.n_feat_num:]
        counter_feat = 0
        for i, layer in enumerate(decoder.list_layer_embed_inv):
            target = x_true_cat[:, i]
            pred = x_hat_cat[:, counter_feat: counter_feat + layer.out_features]
            # compute classification loss (average over features)
            loss_cat = torch.nn.CrossEntropyLoss()(pred, target.long())/len(decoder.list_layer_embed_inv)
            loss = loss + loss_cat
            #
            counter_feat += layer.out_features

        return loss

    def loss_func_KL(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        '''
        Function to compute the KL loss.

        Args:
            mu: Mean values of features in latent space.
            log_sigma: Variances in latent space.

        Returns:
            loss: KL loss.
        '''
        loss = -0.5*torch.mean(1 + log_sigma - mu**2 - torch.exp(log_sigma))
        return loss

    def loss_func(self, decoder: torch.nn.Module, x_true: torch.Tensor, x_hat_num: torch.Tensor, x_hat_cat: torch.Tensor,
                  mu: torch.Tensor, log_sigma: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
        '''
        Function to compute the total loss, as the sum of reconstruction and KL losses.

        Args:
            decoder: VAE decoder.
            x_true: Input tensor to the VAE.
            x_hat_num: Portion of reconstructed tensor associated to numerical features.
            x_hat_cat: Portion of reconstructed tensor associated to categorical features.
            mu: Mean values of features in latent space.
            log_sigma: Variances in latent space.
            beta: Coefficient of the KL loss function.

        Returns:
            loss: total loss.
        '''
        loss_reconstr = self.loss_func_reconstr(decoder = decoder, x_true = x_true, x_hat_num = x_hat_num, x_hat_cat = x_hat_cat)
        loss_kl = self.loss_func_KL(mu = mu, log_sigma = log_sigma)
        loss = loss_reconstr + beta*loss_kl
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
        model = self.model
        device = next(model.parameters()).device
        #
        if training == True:
            self.optimizer.zero_grad()
        #
        X = batch
        X = X.to(device)
        mu, log_sigma, x_hat_num, x_hat_cat = model(X)
        #
        loss = self.loss_func(decoder = model.decoder, x_true = X, x_hat_num = x_hat_num, x_hat_cat = x_hat_cat, mu = mu,
                              log_sigma = log_sigma, beta = self.dict_params_training['beta_vae'])
        #
        if training == True:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        self.model.train()
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
        self.model.eval()
        loss_epoch = 0
        with torch.no_grad():
            for batch in self.dataloader_valid:
                loss_epoch += self._model_on_batch(batch = batch, training = False)
        return loss_epoch/len(self.dataloader_valid)

    def train_model(self) -> Tuple[torch.nn.Module, list, list]:
        '''
        Function to train the model.
        
        Args: None.
            
        Returns:
            model: Trained model.
            list_loss_train: List of training loss function across the epochs.
            list_loss_valid: List of validation loss function across the epochs.
        '''
        model = self.model
        dict_params_training = self.dict_params_training
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
                best_weights = model.state_dict()
                if path_artifacts is not None:
                        torch.save(model.state_dict(), path_artifacts)
            if counter_patience >= dict_params_training['patience']:
                print(f'Training stopped at epoch {epoch}. Restoring weights from epoch {np.argmin(list_loss_valid) + 1}.')
                model.load_state_dict(torch.load(path_artifacts))
                break
            #
            print(f'Epoch {epoch}: training loss = {loss_train:.4f}, validation loss = {loss_valid:.4f}, patience counter = {counter_patience}.')
            self.scheduler.step(loss_valid)
            #
            list_loss_train.append(loss_train)
            list_loss_valid.append(loss_valid)
        #
        if path_artifacts is not None:
            model.load_state_dict(torch.load(path_artifacts))
        model.load_state_dict(best_weights)
        return model, list_loss_train, list_loss_valid