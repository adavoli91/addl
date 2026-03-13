import numpy as np
import torch
import copy
from typing import Tuple

class Encoder(torch.nn.Module):
    def __init__(self, n_feat_num: int, list_neurons: list, list_num_vals_cat: list, dim_embed: int = 5, dropout: float = 0) -> None:
        '''
        Class to implement an encoder.

        Args:
            n_feat_num: Number of numerical features.
            list_neurons: List of neurons for the hidden layers.
            list_num_vals_cat: List containing the number of different values for each categorical variable.
            dim_embed: Embedding dimension.
            dropout: Dropout probability.

        Returns: None.
        '''
        super().__init__()

        n_feat = n_feat_num
        self.n_feat_num = n_feat_num
        self.list_num_vals_cat = list_num_vals_cat

        ## embedding for categorical features
        if len(list_num_vals_cat) > 0:
            list_layers_embed = []
            for n_vals in list_num_vals_cat:
                list_layers_embed.append(torch.nn.Embedding(num_embeddings = n_vals, embedding_dim = dim_embed))
            self.list_layers_embed = torch.nn.ModuleList(list_layers_embed)
            #
            n_feat += len(list_num_vals_cat)*dim_embed

        ## encoding layers
        list_neurons = [n_feat] + list_neurons
        list_layers, list_layers_bn = [], []
        for i in range(1, len(list_neurons)):
            list_layers.append(torch.nn.Linear(in_features = list_neurons[i-1], out_features = list_neurons[i]))
            list_layers_bn.append(torch.nn.BatchNorm1d(num_features = list_neurons[i]))
        self.list_layers = torch.nn.ModuleList(list_layers)
        self.list_layers_bn = torch.nn.ModuleList(list_layers_bn)

        ## relu
        self.relu = torch.nn.ReLU()

        ## dropout
        self.dropout = torch.nn.Dropout(p = dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Function to encode the input.

        Args:
            x: Input tensor: all the numerical features are assumed to be in the first `n_feat_num` columns, and categorical ones are
               assumed to be expressed as indices of some encoder (e.g., OrdinalEncoder).

        Returns:
            x_enc: Encoded input.
        '''
        x_enc = x

        ## check dimensions
        if x_enc.shape[1] != self.n_feat_num + len(self.list_num_vals_cat):
            raise(ValueError(f'Dimensions do not match: the model expects an input with {self.n_feat_num} numerical and {len(self.list_num_vals_cat)} categorical features, but the provided one has {x_enc.shape[1]} features.'))
        
        ## embed categorical variables
        if hasattr(self, 'list_layers_embed'):
            x_enc_num = x[:, :self.n_feat_num]
            x_temp = x[:, self.n_feat_num:]
            #
            counter_var = 0
            x_enc_cat = []
            for layer in self.list_layers_embed:
                x_enc_cat.append(layer(x_temp[:, counter_var: counter_var + 1].int()))
                counter_var += 1
            x_enc_cat = torch.cat(x_enc_cat, dim = 1)
            x_enc_cat = x_enc_cat.view(x_enc_cat.shape[0], -1)
            #
            x_enc = torch.cat((x_enc_num, x_enc_cat), dim = 1)

        ## encode (embedded) input
        for i in range(len(self.list_layers)):
            x_enc = self.list_layers[i](x_enc)
            x_enc = self.list_layers_bn[i](x_enc)
            x_enc = self.relu(x_enc)
            x_enc = self.dropout(x_enc)

        return x_enc
    
class Decoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        '''
        Class to implement a decoder.

        Args:
            encoder: Encoder.

        Returns: None.
        '''
        super().__init__()

        ## decoding layers
        list_layers, list_layers_bn = [], []
        for i, layer in enumerate(reversed(encoder.list_layers)):
            list_layers.append(torch.nn.Linear(in_features = layer.out_features, out_features = layer.in_features))
            if i < len(encoder.list_layers) - 1:
                list_layers_bn.append(torch.nn.BatchNorm1d(num_features = layer.in_features))
        self.list_layers = torch.nn.ModuleList(list_layers)
        self.list_layers_bn = torch.nn.ModuleList(list_layers_bn)

        ## inverse embedding
        if hasattr(encoder, 'list_layers_embed'):
            list_layers_embed_inv = []
            for layer in encoder.list_layers_embed:
                list_layers_embed_inv.append(torch.nn.Linear(in_features = layer.embedding_dim, out_features = layer.num_embeddings))
            self.list_layers_embed_inv = torch.nn.ModuleList(list_layers_embed_inv)
    
        ## relu
        self.relu = torch.nn.ReLU()
    
        ## dropout
        self.dropout = torch.nn.Dropout(p = encoder.dropout.p)

    def forward(self, x_enc: torch.Tensor) -> Tuple[torch.Tensor, list]:
        '''
        Function to decode the input.

        Args:
            x_enc: Encoded input.

        Returns:
            x_dec_num: Decoded input corresponding to numerical variables.
            x_dec_cat: Decoded input corresponding to categorical variables: each element of the list is the decoded representation of a single
                       categorical feature.
        '''
        x_dec = x_enc
        
        # reverse encoding
        for i in range(len(self.list_layers_bn)):
            x_dec = self.list_layers[i](x_dec)
            x_dec = self.list_layers_bn[i](x_dec)
            x_dec = self.relu(x_dec)
            x_dec = self.dropout(x_dec)
        x_dec = self.list_layers[-1](x_dec)

        ## reverse embedding of categorical variables (if any)
        if hasattr(self, 'list_layers_embed_inv'):
            n_vals_cat = len(self.list_layers_embed_inv)
            dim_embed = self.list_layers_embed_inv[0].in_features
            #
            x_dec_num = x_dec[:, :-n_vals_cat*dim_embed]
            x_dec_cat_emb = x_dec[:, -n_vals_cat*dim_embed:]
            #
            x_dec_cat = []
            for i in range(n_vals_cat):
                x_dec_cat.append(self.list_layers_embed_inv[i](x_dec_cat_emb[:, i*dim_embed: (i+1)*dim_embed]))
        else:
            n_vals_cat = 0
            x_dec_num = x_dec
            x_dec_cat = []

        return x_dec_num, x_dec_cat
    
class Autoencoder(torch.nn.Module):
    def __init__(self, n_feat_num: int, list_neurons: list, list_num_vals_cat: list, dim_embed: int = 5, dropout: float = 0):
        '''
        Class to implement an autoencoder which works with both numerical and categorical variables.

        Args:
            n_feat_num: Number of numerical features.
            list_neurons: List of neurons for the hidden layers.
            list_num_vals_cat: List containing the number of different values for each categorical variable.
            dim_embed: Embedding dimension.
            dropout: Dropout probability.

        Returns: None.
        '''
        super().__init__()

        ## encoder
        self.encoder = Encoder(n_feat_num = n_feat_num, list_neurons = list_neurons, list_num_vals_cat = list_num_vals_cat, dim_embed = dim_embed,
                               dropout = dropout)

        ## decoder
        self.decoder = Decoder(encoder = self.encoder)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        '''
        Function to decode the input.

        Args:
            x: Input tensor: all the numerical features are assumed to be in the first `n_feat_num` columns, and categorical ones are
               assumed to be expressed as indices of some encoder (e.g., OrdinalEncoder).

        Returns:
            x_dec_num: Decoded input corresponding to numerical variables.
            x_dec_cat: Decoded input corresponding to categorical variables: each element of the list is the decoded representation of a single
                       categorical feature.
        '''
        ## encder
        x_enc = self.encoder(x)

        ## decoder
        x_dec_num, x_dec_cat = self.decoder(x_enc)

        return x_dec_num, x_dec_cat

class TrainModel:
    def __init__(self, model: torch.nn.Module, dict_params: dict, dataloader_train: torch.utils.data.DataLoader,
                 dataloader_valid: torch.utils.data.DataLoader, rel_weight_losses: float = 0.5) -> None:
        '''
        Class to train an autoencoder for both numerical and categorical variables.

        Args:
            model: Model to be trained.
            dict_params: Dictionary containing the relevant parameters for training.
            dataloader_train: Training dataloader.
            dataloader_valid: Validation dataloader.
            rel_weight_losses: Relative weight to be used between numerical and categorical losses;
                               the total loss is `rel_weight_losses`*MSE + (1 - `rel_weight_losses`)*CrossEntropy.

        Returns:
            None.
        '''
        self.model = model
        self.dict_params_training = dict_params['training']
        self.optimizer = torch.optim.Adam(params = model.parameters(), lr = dict_params['training']['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer, factor = 0.5)
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.path_artifacts = dict_params['training']['path_artifacts']
        self.n_feat_num = model.encoder.n_feat_num
        self.rel_weight_losses = rel_weight_losses

    def _loss_num(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        '''
        Function to compute the loss for numerical variables.

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
        Function to compute the loss for categorical variables.

        Args:
            input: Input (reconstructed) tensor.
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

    def loss(self, x_dec_num: torch.Tensor, x_dec_cat, target: torch.Tensor) -> torch.Tensor:
        '''
        Function to compute the loss for both numerical and categorical variables.

        Args:
            x_dec_num: Input (reconstructed) tensor for numerical variables.
            x_dec_cat: Input (reconstructed) tensor for categorical variables.
            target: Target (ground truth) tensor.

        Returns:
            loss: Value of the loss function.
        '''
        rel_weight_losses = self.rel_weight_losses
        loss = 0

        # loss from numerical features
        if x_dec_num.shape[1] > 0:
            loss = loss + rel_weight_losses*self._loss_num(input = x_dec_num, target = target[:, :self.n_feat_num])
            
        # loss from categorical features
        if len(x_dec_cat) > 0:
            loss = loss + (1 - rel_weight_losses)*self._loss_cat(input = x_dec_cat, target = target[:, self.n_feat_num:])

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
        X_dec_num, X_dec_cat = model(X)
        loss = self.loss(x_dec_num = X_dec_num, x_dec_cat = X_dec_cat, target = X)
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
                best_weights = copy.deepcopy(model.state_dict())
                if path_artifacts is not None:
                    torch.save(model.state_dict(), path_artifacts)
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
            model.load_state_dict(torch.load(path_artifacts))
        model.load_state_dict(best_weights)
        return model, list_loss_train, list_loss_valid