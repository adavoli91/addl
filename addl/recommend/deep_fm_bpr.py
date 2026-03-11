import numpy as np
import torch
from typing import Tuple
from fm_bpr import FactorizationMachine
    
class Deep(torch.nn.Module):
    def __init__(self, list_neurons_deep: list, fm: torch.nn.Module, dropout: float = 0) -> None:
        '''
        Class to implement the deep part of DeepFM.

        Args:
            list_neurons_deep: List of neurons for the deep part of DeepFM.
            fm: FM model.
            dropout: Dropout probability.

        Returns: None.
        '''
        super().__init__()

        self.fm = fm
        # number of features
        n_feat = len(fm.list_layers_lin_cat)
        if 'layer_lin_num' in [module[0] for module in fm.named_parameters()]:
            n_feat += fm.layer_lin_num.shape[-1]
        # embedding dimension
        dim_embedding = fm.list_layers_embed_cat[0].embedding_dim

        ## dense layers
        list_neurons_deep = [n_feat*dim_embedding] + list_neurons_deep + [1]
        list_layers_lin, list_layers_bn = [], []
        for i in range(1, len(list_neurons_deep)):
            list_layers_lin.append(torch.nn.Linear(in_features = list_neurons_deep[i-1], out_features = list_neurons_deep[i]))
            if i < len(list_neurons_deep) - 1:
                list_layers_bn.append(torch.nn.BatchNorm1d(num_features = list_neurons_deep[i]))
        self.list_layers_lin = torch.nn.ModuleList(list_layers_lin)
        self.layer_first_bn = torch.nn.BatchNorm1d(num_features = n_feat*dim_embedding)
        self.list_layers_bn = torch.nn.ModuleList(list_layers_bn)
        #
        self.relu = torch.nn.ReLU()
        #
        self.dropout = torch.nn.Dropout(p = dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Function to  make the prediction.

        Args:
            x: Input tensor with indices for the categorical variables (and possibly numerical values).

        Returns:
            y: Predicted variable.
        '''
        ## extract embeddings
        fm = self.fm
        # embedding terms - categorical variables
        counter_var = 0
        list_embed_terms = []
        for layer in fm.list_layers_embed_cat:
            list_embed_terms.append(layer(x[:, counter_var: counter_var + 1].int()))
            counter_var += 1
        # concatenate results
        list_embed_terms = torch.cat(list_embed_terms, dim = 1)
        # embedding terms - numerical variables
        if 'layer_embed_num' in [module[0] for module in fm.named_children()]:
            y_num = fm.layer_embed_num(x[:, counter_var:]).reshape(x.shape[0], fm.n_feat_num, -1)
            # concatenate results
            list_embed_terms = torch.cat((list_embed_terms, y_num), dim = 1)
        
        y = list_embed_terms.reshape(x.shape[0], -1)
        y = self.layer_first_bn(y)
        ## deep part
        for i in range(len(self.list_layers_bn)):
            y = self.list_layers_lin[i](y)
            y = self.list_layers_bn[i](y)
            y = self.relu(y)
            y = self.dropout(y)
        y = self.list_layers_lin[-1](y)
        
        return y

class DeepFM(torch.nn.Module):
    def __init__(self, list_num_vals_cat: list, dim_embed: int, list_neurons_deep: list, n_feat_num: int = 0, dropout: float = 0) -> None:
        '''
        Class to implement DeepFM.

        Args:
            list_num_vals_cat: List containing the number of different values for each categorical variable.
            dim_embed: Embedding dimension.
            list_neurons_deep: List of neurons for the deep part of DeepFM.
            n_feat_num: Number of numerical features.
            dropout: Dropout probability.

        Returns: None.
        '''
        super().__init__()
        
        ## FM
        self.fm = FactorizationMachine(list_num_vals_cat = list_num_vals_cat, dim_embed = dim_embed, n_feat_num = n_feat_num)

        ## deep
        self.deep = Deep(list_neurons_deep = list_neurons_deep, fm = self.fm, dropout = dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Function to  make the prediction.

        Args:
            x: Input tensor with indices for the categorical variables (and possibly numerical values).

        Returns:
            y: Predicted variable, as the sum of FM and deep parts.
        '''
        ## FM part
        y_fm = self.fm(x)
    
        ## deep part
        y_deep = self.deep(x)
        
        return y_fm + y_deep

class DeepFMBPR(torch.nn.Module):
    def __init__(self, list_num_vals_cat: list, dim_embed: int, list_neurons_deep: list, dict_unseen: dict,
                 n_feat_num: int = 0, dropout: float = 0, seed: int = 123) -> None:
        '''
        Class to implement a DeepFM which computes p(i > j | Theta).

        Args:
            list_num_vals_cat: List containing the number of different values for each categorical variable.
            dim_embed: Embedding dimension.
            list_neurons_deep: List of neurons for the deep part of DeepFM.
            dict_unseen: Dictionary which contains, for each user, a list of tuples indicating unseen items; it
                         should be of the form {user_1: [(item_1, ...), (item_2, ...)]}
            n_feat_num: Number of numerical features.
            dropout: Dropout probability.
            seed: Random seed.

        Returns: None.
        '''
        super().__init__()
        #
        self.dict_unseen = dict_unseen
        self.rng = np.random.default_rng(seed = seed)

        ## FM
        self.deep_fm = DeepFM(list_num_vals_cat = list_num_vals_cat, dim_embed = dim_embed,
                              list_neurons_deep = list_neurons_deep, n_feat_num = n_feat_num, dropout = dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Function to make the prediction.

        Args:
            x: Input tensor with indices for the categorical variables (and possibly numerical values).

        Returns:
            x_uij: Difference between FM prediction on the two inputs.
        '''
        x_unseen = torch.tensor([self.rng.choice(self.dict_unseen[i]) for i in x[:, 0].detach().numpy()])
        x_unseen = torch.cat((x[:, :1], x_unseen), dim = 1)
        # FM prediction
        x_ui = self.deep_fm(x)
        x_uj = self.deep_fm(x_unseen)
        #
        x_uij = x_ui - x_uj
        return x_uij

class TrainModel:
    def __init__(self, model: torch.nn.Module, dict_params: dict, dataloader_train: torch.utils.data.DataLoader,
                 dataloader_valid: torch.utils.data.DataLoader, coef_reg: float = 1e-3) -> None:
        '''
        Class to train a Deep FM with BPR loss.

        Args:
            model: Model to be trained.
            dict_params: Dictionary containing the relevant parameters for training.
            dataloader_train: Training dataloader.
            dataloader_valid: Validation dataloader.
            coef_reg: Regularization coefficient.

        Returns:
            None.
        '''
        self.model = model
        self.dict_params_training = dict_params['training']
        self.optimizer = torch.optim.AdamW(params = model.parameters(), lr = dict_params['training']['lr'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer, factor = 0.5)
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.coef_reg = coef_reg
        self.path_artifacts = dict_params['training']['path_artifacts']

    def loss_func(self, x_uij: torch.Tensor) -> float:
        '''
        Function to implement the BPR loss.

        Args:
            x_uij: Difference between the score returned by to model on seen and unseen instances.
        
        Returns:
            loss: BPR loss.
        '''
        loss = -torch.nn.LogSigmoid()(x_uij).mean()
        for module in self.model.children():
            if type(module) == torch.nn.ModuleList:
                loss += self.coef_reg*sum([torch.linalg.norm([layer.weight])**2 for layer in module])
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
        x_uij = model(X).to(device)
        #
        loss = self.loss_func(x_uij = x_uij)
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