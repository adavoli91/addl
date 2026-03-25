import numpy as np
import pandas as pd
import torch
import copy
from typing import Tuple

class LightFM(torch.nn.Module):
    def __init__(self, list_n_vals_cat_user: list = [], n_feat_num_user: int = 0,
                 list_n_vals_cat_item: list = [], n_feat_num_item: int = 0, dim_embed: int = 5) -> None:
        '''
        Class to implement LightFM.

        Args:
            list_n_vals_cat_user: List containing the number of different values for each categorical variable related to users.
            n_feat_num_user: Number of numerical features related to users.
            list_n_vals_cat_item: List containing the number of different values for each categorical variable related to items.
            n_feat_num_item: Number of numerical features related to items.
            dim_embed: Embedding dimension.

        Returns: None.
        '''
        super().__init__()
        self.n_feat_num_user = n_feat_num_user
        self.n_feat_num_item = n_feat_num_item
        self.dim_embed = dim_embed

        ## q_u
        # embedding of categorical variables
        if len(list_n_vals_cat_user) > 0:
            list_layers_q_user_cat = []
            for n_val in list_n_vals_cat_user:
                layer = torch.nn.Embedding(num_embeddings = n_val, embedding_dim = dim_embed)
                torch.nn.init.zeros_(layer.weight)
                list_layers_q_user_cat.append(layer)
            self.list_layers_q_user_cat = torch.nn.ModuleList(list_layers_q_user_cat)
        # embedding of numerical variables
        if n_feat_num_user > 0:
            self.list_layers_q_user_num = torch.nn.Linear(in_features = n_feat_num_user,
                                                          out_features = n_feat_num_user*dim_embed, bias = False)
            
        ## p_i
        # embedding of categorical variables
        if len(list_n_vals_cat_item) > 0:
            list_layers_p_item_cat = []
            for n_val in list_n_vals_cat_item:
                layer = torch.nn.Embedding(num_embeddings = n_val, embedding_dim = dim_embed)
                torch.nn.init.zeros_(layer.weight)
                list_layers_p_item_cat.append(layer)
            self.list_layers_p_item_cat = torch.nn.ModuleList(list_layers_p_item_cat)
        # embedding of numerical variables
        if n_feat_num_item > 0:
            self.list_layers_p_item_num = torch.nn.Linear(in_features = n_feat_num_item,
                                                          out_features = n_feat_num_item*dim_embed, bias = False)
            
        ## b_u
        # embedding of categorical variables
        if len(list_n_vals_cat_user) > 0:
            list_layers_b_user_cat = []
            for n_val in list_n_vals_cat_user:
                layer = torch.nn.Embedding(num_embeddings = n_val, embedding_dim = 1)
                torch.nn.init.zeros_(layer.weight)
                list_layers_b_user_cat.append(layer)
            self.list_layers_b_user_cat = torch.nn.ModuleList(list_layers_b_user_cat)
        # embedding of numerical variables
        if n_feat_num_user > 0:
            self.list_layers_b_user_num = torch.nn.Linear(in_features = n_feat_num_user,
                                                          out_features = n_feat_num_user, bias = False)
            
        ## b_i
        # embedding of categorical variables
        if len(list_n_vals_cat_item) > 0:
            list_layers_b_item_cat = []
            for n_val in list_n_vals_cat_item:
                layer = torch.nn.Embedding(num_embeddings = n_val, embedding_dim = 1)
                torch.nn.init.zeros_(layer.weight)
                list_layers_b_item_cat.append(layer)
            self.list_layers_b_item_cat = torch.nn.ModuleList(list_layers_b_item_cat)
        # embedding of numerical variables
        if n_feat_num_item > 0:
            self.list_layers_b_item_num = torch.nn.Linear(in_features = n_feat_num_item,
                                                          out_features = n_feat_num_item, bias = False)
            
    def forward(self, x_user: torch.Tensor, x_item: torch.Tensor) -> torch.Tensor:
        '''
        Function to make the prediction.

        Args:
            x_user: Input tensor with indices for the categorical variables (and possibly numerical values) for users.
            x_item: Input tensor with indices for the categorical variables (and possibly numerical values) for items.

        Returns:
            r: Predicted score
        '''
        ## q_u
        q_u = torch.zeros(x_user.shape[0], self.dim_embed, device = x_user.device)
        counter_var = 0
        # embedding of categorical variables
        if hasattr(self, 'list_layers_q_user_cat'):
            for layer in self.list_layers_q_user_cat:
                q_u = q_u + layer(x_user[:, counter_var].int())
                counter_var += 1
        # embedding of numerical variables
        if hasattr(self, 'list_layers_q_user_num'):
            temp = self.list_layers_q_user_num(x_user[:, counter_var:]).chunk(chunks = self.n_feat_num_user,
                                                                              dim = 1)
            for q_temp in temp:
                q_u = q_u + q_temp
        
        ## p_i
        p_i = torch.zeros(x_item.shape[0], self.dim_embed, device = x_item.device)
        counter_var = 0
        # embedding of categorical variables
        if hasattr(self, 'list_layers_p_item_cat'):
            for layer in self.list_layers_p_item_cat:
                p_i = p_i + layer(x_item[:, counter_var].int())
                counter_var += 1
        # embedding of numerical variables
        if hasattr(self, 'list_layers_p_item_num'):
            temp = self.list_layers_p_item_num(x_item[:, counter_var:]).chunk(chunks = self.n_feat_num_item,
                                                                              dim = 1)
            for p_temp in temp:
                p_i = p_i + p_temp
        
        ## b_u
        b_u = torch.zeros(x_user.shape[0], 1, device = x_user.device)
        counter_var = 0
        # embedding of categorical variables
        if hasattr(self, 'list_layers_b_user_cat'):
            for layer in self.list_layers_b_user_cat:
                b_u = b_u + layer(x_user[:, counter_var].int())
                counter_var += 1
        # embedding of numerical variables
        if hasattr(self, 'list_layers_b_user_num'):
            temp = self.list_layers_b_user_num(x_user[:, counter_var:]).chunk(chunks = self.n_feat_num_user,
                                                                              dim = 1)
            for b_temp in temp:
                b_u = b_u + b_temp
        
        ## b_i
        b_i = torch.zeros(x_item.shape[0], 1, device = x_item.device)
        counter_var = 0
        # embedding of categorical variables
        if hasattr(self, 'list_layers_b_item_cat'):
            for layer in self.list_layers_b_item_cat:
                b_i = b_i + layer(x_item[:, counter_var].int())
                counter_var += 1
        # embedding of numerical variables
        if hasattr(self, 'list_layers_b_item_num'):
            temp = self.list_layers_b_item_num(x_item[:, counter_var:]).chunk(chunks = self.n_feat_num_item,
                                                                              dim = 1)
            for b_temp in temp:
                b_i = b_i + b_temp

        ## r
        r = (q_u*p_i).sum(dim = 1).unsqueeze(dim = 1) + b_u + b_i

        return r
    
class LightFMBPR(torch.nn.Module):
    def __init__(self, dict_unseen: dict, df_user: pd.DataFrame, df_item: pd.DataFrame,
                 list_n_vals_cat_user: list = [], n_feat_num_user: int = 0,
                 list_n_vals_cat_item: list = [], n_feat_num_item: int = 0, dim_embed: int = 5,
                 seed: int = 123) -> None:
        '''
        Class to implement a LightFM which computes p(i > j | Theta).

        Args:
            dict_unseen: Dictionary which contains, for each user, a list of tuples indicating unseen items; it
                         should be of the form {user_1: [(item_1, ...), (item_2, ...)]}
            df_user: Dataframe which contains, for each user, its features.
            df_item: Dataframe which contains, for each item, its features. 
            list_n_vals_cat_user: List containing the number of different values for each categorical variable related to users.
            n_feat_num_user: Number of numerical features related to users.
            list_n_vals_cat_item: List containing the number of different values for each categorical variable related to items.
            n_feat_num_item: Number of numerical features related to items.
            dim_embed: Embedding dimension.
            seed: Random seed.

        Returns: None.
        '''
        super().__init__()
        #
        self.dict_unseen = dict_unseen
        self.rng = np.random.default_rng(seed = seed)
        self.df_user = df_user
        self.df_item = df_item

        ## LightFM
        self.light_fm = LightFM(list_n_vals_cat_user = list_n_vals_cat_user, n_feat_num_user = n_feat_num_user, 
                                list_n_vals_cat_item = list_n_vals_cat_item, n_feat_num_item = n_feat_num_item,
                                dim_embed = dim_embed)

    def forward(self, x_user: torch.Tensor, x_item: torch.Tensor) -> torch.Tensor:
        '''
        Function to make the prediction.

        Args:
            x_user: Input tensor with indices for the categorical variables (and possibly numerical values) for users.
            x_item: Input tensor with indices for the categorical variables (and possibly numerical values) for items.

        Returns:
            x_uij: Difference between LightFM prediction on seen and unseen instances.
        '''
        df_user = self.df_user
        df_item = self.df_item
        #
        list_users = x_user[:, 0].detach().numpy()
        x_unseen_user = torch.tensor(pd.concat([df_user[df_user['user_id'] == id] for id in
                                                list_users]).values).float()
        #
        list_items = [self.rng.choice(self.dict_unseen[i]) for i in list_users]
        x_unseen_item = torch.tensor(pd.concat([df_item[df_item['item_id'] == id] for id in
                                                list_items]).values).float()
        #
        # LightFM prediction
        x_ui = self.light_fm(x_user = x_user, x_item = x_item)
        x_uj = self.light_fm(x_user = x_unseen_user, x_item = x_unseen_item)
        #
        x_uij = x_ui - x_uj
        return x_uij
    
class TrainModel:
    def __init__(self, model: torch.nn.Module, dict_params: dict, dataloader_train: torch.utils.data.DataLoader,
                 dataloader_valid: torch.utils.data.DataLoader, coef_reg: float = 1e-3) -> None:
        '''
        Class to train a LightFM with BPR loss.

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
        self.optimizer = torch.optim.Adam(params = model.parameters(), lr = dict_params['training']['lr'])
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
        X_user, X_item = batch
        X_user, X_item = X_user.to(device), X_item.to(device)
        x_uij = model(x_user = X_user, x_item = X_item).to(device)
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