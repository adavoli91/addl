import numpy as np
import torch
from typing import Tuple

class FactorizationMachine(torch.nn.Module):
    def __init__(self, list_num_vals_cat: list, dim_embed: int, n_feat_num: int = 0) -> None:
        '''
        Class to implement FM.

        Args:
            list_num_vals_cat: List containing the number of different values for each categorical variable.
            dim_embed: Embedding dimension.
            n_feat_num: Number of numerical features.

        Returns: None.
        '''
        super().__init__()

        ## bias
        self.w_0 = torch.nn.Parameter(torch.FloatTensor([[0.]]))

        ## linear terms
        # linear terms - categorical variables (each term represents 'Σ_i x_i w_i' for a given categorical variable)
        list_layers_lin_cat = []
        for n_vals in list_num_vals_cat:
            layer = torch.nn.Embedding(num_embeddings = n_vals, embedding_dim = 1)
            torch.nn.init.zeros_(layer.weight)
            list_layers_lin_cat.append(layer)
        self.list_layers_lin_cat = torch.nn.ModuleList(list_layers_lin_cat)

        # linear terms - numerical variables (each term represents 'Σ_i x_i w_i' for a given numerical variable)
        if n_feat_num > 0:
            self.layer_lin_num = torch.nn.Parameter(torch.tensor([[0]*n_feat_num]).float())
            self.n_feat_num = n_feat_num

        ## embedding terms
        # embedding terms - categorical variables (each term represents 'Σ_i x_i v_{if}' for a given categorical variable)
        list_layers_embed_cat = []
        for n_vals in list_num_vals_cat:
            layer = torch.nn.Embedding(num_embeddings = n_vals, embedding_dim = dim_embed)  
            torch.nn.init.normal_(layer.weight, std = 0.01)
            #
            list_layers_embed_cat.append(layer)
        self.list_layers_embed_cat = torch.nn.ModuleList(list_layers_embed_cat)

        # embedding terms - numerical variables (each term represents 'Σ_i x_i v_{if}' for a given numerical variable)
        if n_feat_num > 0:
            self.layer_embed_num = torch.nn.Linear(in_features = n_feat_num, out_features = n_feat_num*dim_embed, bias = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Function to make the prediction.

        Args:
            x: Input tensor with indices for the categorical variables (and possibly numerical values).

        Returns:
            y: Predicted variable.
        '''
        ## bias
        y = self.w_0

        ## linear terms
        # linear terms - categorical variables
        counter_var = 0
        for layer in self.list_layers_lin_cat:
            y = y + layer(x[:, counter_var: counter_var + 1].int()).sum(dim = -1)
            counter_var += 1

        # linear terms - numerical variables
        if 'layer_lin_num' in [module[0] for module in self.named_parameters()]:
            y_num = (x[:, counter_var:]*self.layer_lin_num).sum(dim = -1).unsqueeze(dim = -1)
            y = y + y_num
        
        ## embedding terms
        # embedding terms - categorical variables
        counter_var = 0
        list_embed_terms = []
        for layer in self.list_layers_embed_cat:
            list_embed_terms.append(layer(x[:, counter_var: counter_var + 1].int()))
            counter_var += 1
        # concatenate results
        list_embed_terms = torch.cat(list_embed_terms, dim = 1)
        
        # embedding terms - numerical variables
        if 'layer_embed_num' in [module[0] for module in self.named_children()]:
            y_num = self.layer_embed_num(x[:, counter_var:]).reshape(x.shape[0], self.n_feat_num, -1)
            # concatenate results
            list_embed_terms = torch.cat((list_embed_terms, y_num), dim = 1)

        #
        y = y + 0.5*((list_embed_terms.sum(dim = 1)**2) - (list_embed_terms**2).sum(dim = 1)).sum(dim = -1).unsqueeze(dim = 1)
        
        return y
    
class FMBPR(torch.nn.Module):
    def __init__(self, list_num_vals_cat: list, dim_embed: int, dict_unseen: dict, n_feat_num: int = 0,
                 seed: int = 123) -> None:
        '''
        Class to implement a FM which computes p(i > j | Theta).

        Args:
            list_num_vals_cat: List containing the number of different values for each categorical variable.
            dim_embed: Embedding dimension.
            dict_unseen: Dictionary which contains, for each user, a list of tuples indicating unseen items; it
                         should be of the form {user_1: [(item_1, ...), (item_2, ...)]}
            n_feat_num: Number of numerical features.
            seed: Random seed.

        Returns: None.
        '''
        super().__init__()
        #
        self.dict_unseen = dict_unseen
        self.rng = np.random.default_rng(seed = seed)

        ## FM
        self.fm = FactorizationMachine(list_num_vals_cat = list_num_vals_cat, dim_embed = dim_embed, n_feat_num = n_feat_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Function to make the prediction.

        Args:
            x: Input tensor with indices for the categorical variables (and possibly numerical values).

        Returns:
            x_uij: Difference between FM prediction on seen and unseen instances.
        '''
        x_unseen = torch.tensor([self.rng.choice(self.dict_unseen[i]) for i in x[:, 0].detach().numpy()])
        x_unseen = torch.cat((x[:, :1], x_unseen), dim = 1)
        # FM prediction
        x_ui = self.fm(x)
        x_uj = self.fm(x_unseen)
        #
        x_uij = x_ui - x_uj
        return x_uij

class TrainModel:
    def __init__(self, model: torch.nn.Module, dict_params: dict, dataloader_train: torch.utils.data.DataLoader,
                 dataloader_valid: torch.utils.data.DataLoader, coef_reg: float = 1e-3) -> None:
        '''
        Class to train a FM with BPR loss.

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