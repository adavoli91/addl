import numpy as np
import torch
from torch import nn
from typing import Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.utils.data import DataLoader

class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        '''
        Class to define a masked linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Whether to include the bias term.

        Returns: None.
        '''
        super().__init__(in_features = in_features, out_features = out_features, bias = bias)

    def forward(self, x: torch.tensor, mask_concat: torch.tensor) -> torch.tensor:
        '''
        Function to apply a masked linear layer.

        Args:
            x: Input tensor, of shape (n_ensemb, batch_size, in_features).
            mask_concat: Concatenated mask, of shape (n_ensemb, out_features, in_features).

        Returns:
            y: Output tensor, of shape (n_ensemb, batch_size, out_features).
        '''
        #
        weight = self.weight
        # get masked weights, of shape (n_ensemb, out_features, in_features)
        masked_weights = torch.einsum('eoi, oi -> eoi', mask_concat, weight)
        # get output, of shape (n_ensemb, batch_size, out_features)
        y = torch.einsum('eni, eio -> eno', x, masked_weights.transpose(1, 2))
        return y
    
class MADE(nn.Module):
    def __init__(self, n_feat: int, list_neurons: list, use_fix_mask: bool = False, n_ensemb: int = 10,
                 force_ensembling: bool = False, n_masks_to_draw_from: int = 50) -> None:
        '''
        Implementation of the MADE model.

        Args:
            n_feat: Number of input features.
            list_neurons: List of neurons of the hidden layer(s).
            use_fix_mask: Whether to use a fixed mask.
            n_ensemb: Number of ensembles to be considered.
            force_ensembling: Whether to force ensembling, even during valiation.
            n_masks_to_draw_from: Number of masks which are generated at the beginning and one samples `n_ensemb` from.

        Returns: None.
        '''
        super().__init__()
        # disable ensembling if not desired or during validation
        if ((use_fix_mask == True) or (self.training == False)) and (force_ensembling == False):
            n_ensemb = 1
        #
        list_neurons = [n_feat] + list_neurons + [n_feat]
        #
        self.n_feat = n_feat
        self.list_neurons = list_neurons
        self.use_fix_mask = use_fix_mask
        self.n_ensemb = n_ensemb
        # construct layers
        list_layer = []
        for i in range(1, len(list_neurons)):
            list_layer.append(MaskedLinear(in_features = list_neurons[i-1], out_features = list_neurons[i]))
        self.layer_skip_conn = MaskedLinear(in_features = n_feat, out_features = n_feat, bias = False)
        self.list_layer = nn.ModuleList(list_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # set of masks and coefficients to draw from
        dict_temp_mask, dict_temp_coeff = {i: [] for i in range(len(list_neurons) - 1)}, {i: [] for i in range(len(list_neurons))}
        for i in range(n_masks_to_draw_from):
            mask, coeff = self._define_mask()
            for j in range(len(list_neurons) - 1):
                dict_temp_mask[j].append(mask[j])
                dict_temp_coeff[j].append(coeff[j])
            dict_temp_coeff[j + 1].append(coeff[j + 1])
        # concatenate masks and coefficients according to the layer they refer to
        for i in range(len(dict_temp_coeff)):
            # for each layer, construct a tensor of masks of shape (n_masks_to_draw_from, dim_out, dim_in)
            if i < len(dict_temp_mask):
                concat_mask_i = torch.cat([mat.unsqueeze(0) for mat in dict_temp_mask[i]], axis = 0)
                self.register_buffer(f'list_masks_{i}', concat_mask_i)
            # for each layer, construct a tensor of masking coefficients of shape (n_masks_to_draw_from, dim_out)
            if i == 0:
                concat_coeff_mask_0 = torch.cat([coeff.unsqueeze(0) for coeff in dict_temp_coeff[i]], axis = 0)
                self.register_buffer(f'list_coeff_mask_{i}', concat_coeff_mask_0)
            else:
                concat_coeff_mask_i = torch.cat([coeff.unsqueeze(0) for coeff in dict_temp_coeff[i]], axis = 0)
                self.register_buffer(f'list_coeff_mask_{i}', concat_coeff_mask_i)
            # for each layer, construct a tensor of masks used for skip connections of shape (n_masks_to_draw_from, dim_out, n_feat)
            if i == len(dict_temp_coeff) - 1:
                self.register_buffer('mask_skip_conn', concat_coeff_mask_i.unsqueeze(2) > concat_coeff_mask_0.unsqueeze(1))

    def _define_mask(self) -> Tuple[list, list]:
        '''
        Function to define a list of maskings which can be randomly sampled during training and used as an ensemble.

        Args: None.

        Returns:
            list_masks: List of masking matrices.
            list_coeff_mask: List of masking coefficients used to construct the masking matrices.
        '''
        n_feat = self.n_feat
        # keep neurons from hidden layers only
        list_neurons = self.list_neurons[1:-1]
        # initial mask coefficients (i.e., permutation of the features)
        min_connectivity = 1
        list_coeff_mask = [np.random.permutation(n_feat) + 1]
        for n_neur in list_neurons:
            # draw mask coefficients
            coeff_mask = np.random.randint(low = min_connectivity, high = n_feat, size = n_neur)
            list_coeff_mask.append(coeff_mask)
            min_connectivity = coeff_mask.min()
        list_coeff_mask.append(list_coeff_mask[0])
        # create mask matrices
        list_masks = []
        for i, _ in enumerate(list_coeff_mask):
            if i == 0:
                continue
            if i < len(list_coeff_mask) - 1:
                list_masks.append((list_coeff_mask[i].reshape(-1, 1) >= list_coeff_mask[i-1]).astype(int))
            else:
                list_masks.append((list_coeff_mask[i].reshape(-1, 1) > list_coeff_mask[i-1]).astype(int))
        #
        list_coeff_mask = [torch.tensor(coeff_mask) for coeff_mask in list_coeff_mask]
        list_masks = [torch.tensor(mask) for mask in list_masks]
        #
        return list_masks, list_coeff_mask
        
    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        '''
        Function to reconstruct the input with MADE.

        Args:
            x: Input tensor, of shape (batch_size, 1, n_feat).

        Returns:
            x_in: Input tensor, same as `x`.
            x_hat: Reconstructed input.
        '''
        # list_masks = self.list_masks
        n_ensemb = self.n_ensemb
        # random draw of masks and coefficients
        idx_masks = np.random.choice(self.list_masks_0.shape[0], size = n_ensemb)
        # repeat input tensor
        x_in = x.unsqueeze(axis = 0).repeat(n_ensemb, 1, 1)
        # apply masked linear layers
        x_hat = x_in
        for i in range(len(self.list_layer)):
            layer = self.list_layer[i]
            x_hat_lin = layer(x = x_hat, mask_concat = getattr(self, f'list_masks_{i}')[idx_masks])
            #
            # layer_skip_conn = self.list_layer_skip_conn[i]
            # x_skip = layer_skip_conn(x = x_in, mask_concat = getattr(self, f'list_masks_skip_conn_{i}')[idx_masks])
            #
            x_hat = self.relu(x_hat_lin)
        x_skip = self.layer_skip_conn(x = x_in, mask_concat = self.mask_skip_conn[idx_masks])
        x_hat = self.sigmoid(x_hat + x_skip)
        #
        if self.training == False:
            return x_in[0], x_hat[0]
        return x_in, x_hat
    

class TrainModel:
    def __init__(self, model: torch.nn.Module, dict_params: dict, dataloader_train: DataLoader, dataloader_valid: DataLoader,
                 path_artifacts: str = None) -> None: 
        '''
        Class to train the model.

        Args:
            model: PyTorch model.
            dict_params: Dictionary containing information about the model architecture.
            dataloader_train: Dataloader containing training data.
            dataloader_valid: Dataloader containing validation data.
            path_artifacts: Path where to save artifacts.

        Returns: None.
        '''
        self.model = model
        self.dict_params = dict_params
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.path_artifacts = path_artifacts
        #
        self.optimizer = Adam(params = model.parameters())
        self.scheduler = ReduceLROnPlateau(optimizer = self.optimizer, patience = dict_params['training']['patience_sched'], factor = 0.5)
        self.loss_func = nn.BCELoss(reduction = 'none')

    def _model_on_batch(self, batch: torch.Tensor, training: bool, loss_epoch: float)-> float:
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
        loss = self.loss_func
        #
        if training == True:
            self.optimizer.zero_grad()
        #
        X = batch.to(next(model.parameters()).device)
        #
        X, X_hat = model(X)
        X = X.to(next(model.parameters()).device)
        X_hat = X_hat.to(next(model.parameters()).device)
        # sum over features and average over ensemble
        loss_val = loss(X_hat, X).sum(axis = -1).mean(axis = 0)
        # average over batch
        loss_val = loss_val.mean()
        #
        if training == True:
            loss_val.backward()
            self.optimizer.step()
        #
        loss_epoch += loss_val.item()
        return loss_epoch

    def _train(self) -> float:
        '''
        Function to train the model on a single epoch.

        Args: None.
            
        Returns:
            loss: Value of the training loss function per batch.
        '''
        dataloader = self.dataloader_train
        loss = 0
        self.model.train()
        for batch in dataloader:
            loss += self._model_on_batch(batch, training = True, loss_epoch = loss)/len(dataloader)
        return loss

    def _eval(self) -> float:
        '''
        Function to evaluate the model on the validation set on a single epoch.

        Args: None.
            
        Returns:
            loss: Value of the validation loss function per batch.
        '''
        dataloader = self.dataloader_valid
        loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                loss += self._model_on_batch(batch, training = False, loss_epoch = loss)/len(dataloader)
        return loss

    def train_model(self) -> Tuple[torch.nn.Module, dict]:
        '''
        Function to train the model.

        Args: None.
            
        Returns:
            model: Trained model.
            dict_artifacts: Dictionary containing model weights and loss functions.
        '''
        model = self.model
        dict_params = self.dict_params
        path_artifacts = self.path_artifacts
        n_epochs = dict_params['training']['n_epochs']
        #
        dict_artifacts = {}
        #
        list_loss_train, list_loss_valid = [], []
        counter_patience = 0
        for epoch in range(n_epochs):
            loss_train = self._train()
            loss_valid = self._eval()
            #
            self.scheduler.step(loss_valid)
            #
            if (len(list_loss_valid) > 0) and (loss_valid >= np.min(list_loss_valid)*(1 - dict_params['training']['min_delta'])):
                counter_patience += 1
            if (len(list_loss_valid) == 0) or (loss_valid < np.min(list_loss_valid)):
                counter_patience = 0
                dict_artifacts['weights'] = model.state_dict()
                    # save weights
                if path_artifacts is not None:
                    torch.save(model.state_dict(), path_artifacts)
            #
            list_loss_train.append(loss_train)
            list_loss_valid.append(loss_valid)
            #
            print(f"Epoch {epoch + 1}: training loss: {loss_train:.7f}, validation loss: {loss_valid:.7f}, learning rate = {self.optimizer.param_groups[0]['lr']}, counter patience = {counter_patience}")
            #
            if counter_patience >= dict_params['training']['patience']:
                print(f'Training stopped at epoch {epoch + 1}. Restoring weights from epoch {np.argmin(list_loss_valid) + 1}.')
                break

        dict_artifacts['loss_train'] = list_loss_train
        dict_artifacts['loss_valid'] = list_loss_valid
        if path_artifacts is not None:
            model.load_state_dict(torch.load(path_artifacts))
        return model, dict_artifacts