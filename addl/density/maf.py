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
            x: Input tensor, of shape (batch_size, in_features).
            mask_concat: Concatenated mask, of shape (out_features, in_features).

        Returns:
            y: Output tensor, of shape (batch_size, out_features).
        '''
        #
        weight = self.weight
        # get masked weights, of shape (out_features, in_features)
        masked_weights = torch.einsum('oi, oi -> oi', mask_concat, weight)
        # get output, of shape (batch_size, out_features)
        y = torch.einsum('ni, io -> no', x, masked_weights.transpose(0, 1))
        # add bias, of shape (batch_size, out_features)
        if self.bias is not None:
            y = y + self.bias.view(1, -1)
        return y
    
class MADE(nn.Module):
    def __init__(self, n_feat: int, list_neurons: list, n_feat_cond: int = None, random_initial_order: bool = True,
                 rand_rng: np.random._generator.default_rng = np.random.default_rng(123)) -> None:
        '''
        Implementation of the MADE model.

        Args:
            n_feat: Number of input features.
            list_neurons: List of neurons of the hidden layer(s).
            n_feat_cond: Number of features of conditional variables.
            random_initial_order: Whether to initially random permutate features.
            rand_rng: Random generator.

        Returns: None.
        '''
        super().__init__()
        #
        if (n_feat_cond is not None) and (n_feat_cond > 0):
            list_neurons = [n_feat + n_feat_cond] + list_neurons + [2*n_feat]
        else:
            list_neurons = [n_feat] + list_neurons + [2*n_feat]
        #
        self.n_feat = n_feat
        self.list_neurons = list_neurons
        self.n_feat_cond = n_feat_cond
        self.random_initial_order = random_initial_order
        self.rand_rng = rand_rng
        # construct layers
        list_layer = []
        for i in range(1, len(list_neurons)):
            if i < len(list_neurons) - 1:
                list_layer.append(MaskedLinear(in_features = list_neurons[i-1], out_features = list_neurons[i]))
            else:
                list_layer.append(MaskedLinear(in_features = list_neurons[i-1], out_features = list_neurons[i]))
        self.list_layer = nn.ModuleList(list_layer)
        self.relu = nn.ReLU()
        # set of masks and coefficients to draw from
        dict_temp_mask, dict_temp_coeff = {}, {}
        mask, coeff = self._define_mask()
        for i in range(len(list_neurons) - 1):
            self.register_buffer(f'list_masks_{i}', mask[i])
            self.register_buffer(f'list_coeff_mask_{i}', coeff[i])
        self.register_buffer(f'list_coeff_mask_{i + 1}', coeff[i + 1])
        # set weights and bias to 0 so that it outputs mu = 0 = alpha -> at first, u = x (identity mapping)
        with torch.no_grad():
            self.list_layer[-1].weight.zero_()
            if self.list_layer[-1].bias is not None:
                self.list_layer[-1].bias.zero_()

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
        #
        min_connectivity = 1
        # initial mask coefficients (i.e., permutation of the features)
        if self.random_initial_order == True:
            list_coeff_mask = [self.rand_rng.permutation(n_feat) + 1]
        else:
            list_coeff_mask = [np.arange(n_feat) + 1]
        #
        for n_neur in list_neurons:
            # draw mask coefficients
            coeff_mask = self.rand_rng.integers(low = min_connectivity, high = n_feat, size = n_neur)
            list_coeff_mask.append(coeff_mask)
            min_connectivity = coeff_mask.min()
        list_coeff_mask.append(list_coeff_mask[0])
        # create mask matrices
        list_masks = []
        for i, _ in enumerate(list_coeff_mask):
            if i == 0:
                continue
            if i < len(list_coeff_mask) - 1:
                # add (fake) masking for conditionals
                if (i == 1) and (self.n_feat_cond is not None) and (self.n_feat_cond > 0):
                    mask = (list_coeff_mask[i].reshape(-1, 1) >= list_coeff_mask[i-1])
                    mask = np.concatenate((np.ones((mask.shape[0], self.n_feat_cond)), mask), axis = 1).astype(int)
                else:
                    mask = (list_coeff_mask[i].reshape(-1, 1) >= list_coeff_mask[i-1]).astype(int)
                list_masks.append(mask)
            else:
                if n_feat > 1:
                    list_masks.append(np.tile((list_coeff_mask[i].reshape(-1, 1) > list_coeff_mask[i-1]).astype(int), reps = (2, 1)))
                # with only one feature, output is always connected to input
                else:
                    list_masks.append(np.tile((list_coeff_mask[i].reshape(-1, 1) >= list_coeff_mask[i-1]).astype(int), reps = (2, 1)))
        #
        list_coeff_mask = [torch.tensor(coeff_mask) for coeff_mask in list_coeff_mask]
        list_masks = [torch.tensor(mask) for mask in list_masks]
        #
        return list_masks, list_coeff_mask
        
    def forward(self, x: torch.tensor, y: torch.tensor = None) -> Tuple[torch.tensor, torch.tensor]:
        '''
        Function to reconstruct the input with MADE.

        Args:
            x: Input tensor, of shape (batch_size, 1, n_feat).
            y: Conditional tensor, of shape (batch_size, 1, n_feat_cond).

        Returns:
            mu: Estimate of mu.
            alpha: Estimate of alpha.
        '''
        # apply masked linear layers
        x_hat = x
        for i in range(len(self.list_layer)):
            layer = self.list_layer[i]
            #
            mask_concat = getattr(self, f'list_masks_{i}')
            if (i == 0) and (self.n_feat_cond is not None) and (self.n_feat_cond > 0):
                x_hat = torch.cat((y, x_hat), dim = 1)
            #
            x_hat_lin = layer(x = x_hat, mask_concat = mask_concat)
            #
            if i < len(self.list_layer) - 1:
                x_hat = self.relu(x_hat_lin)
            else:
                x_hat = x_hat_lin
        # get mu and alpha
        mu, alpha = x_hat.chunk(chunks = 2, dim = 1)
        #
        return mu, alpha
        
class BatchNorm(nn.Module):
    def __init__(self, n_feat: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        '''
        Implementation of batch normalization.

        Args:
            n_feat: Number of input features.
            eps: Stability factor used in the denominator.
            momentum: Momentum factor: the running quantities are computed as: q_t = (1 - momentum)*q_{t-1} + momentum*Q, where Q is the equivalent of q computed
                      on the current batch of data.            

        Returns: None.
        '''
        super().__init__()
        #
        shape = (1, n_feat)
        #
        self.eps = eps
        self.momentum = momentum
        #
        self.beta = nn.Parameter(torch.zeros(shape))
        self.exp_gamma = nn.Parameter(torch.ones(shape))
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))

    def _perform_batch_norm(self, x: torch.tensor, running_mean: torch.tensor, running_var: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        '''
        Function to perform a batch normalization step.

        Args:
            x: Input tensor.
            running_mean: Current running mean.
            running_var: Current running variance.

        Returns:
            x_norm: Normalized input.
            running_mean: Updated running mean.
            running_var: Updated running variance.
            log_det: Log-determinant of Jacobian coming from batch normalization.
        '''
        #
        beta = self.beta
        exp_gamma = self.exp_gamma
        eps = self.eps
        momentum = self.momentum
        # training phase
        if self.training == True:
            mean = x.mean(dim = 0)
            var = x.var(dim = 0)
            # update running mean and variance
            running_mean = (1 - momentum)*running_mean + momentum*mean
            running_var = (1 - momentum)*running_var + momentum*var
            #
            running_var = torch.clamp(running_var, min = 1e-5, max = 1e5)
        # validation/test phase
        else:
            mean = running_mean
            var = running_var
        # rescale x
        x_norm = (x - mean)*(var + eps)**(-0.5)*exp_gamma + beta
        # update log-determinant of the jacobian
        log_det = torch.sum(torch.log(exp_gamma) - 1/2*torch.log(var + eps))
        log_det = log_det.expand(x.shape[0])
        #
        return x_norm, running_mean, running_var, log_det

    @torch.no_grad()
    def _perform_batch_norm_inv(self, x_norm: torch.tensor) -> torch.tensor:
        '''
        Function to perform a batch normalization inverse step.

        Args:
            x_norm: Normalized tensor.
            running_mean: Current running mean.
            running_var: Current running variance.

        Returns:
            x: Original tensor.
        '''
        #
        beta = self.beta
        exp_gamma = self.exp_gamma
        eps = self.eps
        # 
        mean = self.running_mean
        var = self.running_var
        # rescale back x_norm
        x = (x_norm - beta)*(var + eps)**(0.5)/exp_gamma + mean
        #
        return x

    def forward(self, x) -> Tuple[torch.tensor, torch.tensor]:
        '''
        Forward pass of batch normalization.

        Args:
            x: Input tensor.

        Returns:
            x_norm: Batch-normalized tensor.
            log_det: Log-determinant of Jacobian coming from batch normalization.
        '''
        # make sure running quantities are on the same device as x
        if self.running_mean.device != x.device:
            self.running_mean = self.running_mean.to(x.device)
            self.running_var = self.running_var.to(x.device)
        #
        x_norm, self.running_mean, self.running_var, log_det = self._perform_batch_norm(x = x, running_mean = self.running_mean, running_var = self.running_var)
        #
        return x_norm, log_det
        
    @torch.no_grad()
    def inverse(self, x_norm) -> torch.tensor:
        '''
        Invert batch normalization.

        Args:
            x:norm: Batch-normalized tensor.

        Returns:
            x: Original tensor.
        '''
        x = self._perform_batch_norm_inv(x_norm = x_norm)
        #
        return x

class MAF(nn.Module):
    def __init__(self, n_made: int, n_feat: int, list_neurons: list, n_feat_cond: int = None, random_initial_order: bool = False,
                 seed: int = 123) -> None:
        '''
        Implementation of the MAF model.

        Args:
            n_made: Number of MADE's to be stacked.
            n_feat: Number of input features.
            list_neurons: List of neurons of the hidden layer(s) in each MADE.
            n_feat_cond: Number of features of conditional variables.
            random_initial_order: Whether to initially random permutate features.
            seed: Random seed.

        Returns: None.
        '''
        super().__init__()
        rand_rng = np.random.default_rng(seed)
        #
        list_made, list_batch_norm = [], []
        for i in range(n_made):
            list_made.append(MADE(n_feat = n_feat, list_neurons = list_neurons, n_feat_cond = n_feat_cond, random_initial_order = random_initial_order,
                                  rand_rng = rand_rng))
            list_batch_norm.append(BatchNorm(n_feat = n_feat))
        #
        self.list_made = nn.ModuleList(list_made)
        self.list_batch_norm = nn.ModuleList(list_batch_norm)
        device = next(self.parameters()).device
        self.norm_distr = torch.distributions.MultivariateNormal(loc = torch.zeros(n_feat, device = device),
                                                                 covariance_matrix = torch.eye(n_feat, device = device))
        self.n_feat = n_feat
    
    def forward(self, x: torch.tensor, y: torch.tensor = None) -> Tuple[torch.tensor, torch.tensor]:
        '''
        Function to reconstruct the input with MADE.

        Args:
            x: Input tensor.
            y: Conditional tensor, of shape (batch_size, 1, n_feat_cond).

        Returns:
            u_norm: 'Normalized version' of `x`.
            log_p_x: Log-probability of sample data.
        '''
        n_feat = self.n_feat
        device = x.device
        norm_distr = torch.distributions.MultivariateNormal(loc = torch.zeros(n_feat, device = device), covariance_matrix = torch.eye(n_feat, device = device))
        #
        log_det_total = 0
        #
        for made, batch_norm in zip(self.list_made, self.list_batch_norm):

            ## get mu and alpha
            mu, alpha = made(x = x, y = y)
            u = (x - mu)*torch.exp(-alpha)
            # compute determinant from the transformation
            log_det_trans = -alpha.sum(dim = 1)
            log_det_total += log_det_trans

            ## batch normalization
            u_norm, log_det_batch = batch_norm(x = u)
            log_det_total += log_det_batch
            
            #
            x = u_norm

        ## probability
        log_pi_u = norm_distr.log_prob(u_norm)
        log_p_x = log_det_total + log_pi_u
        #
        return u_norm, log_p_x

    @torch.no_grad()
    def inverse(self, u_norm: torch.tensor) -> torch.tensor:
        '''
        Function to perform inverse transformation.

        Args:
            u_norm: 'Normalized version' of `x`.

        Returns:
            x: Original tensor.
        '''
        for batch_norm, made in zip(reversed(self.list_batch_norm), reversed(self.list_made)):
            # Inverse batch normalization
            u = batch_norm.inverse(x_norm=u_norm)
    
            # Initialize x with zeros
            x = torch.zeros_like(u)
    
            # Reconstruct each feature in order
            for i in torch.argsort(made.list_coeff_mask_0):
                # compute mu and alpha from current x
                mu, alpha = made(x)
                # invert the transformation for dimension i
                x[:, i] = u[:, i] * torch.exp(alpha[:, i]) + mu[:, i]
    
            # pass to next MADE
            u_norm = x
    
        return x
    
class TrainModel:
    def __init__(self, model: torch.nn.Module, dict_params: dict, dataloader_train: DataLoader, dataloader_valid: DataLoader,
                 path_artifacts: str = None, weight_decay: float = 1e-5) -> None: 
        '''
        Class to train the model.

        Args:
            model: PyTorch model.
            dict_params: Dictionary containing information about the model architecture.
            dataloader_train: Dataloader containing training data.
            dataloader_valid: Dataloader containing validation data.
            path_artifacts: Path where to save artifacts.
            weight_decay: Weight decay used by Adam optimizer.

        Returns: None.
        '''
        self.model = model
        self.dict_params = dict_params
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid
        self.path_artifacts = path_artifacts
        #
        self.optimizer = Adam(params = model.parameters(), lr = dict_params['training']['lr'], weight_decay = weight_decay)
        self.scheduler = ReduceLROnPlateau(optimizer = self.optimizer, patience = dict_params['training']['patience_sched'], factor = 0.5)

    def _model_on_batch(self, batch: torch.Tensor, training: bool) -> float:
        '''
        Function to perform training on a single batch of data.

        Args:
            batch: Batch of data to use for training/evaluation.
            training: Whether to perform training (if not, evaluation is understood).
            
        Returns:
            loss: Value of the loss function.
        '''
        model = self.model
        #
        if training == True:
            self.optimizer.zero_grad()
        #
        X = batch.to(next(model.parameters()).device)
        #
        if (model.n_feat_cond is None) or (model.n_feat_cond <= 0):
            _, log_p_x = model(x = X)
        else:
            y = X[:, -model.n_feat_cond:]
            X = X[:, :-model.n_feat_cond]
            _, log_p_x = model(x = X, y = y)
        loss = -log_p_x.to(next(model.parameters()).device).mean()
        #
        if training == True:
            loss.backward()
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
        dataloader = self.dataloader_train
        loss = 0
        self.model.train()
        for batch in dataloader:
                loss += self._model_on_batch(batch, training = True)
        return loss/len(dataloader)

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
                loss += self._model_on_batch(batch, training = False)/len(dataloader)
        return loss/len(dataloader)

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
        #
        val_data = torch.cat([i for i in self.dataloader_valid]).to(next(model.parameters()).device)
        #
        def kl_standard_normal(u: torch.tensor) -> float:
            '''
            Function to compute the KL divergence between a N(mu, sigma) and a N(0, 1) distributions.
            
            Args:
            u: Tensor which comes from a N(mu, sigma) distribution.
                
            Returns:
            kl: KL divergence.
            '''
            mu = u.mean(0)
            # depending on the torch version
            try:
                var = u.var(0, unbiased = False)
            except:
                var = u.var(0)
            #
            kl = 0.5*(mu.pow(2) + var - 1 - var.log()).sum()
            return kl
        #
        for epoch in range(n_epochs):
            loss_train = self._train()
            loss_valid = self._eval()
            #
            model = self.model
            model.eval()
            with torch.no_grad():
                if (model.n_feat_cond is None) or (model.n_feat_cond <= 0):
                    u = model(x = val_data)[0].cpu()
                else:
                    val_data_x = val_data[:, :-model.n_feat_cond:]
                    val_data_y = val_data[:, -model.n_feat_cond:]
                    u = model(x = val_data_x, y = val_data_y)[0].cpu()
            kl = kl_standard_normal(u)
            #
            self.scheduler.step(loss_valid)
            #
            if (len(list_loss_valid) > 0) and (loss_valid >= np.min(list_loss_valid)*(1 - dict_params['training']['min_delta_perc'])):
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
            print(f'Epoch {epoch + 1}: training loss: {loss_train:.7f}, validation loss: {loss_valid:.7f}, KL div = {kl:.7f} (mean = [' +
                ', '.join([f'{i.item():.4f}' for i in u.mean(dim = 0)]) + '], var = [' + 
                ', '.join([f'{i.item():.4f}' for i in u.var(dim = 0)]) +
                f"]), learning rate = {self.optimizer.param_groups[0]['lr']}, counter patience = {counter_patience}.")
            #
            if counter_patience >= dict_params['training']['patience']:
                print(f'Training stopped at epoch {epoch + 1}. Restoring weights from epoch {np.argmin(list_loss_valid) + 1}.')
                break

        dict_artifacts['loss_train'] = list_loss_train
        dict_artifacts['loss_valid'] = list_loss_valid
        if path_artifacts is not None:
            model.load_state_dict(torch.load(path_artifacts))
        return model, dict_artifacts