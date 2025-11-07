import numpy as np
import torch
import warnings
from typing import Tuple
from torch.nn import Conv1d, ConvTranspose1d, BatchNorm1d, ReLU, Parameter, Dropout

class Time2Vec(torch.nn.Module):
    def __init__(self, dim_embed: int) -> None:
        '''
            Class to implement Time2Vec
            
            Args:
            dim_embed: Time embedding dimension

            Returns: None.
        '''
        super().__init__()
        self.omega = Parameter(torch.randn(1, dim_embed))
        self.phi = Parameter(torch.randn(1, dim_embed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Function to get time embedding.

        Args:
            x: Input tensor, of shape (batch_size, sequence_length, 1).
            
        Returns:
            t: Time embedding, of shape (batch_size, sequence_length, dim_embed).
        '''
        t = torch.matmul(x, self.omega) + self.phi
        t = torch.concat((t[:, :, :1], torch.sin(t[:, :, 1:])), dim = -1)
        return t

class Encoder(torch.nn.Module):
    def __init__(self, n_feat: int, list_filters: list, list_filter_size: list, list_stride: list, list_padding: list, list_dilation: list,
                 dropout: float) -> None:
        '''
        Class to implement the convolutional encoder.

        Args:
            n_feat: Number of input features.
            list_filters: List containing the number of filters for different layers.
            list_filter_size: List containing the filter sizes for different layers.
            list_stride: List containing the strides for different layers.
            list_padding: List containing the paddings for different layers.
            list_dilation: List containing the dilations for different layers.
            dropout: Dropout probability.

        Returns: None.
        '''
        super().__init__()
        # set stride and padding to default values
        if (list_stride is None) or (len(list_stride) == 0):
            list_stride = [1]*len(list_filters)
        if (list_padding is None) or (len(list_padding) == 0):
            list_padding = [0]*len(list_filters)
        #
        list_layers = []
        for i in range(len(list_filters)):
            list_layers.append(Conv1d(in_channels = n_feat if i == 0 else list_filters[i - 1], out_channels = list_filters[i],
                                      kernel_size = list_filter_size[i], stride = list_stride[i], padding = list_padding[i], dilation = list_dilation[i]))
            list_layers.append(BatchNorm1d(num_features = list_filters[i]))
            if i < len(list_filters) - 1:
                list_layers.append(ReLU())
                list_layers.append(Dropout(p = dropout))
        self.list_layers = torch.nn.ModuleList(list_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Function to encode the input.

        Args:
            x: Input tensor.

        Returns:
            x_enc: Encoded input.
        '''
        x_enc = x
        for layer in self.list_layers:
            x_enc = layer(x_enc)
        return x_enc
    
class Decoder(torch.nn.Module):
    def __init__(self, n_feat: int, len_win: int, list_conv: list, dropout: float) -> None:
        '''
        Class to implement the convolutional decoder.

        Args:
            n_feat: Number of input features.
            len_win: Sequence length of the input of the autoencoder.
            list_conv: List of Conv1D layers of the encoder.
            dropout: Dropout probability.

        Returns: None.
        '''
        super().__init__()
        list_len_output = self._compute_len_output(len_win = len_win, list_conv = list_conv)
        # reverse lists
        list_conv = list_conv[::-1]
        list_len_output = list_len_output[::-1]
        list_len_output.append(len_win)
        #
        list_layers = []
        for i, conv in enumerate(list_conv):
            conv = list_conv[i]
            offset = self._compute_output_padding(len_input = list_len_output[i], conv = conv, len_target = list_len_output[i+1])
            list_layers.append(ConvTranspose1d(in_channels = conv.out_channels,
                                                out_channels = conv.in_channels if (i < len(list_conv) - 1) else n_feat,
                                                kernel_size = conv.kernel_size[0], stride = conv.stride[0], padding = conv.padding[0],
                                                dilation = conv.dilation[0], output_padding = offset))
            if i < len(list_conv) - 1:
                list_layers.append(BatchNorm1d(num_features = conv.in_channels))
                list_layers.append(ReLU())
                list_layers.append(Dropout(p = dropout))
        self.list_layers = torch.nn.ModuleList(list_layers)

    def _compute_len_output(self, len_win: int, list_conv: list) -> list:
        '''
        Function to compute the list of the sequence lengths of the outputs of encoder Conv1D layers.

        Args:
            len_win: Sequence length of the encoder input.
            list_conv: List of encoder Conv1D layers.

        Returns:
            list_len_output: List of the sequence lengths of the outputs of encoder Conv1D layers.
        '''
        list_len_output = []
        len_input = len_win
        for conv in list_conv:
            k = conv.kernel_size[0]
            s = conv.stride[0]
            p = conv.padding[0]
            d = conv.dilation[0]
            #
            len_output = int(np.floor((len_input + 2*p - d*(k-1) - 1)/s + 1))
            #
            list_len_output.append(len_output)
            len_input = len_output
        return list_len_output

    def _compute_output_padding(self, len_input: int, conv: torch.nn.Module, len_target: int) -> int:
        '''
        Function to determine the offset to be used in the ConvTranspose1D layer to ensure the correct sequence length of the output.

        Args:
            len_input: Sequence length of the encoder input.
            conv: Encoder Conv1D layers.
            len_target: Target sequence length, i.e. sequence length the decoder output should have.

        Returns:
            offset: Offset to be used in the ConvTranspose1D layer to ensure the correct sequence length of the output.
        '''
        k = conv.kernel_size[0]
        s = conv.stride[0]
        p = conv.padding[0]
        d = conv.dilation[0]
        #
        len_output = (len_input - 1)*s - 2*p + d*(k - 1) + 1
        #
        offset = len_target - len_output
        return offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Function to decode the input.

        Args:
            x: Input tensor.

        Returns:
            x_dec: Decoded input.
        '''
        x_dec = x
        for layer in self.list_layers:
            x_dec = layer(x_dec)
        return x_dec

class Autoencoder(torch.nn.Module):
    def __init__(self, n_feat: int, len_win: int, list_filters: list, list_filter_size: list, list_stride: list = [],
                 list_padding: list = [], list_dilation: list = [], dropout: float = 0.2, dim_embed_time2vec: int = None) -> None:
        '''
        Class to implement the convolutional autoencoder.

        Args:
            dim_embed: Time embedding dimension.
            n_feat: Number of input features.
            len_win: Sequence length of the input of the autoencoder.
            list_filters: List containing the number of filters for different layers.
            list_filter_size: List containing the filter sizes for different layers.
            list_stride: List containing the strides for different layers.
            list_padding: List containing the paddings for different layers.
            list_dilation: List containing the dilations for different layers.
            dropout: Dropout probability.
            dim_embed_time2vec: Time embedding with time2vec.

        Returns: None.
        '''
        super().__init__()
        # check list lengths
        if len(list_filters) != len(list_filter_size):
            raise ValueError('`list_filters` and `list_filter_size` should have the same length.')
        if (type(list_stride) == list) and (len(list_stride) > 0) and (len(list_stride) != len(list_filters)):
            raise ValueError('`list_stride` and `list_filters` should have the same length.')
        if (type(list_padding) == list) and (len(list_padding) > 0) and (len(list_padding) != len(list_filters)):
            raise ValueError('`list_padding` and `list_filters` should have the same length.')
        if (type(list_dilation) == list) and (len(list_dilation) > 0) and (len(list_dilation) != len(list_filters)):
            raise ValueError('`list_dilation` and `list_filters` should have the same length.')
        #
        if (type(list_stride) == list) and (len(list_stride) == 0):
            list_stride = [1]*len(len(list_filters))
            warnings.warn(f'`list_stride` was not provided; it was automatically set to {list_stride}.')
        if (type(list_padding) == list) and (len(list_padding) == 0):
            list_padding = [0]*len(len(list_filters))
            warnings.warn(f'`list_padding` was not provided; it was automatically set to {list_padding}.')
        if (type(list_dilation) == list) and (len(list_dilation) == 0):
            list_dilation = [1]*len(len(list_dilation))
            warnings.warn(f'`list_dilation` was not provided; it was automatically set to {list_dilation}.')
        #
        self.n_feat = n_feat
        self.dim_embed_time2vec = dim_embed_time2vec
        #
        dim_time_embed = 0
        if dim_embed_time2vec > 0:
            self.time_to_vec = Time2Vec(dim_embed = dim_embed_time2vec)
            dim_time_embed += dim_embed_time2vec
        self.encoder = Encoder(n_feat = n_feat + dim_time_embed, list_filters = list_filters, list_filter_size = list_filter_size,
                               list_stride = list_stride, list_padding = list_padding, list_dilation = list_dilation, dropout = dropout)
        list_conv = [i for i in self.encoder.list_layers if type(i) == torch.nn.Conv1d]
        self.decoder = Decoder(n_feat = n_feat + dim_time_embed, len_win = len_win, list_conv = list_conv, dropout = dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Function to decode the input.

        Args:
            x: Input tensor, of shape (batch, n_feat, len_win).

        Returns:
            x_hat: Reconstructed input.
        '''
        if self.dim_embed_time2vec > 0:
            # apply time2vec
            t = self.time_to_vec(x.transpose(1, 2)[:, :, :1]).transpose(1, 2)
            # concatenate data and time embedding
            x = torch.cat((x, t), dim = 1)
        # encode input
        x_enc = self.encoder(x)
        # decode encoded input
        x_hat = self.decoder(x_enc)
        #
        return x_hat[:, :self.n_feat, :]

class TrainModel:
    def __init__(self, model: torch.nn.Module, dict_params: dict, dataloader_train: torch.utils.data.DataLoader, dataloader_valid: torch.utils.data.DataLoader,
                 path_artifacts: str, weight_decay: float = 1e-5) -> None:
        '''
        Class to train a pytorch model.
        
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
        self.path_artifacts = path_artifacts
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params = model.parameters(), lr = dict_params['training']['lr'], weight_decay = weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer, factor = 0.5)
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid

    def _model_on_batch(self, batch: tuple, training: bool) -> float:
        '''
        Function to perform training on a single batch of data.
        
        Args:
            batch: Batch of data to use for training/evaluation.
            training: Whether to perform training (if not, evaluation is understood).
            
        Returns:
            loss: Value of the loss function.
        '''
        if training == True:
            self.optimizer.zero_grad()
        #
        x = batch.to('cpu')
        x_hat = self.model(x).to('cpu')
        #
        loss = self.loss_func(x_hat, x)
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
            dict_artifacts: Dictionary containing the lists of training and validation losses and model weights.
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
        for epoch in range(n_epochs):
            loss_train = self._train()
            loss_valid = self._eval()
            #
            self.scheduler.step(loss_valid)
            #
            if (len(list_loss_valid) > 0) and (loss_valid >= np.min(list_loss_valid)*(1 - dict_params['training']['min_delta_loss_perc'])):
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
            print(f'Epoch {epoch + 1}: training loss: {loss_train:.7f}, validation loss: {loss_valid:.7f}, ' +
                f"learning rate = {self.optimizer.param_groups[0]['lr']}, counter patience = {counter_patience}.")
            #
            if counter_patience >= dict_params['training']['patience']:
                print(f'Training stopped at epoch {epoch + 1}. Restoring weights from epoch {np.argmin(list_loss_valid) + 1}.')
                break

        dict_artifacts['loss_train'] = list_loss_train
        dict_artifacts['loss_valid'] = list_loss_valid
        
        if path_artifacts is not None:
            model.load_state_dict(torch.load(path_artifacts))
        return model, dict_artifacts