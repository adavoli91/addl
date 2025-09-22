import numpy as np
import torch
import os
import pickle
import torch.nn.functional as F
from typing import Tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam
from torch.utils.data import DataLoader

class BatchNorm(torch.nn.Module):
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
        self.beta = torch.nn.Parameter(torch.zeros(shape))
        self.exp_gamma = torch.nn.Parameter(torch.ones(shape))
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
            x_norm: Batch-normalized tensor.

        Returns:
            x: Original tensor.
        '''
        x = self._perform_batch_norm_inv(x_norm = x_norm)
        #
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, n_hidden: int) -> None:
        '''
        Class to implement the residual connection used to compute theta.

        Args:
            n_hidden: Number of hidden layers of the linear transformation.

        Returns: None.
        '''
        super().__init__()
        #
        self.linear = torch.nn.Linear(in_features = n_hidden, out_features = n_hidden)
        self.relu = torch.nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        '''
        Function to implement residual connection.

        Args:
            x: Input tensor.

        Returns:
            y: Output tensor.
        '''
        y = self.relu(x + self.linear(x))
        return y

class ResNetConditioner(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int, n_hidden: int, n_blocks: int) -> None:
        '''
        Class to implement the linear model used to compute theta.

        Args:
            dim_in: Input dimension.
            dim_out: Output dimension.
            n_hidden: Number of hidden layers of the linear transformation.
            n_blocks: Number of blocks of the linear transformation.

        Returns: None.
        '''
        super().__init__()
        # residual block used to compute theta
        list_layers = [torch.nn.Linear(in_features = dim_in, out_features = n_hidden), torch.nn.ReLU()]
        for _ in range(n_blocks):
            list_layers.append(ResidualBlock(n_hidden = n_hidden))
        list_layers.append(torch.nn.Linear(in_features = n_hidden, out_features = dim_out))
        #
        self.net = torch.nn.Sequential(*list_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Function to implement theta from the input.

        Args:
            x: Input tensor.

        Returns:
            theta: Output tensor representing theta.
        '''
        theta = self.net(x)
        return theta

class RationalQuadraticSpline(torch.nn.Module):
    def __init__(self, d: int, n_feat: int, K: int, B: int, n_hidden: int, n_blocks: int, n_feat_cond: int, min_dist_knots: float, min_der: float,
                 min_offset: float) -> None:
        '''
        Class to implement the rational quadratic spline computation.

        Args:
            d: Number of features to use as conditionals.
            n_feat: Total number of features.
            K: Number of bins (the number of knots is `K + 1`).
            B: Maximum coordinates of the bin.
            n_hidden: Number of hidden layers of the linear transformation.
            n_blocks: Number of blocks of the linear transformation.
            n_feat_cond: Number of features of conditional variables.
            min_dist_knots: Minimum percentage distance between consecutive knots, if the total range length is 1.
            min_der: Minimum allowed value for the derivative of the spline.
            min_offset: Minimum offset used to regularize denominators.

        Returns: None.
        '''
        super().__init__()
        #
        self.d = d
        self.n_feat = n_feat
        self.K = K
        self.B = B
        self.min_dist_knots = min_dist_knots
        self.min_der = min_der
        self.min_offset = min_offset
        #
        if (n_feat_cond is None) or (n_feat_cond <= 0):
            n_feat_cond = 0
        self.n_feat_cond = n_feat_cond
        #
        self.linear = ResNetConditioner(dim_in = max(d, 1) + n_feat_cond, dim_out = (n_feat - d)*(3*K - 1), n_hidden = n_hidden, n_blocks = n_blocks)

    def _compute_knots(self, theta_width: torch.Tensor, theta_height: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Function to compute width and height knots.

        Args:
            theta_width: Tensor containing the theta's interpreted as width coordinates.
            theta_height: Tensor containing the theta's interpreted as height coordinates.

        Returns:
            knots_width: Knots along the width direction.
            knots_height: Knots along the height direction.
        '''
        K = self.K
        B = float(self.B)
        min_dist_knots = self.min_dist_knots
        # convert bins to normalized bins, defined in the range [0, 1], and add the initial knot
        knots_width = (1 - min_dist_knots*K)*F.softmax(theta_width, dim = -1) + min_dist_knots
        knots_width = knots_width.cumsum(dim = -1)
        knots_width = torch.cat((torch.zeros_like(knots_width[..., :1]), knots_width), dim = -1)
        knots_height = (1 - min_dist_knots*K)*F.softmax(theta_height, dim = -1) + min_dist_knots
        knots_height = knots_height.cumsum(dim = -1)
        knots_height = torch.cat((torch.zeros_like(knots_height[..., :1]), knots_height), dim = -1)
        # convert bins to the range [-B, B]
        knots_width = 2*B*knots_width - B
        knots_height = 2*B*knots_height - B
        # make sure boundaries are correctly fixed, in order to avoid numerical issues
        knots_width[..., 0] = -B
        knots_width[..., -1] =  B
        knots_height[..., 0] = -B
        knots_height[..., -1] =  B
        #
        return knots_width, knots_height
    
    def _get_relevant_bin_der(self, x: torch.Tensor, knots_width: torch.Tensor, knots_height: torch.Tensor, der: torch.Tensor,
                              inverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Function to get the relevant knots and derivatives, for each coordinate of the input tensor.

        Args:
            x: Input tensor used to compute the quadratic spline (or its inverse), of shape (batch, n_feat - d, 1).
            knots_width: Knots along the width direction, of shape (batch, n_feat - d, K + 1).
            knots_height: Knots along the height direction, of shape (batch, n_feat - d, K + 1).
            der: Derivative values at the knots, of shape (batch, n_feat - d, K + 1).
            inverse: Whether performing inverse pass.

        Returns:
            rel_knot_width_below: Relevant lower knot coordinate along width direction for the given input dimension (x^k, in the paper), of shape (batch, n_feat - d, 1).
            rel_knot_width_above: Relevant upper knot coordinate along width direction for the given input dimension (x^{k+1}, in the paper), of shape (batch, n_feat - d, 1).
            rel_knot_height_below: Relevant lower knot coordinate along heigth direction for the given input dimension (y^k, in the paper), of shape (batch, n_feat - d, 1).
            rel_knot_height_above: Relevant upper knot coordinate along heigth direction for the given input dimension (y^{k+1}, in the paper), of shape (batch, n_feat - d, 1).
            rel_der_below: Relevant lower derivative for the given input dimension (delta^k, in the paper), of shape (batch, n_feat - d, 1).
            rel_der_above: Relevant upper derivative for the given input dimension (delta^{k+1}, in the paper), of shape (batch, n_feat - d, 1).
        '''
        K = self.K
        # for input values inside the range [-B, B], find the index of the first knot larger than input
        if inverse == False:
            idx_knot_above = torch.searchsorted(knots_width.contiguous(), x, right = True).clamp(min = 1, max = K)
        else:
            idx_knot_above = torch.searchsorted(knots_height.contiguous(), x, right = True).clamp(min = 1, max = K)
        idx_knot_below = idx_knot_above - 1
        # compute relevant knots for each input values
        rel_knot_width_below = knots_width.gather(dim = 2, index = idx_knot_below)
        rel_knot_width_above = knots_width.gather(dim = 2, index = idx_knot_above)
        rel_knot_height_below = knots_height.gather(dim = 2, index = idx_knot_below)
        rel_knot_height_above = knots_height.gather(dim = 2, index = idx_knot_above)
        rel_der_below = der.gather(dim = 2, index = idx_knot_below)
        rel_der_above = der.gather(dim = 2, index = idx_knot_above)
        #
        return rel_knot_width_below, rel_knot_width_above, rel_knot_height_below, rel_knot_height_above, rel_der_below, rel_der_above
    
    def _compute_spline(self, x_spline: torch.Tensor, rel_knot_width_below: torch.Tensor, rel_knot_width_above: torch.Tensor, rel_knot_height_below: torch.Tensor, rel_knot_height_above: torch.Tensor,
                        rel_der_below: torch.Tensor, rel_der_above: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Function to get the relevant knots and derivatives, for each coordinate of the input tensor.

        Args:
            x_spline: Input tensor used to compute the quadratic spline, of shape (batch, n_feat - d, 1).
            rel_knot_width_below: Relevant lower knot coordinate along width direction for the given input dimension (x^k, in the paper), of shape (batch, n_feat - d, 1).
            rel_knot_width_above: Relevant upper knot coordinate along width direction for the given input dimension (x^{k+1}, in the paper), of shape (batch, n_feat - d, 1).
            rel_knot_height_below: Relevant lower knot coordinate along heigth direction for the given input dimension (y^k, in the paper), of shape (batch, n_feat - d, 1).
            rel_knot_height_above: Relevant upper knot coordinate along heigth direction for the given input dimension (y^{k+1}, in the paper), of shape (batch, n_feat - d, 1).
            rel_der_below: Relevant lower derivative for the given input dimension (delta^k, in the paper), of shape (batch, n_feat - d, 1).
            rel_der_above: Relevant upper derivative for the given input dimension (delta^{k+1}, in the paper), of shape (batch, n_feat - d, 1).

        Returns:
            g: Spline value, of shape (batch, n_feat - d, 1).
            der_g: Derivative value, of shape (batch, n_feat - d, 1).
        '''
        min_offset = self.min_offset
        #
        xi = (x_spline - rel_knot_width_below)/(rel_knot_width_above - rel_knot_width_below + min_offset)
        xi = xi.clamp(min = 0, max = 1)
        s = (rel_knot_height_above - rel_knot_height_below)/(rel_knot_width_above - rel_knot_width_below + min_offset)
        num_g = (rel_knot_height_above - rel_knot_height_below)*(s*xi**2 + rel_der_below*xi*(1 - xi))
        den_g = s + (rel_der_above + rel_der_below - 2*s)*xi*(1 - xi) + min_offset
        num_der_g = s**2*(rel_der_above*xi**2 + 2*s*xi*(1 - xi) + rel_der_below*(1 - xi)**2)
        g = rel_knot_height_below + num_g/den_g
        der_g = num_der_g/den_g**2
        #
        return g, der_g

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Function to compute the rational quadratic spline and the transformation Jacobian.

        Args:
            x: Input tensor, of shape (batch, n_feat).
            y: Conditional tensor, of shape (batch_size, n_feat_cond).

        Returns:
            u: Transformed input, of shape (batch, n_feat).
            log_det: Absolute logarithm of the Jacobian determinant.
        '''
        d = self.d
        n_feat = self.n_feat
        K = self.K
        B = float(self.B)
        n_feat_cond = self.n_feat_cond
        min_der = self.min_der
        min_offset = self.min_offset
        # split the input in the part used to compute theta and in that used to compute the spline
        x_theta = x[:, :d]
        x_spline = x[:, d:].unsqueeze(dim = -1)
        # compute theta and split it in components
        if self.d > 0:
            # include conditionals
            if (n_feat_cond > 0) and (y is not None):
                x_theta = torch.cat((y, x_theta), dim = -1)
            theta = self.linear(x_theta)
        else:
            if (n_feat_cond <= 0) or (y is None):
                theta = self.linear(torch.ones(x.shape[0], 1))
            # include conditionals
            else:
                theta = self.linear(torch.cat((y, torch.ones(x.shape[0], 1)), dim = -1))
        theta = theta.view(-1, n_feat - d, 3*K - 1)
        theta_width = theta[..., :K]
        theta_height = theta[..., K: 2*K]
        theta_der = theta[..., 2*K:]
        # get bin knots
        knots_width, knots_height = self._compute_knots(theta_width = theta_width, theta_height = theta_height)
        # make derivative strictly positive
        der = F.softplus(theta_der) + min_der
        der = torch.cat((torch.ones_like(der[..., :1]), der, torch.ones_like(der[..., :1])), dim = -1)
        # get relevant knots and derivative
        (rel_knot_width_below, rel_knot_width_above, rel_knot_height_below, rel_knot_height_above, rel_der_below,
         rel_der_above) = self._get_relevant_bin_der(x = x_spline, knots_width = knots_width, knots_height = knots_height, der = der)
        # compute spline and its derivative
        g, der_g = self._compute_spline(x_spline = x_spline, rel_knot_width_below = rel_knot_width_below, rel_knot_width_above = rel_knot_width_above, rel_knot_height_below = rel_knot_height_below,
                                        rel_knot_height_above = rel_knot_height_above, rel_der_below = rel_der_below, rel_der_above = rel_der_above)
        # check whether the input values are inside the range [-B, B]
        is_in_range = (x_spline >= -B) & (x_spline <= B)
        # compute logarithm of the absolute jacobian
        der_g = torch.where(is_in_range, der_g, torch.ones_like(der_g))
        log_det = torch.log(der_g.clamp_min(min_offset)).sum(dim = (1, 2))
        # define `u` as identity outside the range [-B, B]
        u = torch.where(is_in_range, g, x_spline).squeeze(-1)
        # concatenate with untransformed input data
        u = torch.cat((x_theta, u), dim = 1)
        #
        return u, log_det
    
    def _invert_spline(self, u_spline: torch.Tensor, rel_knot_width_below: torch.Tensor, rel_knot_width_above: torch.Tensor, rel_knot_height_below: torch.Tensor, rel_knot_height_above: torch.Tensor,
                        rel_der_below: torch.Tensor, rel_der_above: torch.Tensor) -> torch.Tensor:
        '''
        Function to invert the computation of the quadratic spline.

        Args:
            u_spline: Quadratic spline, of shape (batch, n_feat - d, 1).
            rel_knot_width_below: Relevant lower knot coordinate along width direction for the given input dimension (x^k, in the paper), of shape (batch, n_feat - d, 1).
            rel_knot_width_above: Relevant upper knot coordinate along width direction for the given input dimension (x^{k+1}, in the paper), of shape (batch, n_feat - d, 1).
            rel_knot_height_below: Relevant lower knot coordinate along heigth direction for the given input dimension (y^k, in the paper), of shape (batch, n_feat - d, 1).
            rel_knot_height_above: Relevant upper knot coordinate along heigth direction for the given input dimension (y^{k+1}, in the paper), of shape (batch, n_feat - d, 1).
            rel_der_below: Relevant lower derivative for the given input dimension (delta^k, in the paper), of shape (batch, n_feat - d, 1).
            rel_der_above: Relevant upper derivative for the given input dimension (delta^{k+1}, in the paper), of shape (batch, n_feat - d, 1).

        Returns:
            x_spline: Original, inverse-transformed, tensor, of shape (batch, n_feat - d, 1).
        '''
        min_offset = self.min_offset
        #
        s = (rel_knot_height_above - rel_knot_height_below)/(rel_knot_width_above - rel_knot_width_below + min_offset)
        #
        a = (rel_knot_height_above - rel_knot_height_below)*(s - rel_der_below) + (u_spline - rel_knot_height_below)*(rel_der_above + rel_der_below - 2*s)
        b = (rel_knot_height_above - rel_knot_height_below)*rel_der_below - (u_spline - rel_knot_height_below)*(rel_der_above + rel_der_below - 2*s)
        c = -s*(u_spline - rel_knot_height_below)
        #
        xi = -2*c/(b + torch.sqrt(torch.clamp(b**2 - 4*a*c, min = 0)) + min_offset)
        xi = xi.clamp(min = 0, max = 1)
        #
        x_spline = xi*(rel_knot_width_above - rel_knot_width_below) + rel_knot_width_below
        #
        return x_spline
    
    @torch.no_grad()
    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        '''
        Function to invert the quadratic spline transformation.

        Args:
            u: Spline tensor, of shape (batch, n_feat).

        Returns:
            x: Original, inverse-transformed, tensor, of shape (batch, n_feat).
        '''
        d = self.d
        n_feat = self.n_feat
        K = self.K
        B = float(self.B)
        min_der = self.min_der
        # split the input in the part used to compute theta and in that used to compute the spline
        u_theta = u[:, :d]
        u_spline = u[:, d:].unsqueeze(dim = -1)
        # compute theta and split it in components
        if self.d > 0:
            theta = self.linear(u_theta)
        else:
            theta = self.linear(torch.ones(u.shape[0], 1))
        theta = theta.view(-1, n_feat - d, 3*K - 1)
        theta_width = theta[..., :K]
        theta_height = theta[..., K: 2*K]
        theta_der = theta[..., 2*K:]
        # get bin knots
        knots_width, knots_height = self._compute_knots(theta_width = theta_width, theta_height = theta_height)
        # make derivative strictly positive
        der = F.softplus(theta_der) + min_der
        der = torch.cat((torch.ones_like(der[..., :1]), der, torch.ones_like(der[..., :1])), dim = -1)
        # get relevant knots and derivative
        (rel_knot_width_below, rel_knot_width_above, rel_knot_height_below, rel_knot_height_above, rel_der_below,
         rel_der_above) = self._get_relevant_bin_der(x = u_spline, knots_width = knots_width, knots_height = knots_height, der = der, inverse = True)
        # compute spline and its derivative
        x_spline = self._invert_spline(u_spline = u_spline, rel_knot_width_below = rel_knot_width_below, rel_knot_width_above = rel_knot_width_above, rel_knot_height_below = rel_knot_height_below,
                                        rel_knot_height_above = rel_knot_height_above, rel_der_below = rel_der_below, rel_der_above = rel_der_above)
        # check whether the input values are inside the range [-B, B]
        is_in_range = (u_spline >= -B) & (u_spline <= B)
        # define `x` as identity outside the range [-B, B]
        x = torch.where(is_in_range, x_spline, u_spline).squeeze(-1)
        # concatenate with untransformed input data
        x = torch.cat((u_theta, x), dim = 1)
        #
        return x

class LinearInvertibleTransform(torch.nn.Module):
    def __init__(self, n_feat: int, seed: int) -> None:
        '''
        Class to implement the linear invertible transformation.

        Args:
            n_feat: Total number of features.
            seed: Random seed.

        Returns: None.
        '''
        rand_gen = np.random.default_rng(seed = seed)
        #
        super().__init__()
        self.n_feat = n_feat
        self.register_buffer('mat_perm', torch.eye(n_feat)[rand_gen.permutation(range(n_feat))])
        self.params_l = torch.nn.Parameter(torch.zeros(int(n_feat*(n_feat - 1)/2)))
        self.params_u_diag = torch.nn.Parameter(torch.ones(n_feat))
        self.params_u_off_diag = torch.nn.Parameter(torch.zeros(int(n_feat*(n_feat - 1)/2)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Function to implement the linear invertible transformation.

        Args:
            x: Input tensor.

        Returns:
            y: Transformed input.
            log_det: Absolute logarithm of the Jacobian determinant.
        '''
        n_feat = self.n_feat
        mat_perm = self.mat_perm
        #
        mat_l = torch.eye(n_feat).to(x.device)
        mask = torch.ones_like(mat_l, dtype = torch.bool).tril(diagonal = -1)
        mat_l[mask] = self.params_l
        #
        mat_u = torch.zeros(n_feat, n_feat).to(x.device)
        mask = torch.ones_like(mat_u, dtype = torch.bool).triu(diagonal = 1)
        diag_u = torch.nn.functional.softplus(self.params_u_diag)
        mat_u[range(n_feat), range(n_feat)] = diag_u
        mat_u[mask] = self.params_u_off_diag
        #
        mat_w = mat_perm@mat_l@mat_u
        y = x@mat_w.T
        #
        log_det = torch.sum(torch.log(diag_u))
        log_det = log_det.unsqueeze(0).repeat(x.shape[0])
        #
        return y, log_det

    @torch.no_grad()
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        '''
        Function to invert the linear invertible transformation.

        Args:
            y: Transformed tensor.

        Returns:
            x: Input input.
        '''
        n_feat = self.n_feat
        mat_perm = self.mat_perm
        #
        mat_l = torch.eye(n_feat).to(y.device)
        mask = torch.ones_like(mat_l, dtype = torch.bool).tril(diagonal = -1)
        mat_l[mask] = self.params_l
        #
        mat_u = torch.zeros(n_feat, n_feat).to(y.device)
        mask = torch.ones_like(mat_u, dtype = torch.bool).triu(diagonal = 1)
        diag_u = torch.nn.functional.softplus(self.params_u_diag)
        mat_u[range(n_feat), range(n_feat)] = diag_u
        mat_u[mask] = self.params_u_off_diag
        #
        mat_w = mat_perm@mat_l@mat_u
        mat_w_inv = torch.linalg.inv(mat_w)
        x = y@mat_w_inv.T
        #
        return x
    
class NSF(torch.nn.Module):
    def __init__(self, d: int, n_feat: int, K: int, B: int, device: str, n_step: int = 2, n_hidden: int = 100, n_blocks: int = 1, n_feat_cond: int = None,
                 min_dist_knots: float = 1e-4, min_der: float = 1e-6, min_offset: float = 1e-6, seed: int = 123) -> None:
        '''
        Class to implement NSF.

        Args:
            d: Number of features not to be transformed.
            n_feat: Total number of features.
            K: Number of bins (the number of knots is `K + 1`).
            B: Maximum coordinates of the bin.
            device: Device where to save the model.
            n_step: Number of steps in the flow.
            n_hidden: Number of hidden layers of the linear transformation.
            n_blocks: Number of blocks of the linear transformation.
            n_feat_cond: Number of features of conditional variables.
            min_dist_knots: Minimum percentage distance between consecutive knots, if the total range length is 1.
            min_der: Minimum allowed value for the derivative of the spline.
            min_offset: Minimum offset used to regularize denominators.
            seed: Random seed.

        Returns: None.
        '''
        super().__init__()
        #
        if d >= n_feat:
            print(f"Parameter 'd' must be smaller than 'n_feat'; automatically set to {n_feat - 1}.")
            d = n_feat - 1
        #
        list_spline, list_batch_norm, list_lin_inv_trans = [], [], []
        for i in range(n_step):
            list_spline.append(RationalQuadraticSpline(d = d, n_feat = n_feat, K = K, B = B, n_hidden = n_hidden, n_blocks = n_blocks,
                                                       n_feat_cond = n_feat_cond, min_dist_knots = min_dist_knots, min_der = min_der,
                                                       min_offset = min_offset))
            list_batch_norm.append(BatchNorm(n_feat = n_feat))
            list_lin_inv_trans.append(LinearInvertibleTransform(n_feat = n_feat, seed = seed))
        #
        self.list_spline = torch.nn.ModuleList(list_spline)
        self.list_batch_norm = torch.nn.ModuleList(list_batch_norm)
        self.list_lin_inv_trans = torch.nn.ModuleList(list_lin_inv_trans)
        #
        self.n_feat = n_feat
        self.device = device
        self.n_feat_cond = n_feat_cond

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Function to implement NSF.

        Args:
            x: Input tensor, of shape (batch, n_feat).
            y: Conditional tensor, of shape (batch_size, n_feat_cond).

        Returns:
            u: 'Normalized version' of `x`.
            log_p_x: Log-probability of sample data.
        '''
        n_feat = self.n_feat
        device = self.device
        norm_distr = torch.distributions.MultivariateNormal(loc = torch.zeros(n_feat, device = device), covariance_matrix = torch.eye(n_feat, device = device))
        #
        log_det = 0
        for layer_spline, layer_batch_norm, layer_lin_inv_trans in zip(self.list_spline, self.list_batch_norm, self.list_lin_inv_trans):
            # compute rational quadratic spline
            u, log_det_spline = layer_spline(x = x, y = y)
            # perform batch normalization
            u_norm, log_det_batch = layer_batch_norm(x = u)
            # perform linear invertible transform
            u_norm, log_det_lin_inv_trans = layer_lin_inv_trans(u_norm)
            # update log determinant
            log_det += log_det_spline + log_det_batch + log_det_lin_inv_trans
            #
            x = u_norm
        ## probability
        log_pi_u = norm_distr.log_prob(x)
        log_p_x = log_det + log_pi_u
        #
        u = u_norm
        return u, log_p_x

    @torch.no_grad()
    def inverse(self, u_norm: torch.Tensor) -> torch.Tensor:
        '''
        Function to implement the inverse NSF transformation.

        Args:
            u_norm: 'Normalized version' of `x`, of shape (batch, n_feat).

        Returns:
            x: Original tensor.
        '''
        for layer_spline, layer_batch_norm, layer_lin_inv_trans in zip(reversed(self.list_spline), reversed(self.list_batch_norm), reversed(self.list_lin_inv_trans)):
            # invert linear invertible transform
            u_norm = layer_lin_inv_trans.inverse(y = u_norm)
            # invert batch normalization
            u = layer_batch_norm.inverse(x_norm = u_norm)
            # invert spline computation
            x = layer_spline.inverse(u = u)
            #
            u_norm = x
        return x
    
class TrainModel:
    def __init__(self, model: torch.nn.Module, dict_params: dict, dataloader_train: DataLoader, dataloader_valid: DataLoader, path_artifacts: str = None)-> None: 
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
        self.optimizer = Adam(params = model.parameters(), lr = dict_params['training']['lr'])
        self.scheduler = ReduceLROnPlateau(optimizer = self.optimizer, patience = dict_params['training']['patience_sched'], factor = 0.5)

    def _model_on_batch(self, batch: torch.Tensor, training: bool, loss_epoch: float) -> float:
        '''
        Function to perform training on a single batch of data.

        Args:
            batch: Batch of data to use for training/evaluation.
            training: Whether to perform training (if not, evaluation is understood).
            loss_epoch: Loss of the current epoch.
            
        Returns:
            loss: Value of the loss function.
        '''
        #
        if training == True:
            self.optimizer.zero_grad()
        #
        X = batch.to(next(self.model.parameters()).device)
        #
        if (self.model.n_feat_cond is None) or (self.model.n_feat_cond <= 0):
            _, log_p_x = self.model(x = X)
        else:
            y = X[:, -self.model.n_feat_cond:]
            X = X[:, :-self.model.n_feat_cond]
            _, log_p_x = self.model(x = X, y = y)
        loss = -log_p_x.to(next(self.model.parameters()).device).mean()
        #
        if training == True:
            loss.backward()
            self.optimizer.step()
        #
        loss_epoch += loss.item()
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
        #
        val_data = torch.cat([i for i in self.dataloader_valid]).to(next(self.model.parameters()).device)
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
                if (self.model.n_feat_cond is None) or (self.model.n_feat_cond <= 0):
                    u = model(x = val_data)[0].cpu()
                else:
                    val_data_x = val_data[:, :-self.model.n_feat_cond:]
                    val_data_y = val_data[:, -self.model.n_feat_cond:]
                    u = model(x = val_data_x, y = val_data_y)[0].cpu()
            kl = kl_standard_normal(u)
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
            print(f'Epoch {epoch + 1}: training loss: {loss_train:.7f}, validation loss: {loss_valid:.7f}, KL div = {kl:.7f} (mean = [' +
                ', '.join([f'{i.item():.4f}' for i in u.mean(dim = 0)]) + '], var = [' + 
                ', '.join([f'{i.item():.4f}' for i in u.var(dim = 0)]) +
                f"]), learning rate = {self.optimizer.param_groups[0]['lr']}, counter patience = {counter_patience}.")
            #
            if counter_patience >= dict_params['training']['patience']:
                break

        dict_artifacts['loss_train'] = list_loss_train
        dict_artifacts['loss_valid'] = list_loss_valid
        return self.model, dict_artifacts