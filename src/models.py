import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
from typing import Dict, Tuple
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl


def apply_scaled_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):
    def __init__(self,
                 z_size: int,
                 num_input_channels: int,
                 base_channel_size: int,
                 condition_size: int=1):
        """
        Args:
           num_input_channels : Number of input channels of the image.
           base_channel_size : Number of channels we use in the first convolutional layers. 
           z_size : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.z_size = z_size
        self.condition_size = condition_size
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 64x64 => 32x32
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            nn.GELU(),
            nn.Flatten(),  # Image grid to single feature vector
        )
        apply_scaled_init(self.net)

        self.fc_mu = nn.Linear(2 * 8 * 8 * c_hid, self.condition_size * z_size)
        self.fc_log_var = nn.Linear(2 * 8 * 8 * c_hid, self.condition_size * z_size)
        apply_scaled_init(self.fc_mu)
        apply_scaled_init(self.fc_log_var)

    def forward(self, x, condition):
        x = self.net(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        if condition is not None:
            batch_size = len(x)
            mu = mu.reshape(batch_size, self.condition_size, self.z_size)
            mu = mu[torch.arange(batch_size), condition]
            log_var = log_var.reshape(batch_size, self.condition_size, self.z_size)
            log_var = log_var[torch.arange(batch_size), condition]
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self,
                 z_size: int,
                 num_input_channels: int,
                 base_channel_size: int):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct.
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           z_size : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(3 * z_size, 2 * 8 * 8 * c_hid),
            nn.GELU(),
            nn.Unflatten(1, (-1, 8, 8)),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 8x8 => 16x16
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 32x32 => 64x64
            nn.Tanh(),
        )
        apply_scaled_init(self.linear)
        apply_scaled_init(self.net)

    def forward(self, x):
        x = self.linear(x)
        x = self.net(x)
        return x


class BaseModel(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        optimizer_param: dict,
        latent_dim: int=32,
        base_channel_size: int=32,
        num_input_channels: int = 4,
        image_size: int = 64,
        *args,
        **kwargs):
        super().__init__()
        
        self.model_name = model_name
        self.num_input_channels = num_input_channels
        self.width = image_size
        self.height = image_size
        self.base_channel_size = base_channel_size
        self.latent_dim = latent_dim
        self.network_param = {'latent_dim': latent_dim, 'num_input_channels':num_input_channels,
                              'base_channel_size':base_channel_size}

        # Example input array needed for visualizing the graph of the network
        self.optimizer_param = optimizer_param

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def _get_loss(self, x):
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss
    
    def configure_optimizers(self):
        lr = self.optimizer_param['lr']
        if self.optimizer_param['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif self.optimizer_param['optimizer'] == 'SGD':
            momentum = self.optimizer_param['momentum']
            nesterov = self.optimizer_param['nesterov']
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "monitor": "train_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("train_loss", loss.detach())
        while True:  # Inside your training loop
            current_memory_allocated = torch.cuda.memory_allocated() / (1024.0 ** 3)  # Convert bytes to GB
            max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 3)  # Convert bytes to GB
            self.log_dict({'Current GPU Memory (GB)': current_memory_allocated, 'Max GPU Memory (GB)': max_memory_allocated})
            return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("val_loss", loss.detach())

    def test_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("test_loss", loss.detach())


class VAEmodel(BaseModel):
    def __init__(self, 
                 step_size: int,
                 latent_dim = 64,
                 *args,
                 **kwargs):
        super().__init__(latent_dim=latent_dim, *args, **kwargs)
        
        # Initialize the cyclic weight scheduler
        self.kl_weight_scheduler = CyclicWeightScheduler(step_size=step_size)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.encoder = Encoder(**self.network_param)
        self.decoder = Decoder(**self.network_param)
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
    
    def forward(self, x):
        # encode x to get the mu and variance parameters
        mu, log_var = self.encoder(x)
        # sample z 
        log_var = torch.maximum(log_var, torch.tensor(-20)) #clipping to prevent going to -inf
        std = torch.exp(log_var / 2)
        z = self.sampling(mu, std)
        # decoded 
        x_hat = self.decoder(z)
        return x_hat, mu, std
    
    def get_image_embedding(self, x):
        mu, _ = self.encoder(x)
        return mu
    
    @staticmethod
    def sampling(mu, std):
        q = Normal(mu, std)
        return q.rsample()

    @staticmethod
    def reconstruction_loss(x: torch.Tensor,
                            x_pred: torch.Tensor,
                            logsd: torch.Tensor,
                            ):
        sd = torch.exp(logsd)
        dist_x = Normal(x_pred, sd)
        loss_x = -dist_x.log_prob(x).sum(dim=(1, 2, 3))
        return loss_x

    def _generic_loss(self, x, x_hat, mu, std):
        # reconstruction probability
        recon_loss = self.reconstruction_loss(x, x_hat, self.log_scale)

        # kl
        kl = self.latent_kl_divergence(mu, std)
        kl_term_weight = self.kl_weight_scheduler.step()

        # elbo
        elbo = (kl_term_weight*kl + recon_loss)
        elbo = elbo.mean()
        self.log_dict({
            'elbo': elbo.detach(),
            'kl': kl.mean().detach(),
            'recon_loss': recon_loss.mean().detach(), 
            'kl_term_weight': kl_term_weight,
        })
        return elbo

    @staticmethod
    def latent_kl_divergence(variational_mean, 
                             variational_std, 
                             prior_mean=None,
                             prior_std=None) -> torch.Tensor:
        """
        Compute KL divergence between a variational posterior and standard Gaussian prior.
        Args:
        ----
            variational_mean: Mean of the variational posterior Gaussian.
            variational_var: Variance of the variational posterior Gaussian.
        Returns
        -------
            KL divergence for each data point. If number of latent samples == 1,
            the tensor has shape `(batch_size, )`. If number of latent
            samples > 1, the tensor has shape `(n_samples, batch_size)`.
        """
        if prior_mean is None:
            prior_mean = torch.zeros_like(variational_mean)
        prior_std = torch.ones_like(variational_std)
        return kl(
            Normal(variational_mean, variational_std),
            Normal(prior_mean, prior_std),
        ).sum(dim=-1)
    
    def _get_loss(self, x):
        # get reconstruction, mu and std
        x_hat, mu, std = self.forward(x)
        elbo = self._generic_loss(x, x_hat, mu, std)
        return elbo


class ContrastiveVAEmodel(BaseModel):
    """
    Args:
    ----
        n_z_latent: Dimensionality of the background latent space.
        n_s_latent: Dimensionality of the salient latent space.
        wasserstein_penalty: Weight of the Wasserstein distance loss that further
            discourages shared variations from leaking into the salient latent space.
    """
        
    def __init__(self,
                 step_size: float = 2000,
                 z_size: int = 32,
                 y_size: int = 5050,
                 e_size: int = 34,
                 wasserstein_penalty: float = 0,
                 batch_size: int=1024,
                 tc_penalty: float=1,
                 classification_weight: float=1,
                 scale_factor: float=0.1,
                 max_kl_weight: float=1,
                 reg_type: str=None,
                 total_steps: int=3000,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.step_size = step_size
        self.y_size = y_size
        self.e_size = e_size
        self.z_size = z_size
        self.wasserstein_penalty = wasserstein_penalty
        self.batch_size = batch_size
        self.tc_penalty = tc_penalty
        self.classification_weight = classification_weight
        self.scale_factor = scale_factor
        self.reg_type = reg_type
        self.total_steps = total_steps
        self.kl_weight_scheduler = CyclicWeightScheduler(step_size=self.step_size, max_weight=max_kl_weight)

        self.qy = Encoder(self.z_size, self.num_input_channels, self.base_channel_size)
        self.qx = Encoder(self.z_size, self.num_input_channels, self.base_channel_size)
        self.qe = Encoder(self.z_size, self.num_input_channels, self.base_channel_size, self.e_size)

        self.decoder = Decoder(self.z_size, self.num_input_channels, self.base_channel_size)
        
        self.mu_py = nn.Embedding(self.y_size, self.z_size)
        self.mu_px = nn.Embedding(self.y_size, self.z_size)
        self.mu_pe = nn.Embedding(self.e_size, self.z_size)

        self.logsd = nn.Parameter(torch.Tensor([0.0]))

        self.save_hyperparameters()

    def forward(self, batch):
        x_c = batch['background']
        x_t = batch['target']
        y_c = batch['background_label']
        y_t = batch['target_label']
        e_c = batch['background_batch']
        e_t = batch['target_batch']

        prior_mu_c = {
            'mu_px': self.mu_px(y_c.int()),
            'mu_py': self.mu_py(y_c.int()),
            'mu_pe': self.mu_pe(e_c.int())
        }
        prior_mu_t = {
            'mu_px': self.mu_px(y_t.int()),
            'mu_py': self.mu_py(y_t.int()),
            'mu_pe': self.mu_pe(e_t.int())
        }
        inference_out = self.inference(x_c, x_t, e_c, e_t)
        generative_out = self.generative(inference_out['c'], inference_out['t'])
        recon = {'c': generative_out['c']["x_pred"], 't': generative_out['t']["x_pred"]}
        inference_out['c'].update(prior_mu_c)
        inference_out['t'].update(prior_mu_t)

        return recon, inference_out, generative_out

    def get_image_embedding(self, x):
        mu_qy, _ = self.qy(x)
        mu_qx, _ = self.qx(x)
        return torch.cat((mu_qx, mu_qy), dim=1)

    def _generic_inference(self,
                           x: torch.Tensor,
                           e: torch.Tensor
                           ):
        mu_qy, logvar_qy = self.qy(x)
        mu_qx, logvar_qx = self.qx(x)
        mu_qe, logvar_qe = self.qe(x, e)

        # sample from latent distribution
        logvar_qy = torch.maximum(logvar_qy, torch.tensor(-20)) #clipping to prevent going to -inf
        logvar_qx = torch.maximum(logvar_qx, torch.tensor(-20)) #clipping to prevent going to -inf
        logvar_qe = torch.maximum(logvar_qe, torch.tensor(-20))  # clipping to prevent going to -inf
        sd_qy = torch.exp(logvar_qy / 2)
        sd_qx = torch.exp(logvar_qx / 2)
        sd_qe = torch.exp(logvar_qe / 2)
        dist_qy = Normal(mu_qy, sd_qy)
        dist_qx = Normal(mu_qx, sd_qx)
        dist_qe = Normal(mu_qe, sd_qe)
        zy = dist_qy.rsample()
        zx = dist_qx.rsample()
        ze = dist_qe.rsample()

        out = dict(
            mu_qy=mu_qy,
            sd_qy=sd_qy,
            zy=zy,
            mu_qx=mu_qx,
            sd_qx=sd_qx,
            zx=zx,
            mu_qe=mu_qe,
            sd_qe=sd_qe,
            ze=ze
        )
        return out

    def inference(
            self,
            x_c: torch.Tensor,
            x_t: torch.Tensor,
            e_c: torch.Tensor,
            e_t: torch.Tensor
        ) -> Dict[str, Dict[str, torch.Tensor]]:
            batch_size_c = x_c.shape[0]
            batch_size_t = x_t.shape[0]
            x = torch.cat([x_c, x_t], dim=0)
            e = torch.cat([e_c, e_t], dim=0)
            out = self._generic_inference(x=x, e=e)
            out_c, out_t = {}, {}
            for key in out.keys():
                if out[key] is not None:
                    out_c_elem, out_t_elem = torch.split(
                        out[key],
                        [batch_size_c, batch_size_t],
                        dim=0,
                    )
                else:
                    out_c_elem, out_t_elem = None, None
                out_c[key] = out_c_elem
                out_t[key] = out_t_elem
            out_c["s"] = torch.zeros_like(out_c["s"])
            return dict(c=out_c, t=out_t)

    def _generic_generative(self,
                            zy: torch.Tensor,
                            zx: torch.Tensor,
                            ze: torch.Tensor):
        z = torch.cat([zy, zx, ze], dim=-1)
        x_pred = self.decoder(z)
        return dict(x_pred=x_pred)

    def generative(
        self,
        inference_out_c: Dict[str, torch.Tensor],
        inference_out_t: Dict[str, torch.Tensor],
        ) -> Dict[str, Dict[str, torch.Tensor]]:
        zy_shape = inference_out_c["zy"].shape
        batch_size_dim = 0 if len(zy_shape) == 2 else 1
        batch_size_c = inference_out_c["zy"].shape[batch_size_dim]
        batch_size_t = inference_out_t["zy"].shape[batch_size_dim]
        generative_input = {}
        for key in ["zy", "zx", "ze"]:
            generative_input[key] = torch.cat(
                [inference_out_c[key], inference_out_t[key]], dim=batch_size_dim
            )
        out = self._generic_generative(**generative_input)
        out_c, out_t = {}, {}
        if out["x_pred"] is not None:
            x_pred_c, x_pred_t = torch.split(
                out["x_pred"],
                [batch_size_c, batch_size_t],
                dim=batch_size_dim,
            )
        else:
            x_pred_c, x_pred_t = None, None
        out_c["x_pred"] = x_pred_c
        out_t["x_pred"] = x_pred_t
        return dict(c=out_c, t=out_t)

    def _generic_loss(self,
                      x: torch.Tensor,
                      inference_out: Dict[str, torch.Tensor],
                      generative_out: Dict[str, torch.Tensor],
                      )-> Dict[str, torch.Tensor]:
        mu_qy = inference_out["mu_qy"]
        sd_qy = inference_out["sd_qy"]
        mu_qx = inference_out["mu_qx"]
        sd_qx = inference_out["sd_qx"]
        mu_qe = inference_out["mu_qe"]
        sd_qe = inference_out["sd_qe"]
        mu_py = inference_out["mu_py"]
        mu_px = inference_out["mu_px"]
        mu_pe = inference_out["mu_pe"]
        x_pred = generative_out["x_pred"]

        recon_loss = VAEmodel.reconstruction_loss(x, x_pred, self.logsd)
        kl_y = VAEmodel.latent_kl_divergence(mu_qy, sd_qy, prior_mean=mu_py)
        kl_x = VAEmodel.latent_kl_divergence(mu_qx, sd_qx, prior_mean=mu_px)
        kl_e = VAEmodel.latent_kl_divergence(mu_qe, sd_qe, prior_mean=mu_pe)
        return dict(recon_loss=recon_loss, kl_y=kl_y, kl_x=kl_x, kl_e=kl_e)
    
    def compute_independent_loss(self, zb, zc):
        reg_type = self.reg_type
        if reg_type == "TC":
            return self.compute_tc(zb, zc)
        elif reg_type == "HSIC":
            return self.compute_HSIC(zb, zc)
        else:
            raise ValueError("reg_type should be TC or HSIC")
    
    @staticmethod
    def rbf_kernel(X, sigma=1.0):
        # Compute the pairwise squared Euclidean distances
        pairwise_dists = torch.cdist(X, X, p=2) ** 2
        # Apply the RBF kernel function
        values = torch.div(-pairwise_dists, (2 * sigma**2))
        return values.exp()
    
    @staticmethod
    def compute_HSIC(Z_b, Z_c):
        n = Z_b.shape[0]
        # Compute kernel matrices
        K = ContrastiveVAEmodel.rbf_kernel(Z_b)
        L = ContrastiveVAEmodel.rbf_kernel(Z_c)
        # print(K.shape, L.shape)
        # Implement the HSIC formula
        term1 = (1 / (n**2)) * torch.sum(K * L)
        term2 = (1 / (n**4)) * torch.sum(K) * torch.sum(L)
        term3 = (2 / (n**3)) * torch.sum(K @ L)
        HSIC_n = term1 + term2 - term3
        return HSIC_n * n
    
    @staticmethod
    def compute_tc(zb, zc):
        # Calculate the empirical means
        mean_zb = torch.mean(zb, dim=0)
        mean_zc = torch.mean(zc, dim=0)
        # Calculate the centered variables
        centered_zb = zb - mean_zb
        centered_zc = zc - mean_zc
        # Calculate the covariance matrix of the concatenated latent variables
        z_concat = torch.cat([centered_zb, centered_zc], dim=1)
        cov_matrix = torch.matmul(z_concat.T, z_concat) / z_concat.shape[0]
        # Calculate the covariance matrices for zb and zc individually
        cov_zb = torch.matmul(centered_zb.T, centered_zb) / centered_zb.shape[0]
        cov_zc = torch.matmul(centered_zc.T, centered_zc) / centered_zc.shape[0]
        # Calculate total correlation loss
        tc_loss = torch.logdet(cov_matrix) - (torch.logdet(cov_zb) + torch.logdet(cov_zc))
        # Multiply by the weighting factor
        return -tc_loss

    def _get_loss(self, 
             concat_tensors: Dict[str, Tuple[Dict[str, torch.Tensor], int]],
             ):  
        _, inference_outputs, generative_outputs = self.forward(**concat_tensors)            

        background_losses = self._generic_loss(
            concat_tensors["c"],
            inference_outputs["c"],
            generative_outputs["c"],
        )
        target_losses = self._generic_loss(
            concat_tensors["t"],
            inference_outputs["t"],
            generative_outputs["t"],
        )
        recon_loss = background_losses["recon_loss"] + target_losses["recon_loss"]
        kl_divergence_z = background_losses["kl_z"] + target_losses["kl_z"]
        kl_divergence_s = target_losses["kl_s"]

        wasserstein_loss = (
            torch.norm(inference_outputs["c"]["mu_qx"], dim=-1)**2
            + torch.sum(inference_outputs["c"]["sd_qx"]**2, dim=-1)
        )

        if self.reg_type is not None:
            zb = torch.concat([inference_outputs["t"]["mu_qy"], inference_outputs["c"]["mu_qy"]], axis=0)
            zs = torch.concat([inference_outputs["t"]["mu_qx"], inference_outputs["c"]["mu_qx"]], axis=0)
            tc_loss = self.compute_independent_loss(zb, zs)
        else:
            tc_loss = torch.zeros(1, device=self.device)

        kl_term_weight = self.kl_weight_scheduler.step()
        
        elbo = torch.mean(recon_loss + 
                          kl_term_weight * (kl_divergence_s + kl_divergence_z + 
                                            self.wasserstein_penalty * wasserstein_loss +
                                            self.tc_penalty * tc_loss))

        self.log_dict({
            'kl_divergence_z': kl_divergence_z.mean().detach(),
            'kl_divergence_s': kl_divergence_s.mean().detach(),
            'total_recon_loss': recon_loss.mean().detach(),
            'wasserstein_loss': wasserstein_loss.mean().detach(),
            'tc_loss': tc_loss.mean().detach(),
            # 'background_recon_loss': background_losses["recon_loss"].mean().detach(),
            # 'target_recon_loss': target_losses["recon_loss"].mean().detach(),
            'kl_term_weight': kl_term_weight,
        })
        return elbo
    
 
class CyclicWeightScheduler:
    def __init__(self, step_size, base_weight=0., max_weight=1.):
        self.base_weight = base_weight
        self.max_weight = max_weight
        self.step_size = step_size
        self.cycle = 0
        self.step_count = 0

    def step(self):
        # Compute the current position in the cycle
        cycle_position = self.step_count / self.step_size

        if cycle_position <= 1:
            weight = self.base_weight + (self.max_weight - self.base_weight) * cycle_position
        else:
            weight = self.max_weight
            # weight = self.max_weight - (self.max_weight - self.base_weight) * (cycle_position - 1)

        self.step_count = (self.step_count + 1) % (self.step_size * 2)

        return weight