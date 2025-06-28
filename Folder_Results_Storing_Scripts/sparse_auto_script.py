import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import pytorch_lightning as pl
import wandb

#### Drawing from 
#### https://github.com/DanielKerrigan/saefarer/tree/41b5c6789952310f515c461804f12e1dfd1fd32d  
#### https://github.com/openai/sparse_autoencoder/tree/4965b941e9eb590b00b253a2c406db1e1b193942 
#### https://github.com/EleutherAI/sae/tree/main 
#### https://github.com/bartbussmann/BatchTopK/tree/main
#### https://github.com/tylercosgrove/sparse-autoencoder-mistral7b/tree/main
#### https://github.com/jbloomAus/SAELens
#### https://github.com/neelnanda-io/1L-Sparse-Autoencoder 

class FeatureNormalizer:
    #FeatureNormalizer class following LN of OpenAI, Saefarer

    @staticmethod
    def forward(x, epsilon=1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        variance = centered.pow(2).mean(dim=-1, keepdim=True)
        ln = centered / (variance + epsilon).sqrt()
        return ln, mean, variance.sqrt()

class SparseActivation:
    #SparseActivation class following TopK class of Saefarer 
    def __init__(self, default_k, postact_fn=F.relu):
        self.default_k = default_k
        self.postact_fn = postact_fn # toggle as needed. If we want to use relu, else switch to identity 
        #should not make a big difference either way since model is incentivised to result in positive top activations anyway 
        self.srt = False #toggle as needed 

    def forward(self, features, k=None):
        num_active = k if k is not None else self.default_k
        top_vals, top_indexes = torch.topk(features, sorted=self.srt, k=num_active, dim=-1)
        activated = self.postact_fn(top_vals)
        output = torch.zeros_like(features)
        output.scatter_(-1, top_indexes, activated)
        return output, activated, top_indexes

class WeightTier:
    @staticmethod
    def forward(weights, input_dim, hidden_dim):
        temp_linear = nn.Linear(hidden_dim, input_dim, bias=False)
        print(f"temp_linear.weight.shape: {temp_linear.weight.shape}")
        weights[0].data = temp_linear.weight.clone()  
        print(f"self.weights[0].data.shape: {weights[0].data.shape}")
        weights[1].data = temp_linear.weight.T.clone() 
        print(f"self.weights[1].data.shape: {weights[1].data.shape}")
        del temp_linear
        print("Weights Tied successfully")
        return weights

class TopKAuto(nn.Module):
    def __init__(self, input_dim, hidden_dim, k, encoder_decoder_init, 
                 inactive_threshold=200, aux_alpha=0.03125):
        super(TopKAuto, self).__init__()
        
        # Add deterministic settings for CUDA operations
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

        ###toggle False/True to toggle between active latent init vs dead latent init
        self.init_idle_counts_as_threshold = False
        if self.init_idle_counts_as_threshold:
            self.register_buffer('neuron_idle_counts', torch.full((hidden_dim,), inactive_threshold, dtype=torch.long))
        else:
            self.register_buffer('neuron_idle_counts', torch.zeros(hidden_dim, dtype=torch.long)) 
            ###note to self:this is what saefarer/openai intialises as, but starting w neurons already inactive may lead to faster convergence 
            ###depending on how large my threshold is set; could also consider using two different thresholds: one for starting and one for otherwise

        ###BatchTopK, Saefarer:
        # self.b_dec = nn.Parameter(torch.zeros(self.cfg["act_size"]))
        # self.b_enc = nn.Parameter(torch.zeros(self.cfg["dict_size"]))
        # self.W_enc = nn.Parameter(
        #     torch.nn.init.kaiming_uniform_(
        #         torch.empty(self.cfg["act_size"], self.cfg["dict_size"])
        #     )
        # )
        # self.W_dec = nn.Parameter(
        #     torch.nn.init.kaiming_uniform_(
        #         torch.empty(self.cfg["dict_size"], self.cfg["act_size"])
        #     )
        # )
        # self.W_dec.data[:] = self.W_enc.t().data
        # self.W_dec = nn.Parameter(self.W_enc.t().clone()) #Saefarer
        # self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        ##For the encoder_decoder_init = 0 case:
        self.weights = nn.ParameterList([
            nn.Parameter(
                torch.randn(input_dim, hidden_dim) 
                / math.sqrt(input_dim)
                ),  # W_enc (BatchTopK), encoder.weight (Eleuther)
            #when encoder_decoder_init = 0
            nn.Parameter(
                torch.randn(hidden_dim, input_dim) 
                / math.sqrt(hidden_dim)
                ),  # W_dec (both)
            #when encoder_decoder_init = 0
            nn.Parameter(
                torch.zeros(hidden_dim)
                ), # b_enc (BatchTopK), encoder.bias (Eleuther)
            ##### #note to self: all same parameters for Vanilla SAE, but there just need to normalise decoder at every step + L1 norm
            nn.Parameter(
                torch.zeros(input_dim)
                ) # b_dec (both)
        ])


        # Optionally tie weights at initialisation 
        if encoder_decoder_init == 1:
            self.weights = WeightTier.forward(self.weights, input_dim, hidden_dim) #only at initialisation 
        else:
            ##note to self: now removed option for encoder_decoder_init = 0 since using init seems to speed up convergence in toy runs
            raise ValueError("encoder_decoder_init must be set to 1 to reduce dead neurons.")

        ####note to self: we don't have to normalise decoder in Topk (it's optional openai C.4.) unlike Vanilla SAE; so we never normalise
        
        self.k = k
        self.inactive_threshold = inactive_threshold

        # # Heuristic from Appendix B.1 in the paper, Eleuther 
        # k_aux = y.shape[-1] // 2
        self.k_aux = input_dim//2 
        
        self.feature_normalizer = FeatureNormalizer()
        self.sparse_activation = SparseActivation(default_k=self.k)

        self.aux_alpha = aux_alpha #Openai Appendix B1, saefarer config, batchtopk, openai  
        #openai says this does not really change things much


    ##Saefarer
    # def unprocess(
    #     self, x: torch.Tensor, info: dict[str, torch.Tensor] | None = None
    # ) -> torch.Tensor:
    #     if self.cfg.normalize and info:
    #         return x * info["std"] + info["mu"]
    #     else:
    #         return x

    ##Saefarer
    # def decode(
    #     self, latents: torch.Tensor, info: dict[str, Any] | None = None
    # ) -> torch.Tensor:
    #     """
    #     :param latents: autoencoder latents (shape: [batch, n_latents])
    #     :return: reconstructed data (shape: [batch, n_inputs])
    #     """
    #     recontructed = latents @ self.W_dec + self.b_dec
    #     return self.unprocess(recontructed, info)


    #############################################################################################################
    ####(Saefarer model.py)

    # self.stats_last_nonzero: torch.Tensor
    # self.register_buffer(
    #     "stats_last_nonzero",
    #     torch.zeros(cfg.d_sae, dtype=torch.long, device=self.device),
    # )

    # def get_dead_neuron_mask(self) -> torch.Tensor:
    #     return self.stats_last_nonzero > self.cfg.dead_steps_threshold

    # def auxk_masker(self, x: torch.Tensor) -> torch.Tensor:
    #     """mask dead neurons"""
    #     dead_mask = self.get_dead_neuron_mask()
    #     x.data *= dead_mask
    #     return x

    # def forward(self, x: torch.Tensor) -> ForwardOutput:
    #     x_preprocessed, info = self.preprocess(x)
    #     latents_pre_act = self.encode_pre_act(x_preprocessed)
    #     latents = self.topk(latents_pre_act)

    #     recons = self.decode(latents)

    #     mse_loss = normalized_mse(recons, x_preprocessed)

    #     # set all indices of self.stats_last_nonzero where (latents != 0) to 0
    #     self.stats_last_nonzero *= (latents == 0).all(dim=0).long()
    #     self.stats_last_nonzero += 1

    #     num_dead = int(self.get_dead_neuron_mask().sum().item())

    #     if num_dead > 0:
    #         aux_latents = self.aux_topk(self.auxk_masker(latents_pre_act))
    #         aux_recons = self.decode(aux_latents)
    #         aux_loss = self.cfg.aux_k_coef * normalized_mse(
    #             aux_recons, x_preprocessed - recons.detach() + self.b_dec.detach()
    #         ).nan_to_num(0)
    #     else:
    #         aux_loss = mse_loss.new_tensor(0.0)

    #     loss = mse_loss + aux_loss

    ###############################################################################################################

    def helper_for_extraction(self, features, postact_fn=F.relu, srt=False):        
        top_vals, top_indices = torch.topk(features, sorted=srt, k=self.k, dim=-1)
        top_vals = postact_fn(top_vals)  # (defaults to ReLU)
        return top_vals, top_indices


#Lightning module for train and val 
class LitLit(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, k,
                 encoder_decoder_init, 
                 learning_rate, inactive_threshold=200, aux_alpha=0.03125):
        super().__init__()
        self.save_hyperparameters(ignore=['aux_alpha'])
        
        # Add deterministic settings for PyTorch Lightning
        pl.seed_everything(42, workers=True)  # Set a fixed seed for all operations
        
        self.hidden_dim = hidden_dim
        
        self.model = TopKAuto(
            input_dim=input_dim,
            encoder_decoder_init=encoder_decoder_init,
            hidden_dim=hidden_dim,
            inactive_threshold=inactive_threshold,
            k=k,
            aux_alpha=aux_alpha
        )
        
        self.learning_rate = learning_rate
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @classmethod
    def load_from_checkpoint(cls, ckpt_file, *args, **kwargs):
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']
        
        # Extract hyperparameters from checkpoint
        hparams = checkpoint.get('hyper_parameters', {})
        
        # Get dimensions from the state dict
        encoder_shape = state_dict['model.weights.0'].shape
        input_dim, hidden_dim = encoder_shape
        
        # Get k from hyperparameters - raise error if not found
        if 'k' not in hparams:
            raise ValueError("Could not find 'k' in saved hyperparameters. Cannot load model without knowing k value.")
        k = hparams['k']
        print(f"Loading model with k={k}")
        
        # Get other hyperparameters with defaults
        learning_rate = hparams.get('learning_rate', 0.001)
        encoder_decoder_init = hparams.get('encoder_decoder_init', 1)
        
        # Create model instance with extracted parameters
        model = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            k=k,
            encoder_decoder_init=encoder_decoder_init,
            learning_rate=learning_rate
        )
        
        # Load the state dict
        model.load_state_dict(state_dict, strict=True)
        return model







