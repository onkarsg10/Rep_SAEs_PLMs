##FINAL REPRODUCIBLE
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import pytorch_lightning as pl
import wandb

import random
import numpy as np
import os

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seeds at the start of the script
set_seed()

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


    def _encode(self, shifted_input): #Saefarer
        mid_before_sparse = torch.matmul(shifted_input, self.weights[0]) + self.weights[2]
        mid_sparse, activated, top_indexes = self.sparse_activation.forward(mid_before_sparse)
        return mid_before_sparse, mid_sparse

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

    def _decode(self, mid_sparse, mean, std):
        reconstruction = (torch.matmul(mid_sparse, self.weights[1]) + self.weights[3]) * std + mean ##Saefarer decode method includes unprocess 
        return reconstruction

    def _preprocess_input(self, x): #Saefarer
        ln, mean, std = self.feature_normalizer.forward(x)
        shifted_input = ln - self.weights[3]
        return shifted_input, mean, std

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


    def forward(self, x):
        ####TopK SAE (BatchTopK sae.py), Saefarer
        # x, x_mean, x_std = self.preprocess_input(x) #preprocess method created & added the shift into it
        # x_cent = x - self.b_dec 
        # acts = F.relu(x_cent @ self.W_enc) #encode method created & added sparsification into it 
        # acts_topk = torch.topk(acts, self.cfg["top_k"], dim=-1)
        # acts_topk = torch.zeros_like(acts).scatter(
        #     -1, acts_topk.indices, acts_topk.values
        # )
        # x_reconstruct = acts_topk @ self.W_dec + self.b_dec #decode method created & subsumed the post_process output method in it (for unnormalising w std + mean)
        # sae_out = self.postprocess_output(x_reconstruct, x_mean, x_std)

        # Preprocess input
        shifted_input, mean, std = self._preprocess_input(x)
        # Encode
        mid_before_sparse, mid_sparse = self._encode(shifted_input)

        ####Saefarer
        if self.training:
            self._update_neuron_activity_counts(mid_sparse)

        inactive_mask = self.neuron_idle_counts > self.inactive_threshold

        # Initialize aux_loss
        aux_loss = torch.zeros((), device=x.device) #not needed for checkpoint saving
        

        reconstruction = self._decode(mid_sparse, mean, std)
        ##Computing main loss by comparing in normalised space (as BatchTopK)
        # ln, _, _ = self.feature_normalizer.forward(x)
        # reconstruction = self._decode(mid_sparse, 0, 1)
        # l2_loss = (x_reconstruct.float() - x.float()).pow(2).mean() ##BatchTopK
        # main_loss = torch.mean((ln - reconstruction) ** 2) ##Wait! Fails for transcoders since their target is in unnormalised alien space 
        ##So to stay consistent, compute all main losses in unnormalised space - will work for autoencoding & transcoding. Saefarer, Eleuther, tylercosgrove
        main_loss = torch.mean((x - reconstruction) ** 2) #default loss used for both our LitLit steps 
        #in val step will use default main loss for saving best model checkpoints 

        if torch.count_nonzero(inactive_mask).item() != 0 and self.training:
            aux_reconstruction = self._process_inactive_neurons(
                mid_before_sparse, inactive_mask, mean, std
            )

            #Eleutherai
            # Reduce the scale of the loss if there are a small number of dead latents
            #scale = min(num_dead / k_aux, 1.0)
            scale = min(torch.count_nonzero(inactive_mask).item() / self.k_aux, 1.0)
            print(f"scale: {scale}")
            print(f"self.k_aux: {self.k_aux}")
            residual = x - reconstruction
            # l2_loss_aux = (
            #     self.cfg["aux_penalty"]
            #     * (x_reconstruct_aux.float() - residual.float()).pow(2).mean()
            # ) ##BatchTopK
            aux_pre_alpha = torch.mean((residual - aux_reconstruction) ** 2)
            aux_loss = (self.aux_alpha * scale) * aux_pre_alpha

        
        return {
            'main_loss': main_loss, #default loss used for both our LitLit steps 
        #in val step will use default main loss for saving best model checkpoints
            'aux_loss': aux_loss,
        }

    def _update_neuron_activity_counts(self, activations):
        ##(Saefarer model.py)
        is_inactive = (activations == 0).all(dim=0).long()
        self.neuron_idle_counts *= is_inactive
        self.neuron_idle_counts += 1

    def _process_inactive_neurons(self, latents, inactive_mask, mean, std):       
        ##(Saefarer model.py)  
        masked_latents = latents.clone()
        masked_latents.data *= inactive_mask[None] ####note to self:is more intuitive/explicit than: masked_latents.data *= inactive_mask
        
        inactive_activated_new, activ_new, top_indexes_new = self.sparse_activation.forward(
            masked_latents,
            min(torch.count_nonzero(inactive_mask).item(), self.k_aux)
        )

        ##(EleutherAI sae.py)
        # # Second decoder pass for AuxK loss 
        # if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
        #     # Heuristic from Appendix B.1 in the paper
        #     k_aux = y.shape[-1] // 2 
        #     k_aux = min(k_aux, num_dead) 

        #     # Don't include living latents in this loss
        #     auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
        #     #Top-k dead latents
        #     auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

        return self._decode(inactive_activated_new, mean, std) ##Saefarer decode method includes unprocess





#Lightning module for train and val 
class LitLit(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, k, encoder_decoder_init, 
                 learning_rate, inactive_threshold=200, aux_alpha=0.03125):
        super().__init__()
        self.save_hyperparameters(ignore=['aux_alpha'])
        
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
        
    def forward(self, x):
        if self.training:
            outputs = self.model(x)
            return outputs
        else:
            outputs = self.model(x)
            return outputs['main_loss']

    def training_step(self, batch, batch_idx):

        embeddings, metadata, sequences = batch
        
        outputs = self(embeddings)
        main_loss = outputs['main_loss']
        aux_loss = outputs['aux_loss']
        
        
        total_loss = main_loss + aux_loss
        
        self.log('train_main_loss', main_loss, prog_bar=True, on_step=True)
        self.log('train_total_loss', total_loss, prog_bar=True, on_step=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        embeddings, metadata, sequences = batch
        
        main_loss = self(embeddings)  
        
        inactive_mask = self.model.neuron_idle_counts > self.model.inactive_threshold
        dead_ratio = inactive_mask.float().mean()
        
        self.log('val_loss', main_loss, on_step=True)  #main_loss for val
        self.log('val_dead_ratio', dead_ratio, on_step=True)
        
        return main_loss

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





