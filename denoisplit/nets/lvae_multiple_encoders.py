import copy

import torch
import torch.nn as nn
import torch.optim as optim

from denoisplit.core.data_utils import crop_img_tensor
from denoisplit.core.mixed_input_type import MixedInputType
from denoisplit.nets.lvae import LadderVAE
from denoisplit.nets.lvae_layers import BottomUpLayer, MergeLayer


class LadderVAEMultipleEncoders(LadderVAE):

    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at=use_uncond_mode_at, target_ch=target_ch)
        self.bottom_up_layers_ch1 = nn.ModuleList([])
        self.bottom_up_layers_ch2 = nn.ModuleList([])

        fbu_num_blocks = config.model.fbu_num_blocks
        del self.first_bottom_up
        stride = 1 if config.model.no_initial_downscaling else 2
        self.first_bottom_up = self.create_first_bottom_up(stride, num_blocks=fbu_num_blocks)
        self.first_bottom_up_ch1 = self.create_first_bottom_up(stride, num_blocks=fbu_num_blocks)
        self.first_bottom_up_ch2 = self.create_first_bottom_up(stride, num_blocks=fbu_num_blocks)
        shape = (1, config.data.image_size, config.data.image_size)
        self._inp_tensor_ch1 = nn.Parameter(torch.zeros(shape, requires_grad=True))
        self._inp_tensor_ch2 = nn.Parameter(torch.zeros(shape, requires_grad=True))

        self.lowres_first_bottom_ups_ch1 = self.lowres_first_bottom_ups_ch2 = None
        self.share_bottom_up_starting_idx = config.model.share_bottom_up_starting_idx
        self.mixed_input_type = config.data.mixed_input_type
        self.separate_mix_branch_training = config.model.separate_mix_branch_training
        if self.lowres_first_bottom_ups is not None:
            self.lowres_first_bottom_ups_ch1 = copy.deepcopy(self.lowres_first_bottom_ups_ch1)
            self.lowres_first_bottom_ups_ch2 = copy.deepcopy(self.lowres_first_bottom_ups_ch2)

        enable_multiscale = self._multiscale_count is not None and self._multiscale_count > 1
        multiscale_lowres_size_factor = 1

        for i in range(self.n_layers):
            # Whether this is the top layer
            layer_enable_multiscale = enable_multiscale and self._multiscale_count > i + 1
            # if multiscale is enabled, this is the factor by which the lowres tensor will be larger than
            multiscale_lowres_size_factor *= (1 + int(layer_enable_multiscale))
            # Add bottom-up deterministic layer at level i.
            # It's a sequence of residual blocks (BottomUpDeterministicResBlock)
            # possibly with downsampling between them.
            if i >= self.share_bottom_up_starting_idx:
                self.bottom_up_layers_ch1.append(self.bottom_up_layers[i])
                self.bottom_up_layers_ch2.append(self.bottom_up_layers[i])
                continue

            blayer = self.get_bottom_up_layer(i, config.model.multiscale_lowres_separate_branch, enable_multiscale,
                                              multiscale_lowres_size_factor)
            self.bottom_up_layers_ch1.append(blayer)
            blayer = self.get_bottom_up_layer(i, config.model.multiscale_lowres_separate_branch, enable_multiscale,
                                              multiscale_lowres_size_factor)
            self.bottom_up_layers_ch2.append(blayer)

        msg = f'[{self.__class__.__name__}] ShareStartIdx:{self.share_bottom_up_starting_idx} '
        msg += f'SepMixedBranch:{self.separate_mix_branch_training} '
        print(msg)

    def get_bottom_up_layer(self, ith_layer, lowres_separate_branch, enable_multiscale, multiscale_lowres_size_factor):
        return BottomUpLayer(
            n_res_blocks=self.encoder_blocks_per_layer,
            n_filters=self.encoder_n_filters,
            downsampling_steps=self.downsample[ith_layer],
            nonlin=self.get_nonlin(),
            batchnorm=self.batchnorm,
            dropout=self.encoder_dropout,
            res_block_type=self.res_block_type,
            gated=self.gated,
            lowres_separate_branch=lowres_separate_branch,
            enable_multiscale=enable_multiscale,
            multiscale_retain_spatial_dims=self.multiscale_retain_spatial_dims,
            multiscale_lowres_size_factor=multiscale_lowres_size_factor,
        )

    def get_scheduler(self, optimizer):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                    self.lr_scheduler_mode,
                                                    patience=self.lr_scheduler_patience,
                                                    factor=0.5,
                                                    min_lr=1e-12,
                                                    verbose=True)

    def get_encoder_params(self):
        encoder_params = list(self.first_bottom_up.parameters()) + list(self.bottom_up_layers.parameters())
        if self.lowres_first_bottom_ups is not None:
            encoder_params.append(self.lowres_first_bottom_ups.parameters())
        return encoder_params

    def get_ch1_branch_params(self):
        encoder_ch1_params = list(self.first_bottom_up_ch1.parameters()) + list(self.bottom_up_layers_ch1.parameters())
        if self.lowres_first_bottom_ups_ch1 is not None:
            encoder_ch1_params.append(self.lowres_first_bottom_ups_ch1.parameters())
        encoder_ch1_params.append(self._inp_tensor_ch1)
        return encoder_ch1_params

    def get_ch2_branch_params(self):
        encoder_ch2_params = list(self.first_bottom_up_ch2.parameters()) + list(self.bottom_up_layers_ch2.parameters())
        if self.lowres_first_bottom_ups_ch2 is not None:
            encoder_ch2_params.append(self.lowres_first_bottom_ups_ch2.parameters())
        encoder_ch2_params.append(self._inp_tensor_ch2)
        return encoder_ch2_params

    def get_decoder_params(self):
        decoder_params = list(self.top_down_layers.parameters()) + list(self.final_top_down.parameters()) + list(
            self.likelihood.parameters())
        return decoder_params

    def configure_optimizers(self):

        encoder_params = self.get_encoder_params()
        decoder_params = self.get_decoder_params()
        encoder_ch1_params = self.get_ch1_branch_params()
        encoder_ch2_params = self.get_ch2_branch_params()
        # channel 1 params

        if self.separate_mix_branch_training:
            optimizer0 = optim.Adamax(encoder_params, lr=self.lr, weight_decay=0)
        else:
            optimizer0 = optim.Adamax(encoder_params + decoder_params, lr=self.lr, weight_decay=0)
        optimizer1 = optim.Adamax(encoder_ch1_params + encoder_ch2_params + decoder_params, lr=self.lr, weight_decay=0)

        scheduler0 = self.get_scheduler(optimizer0)
        scheduler1 = self.get_scheduler(optimizer1)

        return [optimizer0, optimizer1], [{
            'scheduler': scheduler,
            'monitor': self.lr_scheduler_monitor,
        } for scheduler in [scheduler0, scheduler1]]

    def _forward_mix(self, x):
        img_size = x.size()[2:]

        # Pad input to make everything easier with conv strides
        x_pad = self.pad_input(x)

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(mix_inp=x_pad)

        # Top-down inference/generation
        out, td_data = self.topdown_pass(bu_values)
        # Restore original image size
        out = crop_img_tensor(out, img_size)

        return out, td_data

    def _forward_separate_ch(self, ch1_inp, ch2_inp):
        img_size = ch1_inp.size()[2:] if ch1_inp is not None else ch2_inp.size()[2:]

        # Pad input to make everything easier with conv strides
        ch1_inp = self.pad_input(ch1_inp) if ch1_inp is not None else None
        ch2_inp = self.pad_input(ch2_inp) if ch2_inp is not None else None

        # Bottom-up inference: return list of length n_layers (bottom to top)
        bu_values = self.bottomup_pass(ch1_inp=ch1_inp, ch2_inp=ch2_inp)

        # Top-down inference/generation
        out, td_data = self.topdown_pass(bu_values)
        # Restore original image size
        out = crop_img_tensor(out, img_size)

        return out, td_data

    def _bottomup_pass_ch(self, ch1_inp, ch2_inp):
        if ch1_inp is None:
            ch1_inp = self._inp_tensor_ch1[None]
            assert ch2_inp is not None
            ch1_inp = torch.tile(ch1_inp, (len(ch2_inp), 1, 1, 1))

        if ch2_inp is None:
            ch2_inp = self._inp_tensor_ch2[None]
            assert ch1_inp is not None
            ch2_inp = torch.tile(ch2_inp, (len(ch1_inp), 1, 1, 1))

        x1 = self.first_bottom_up_ch1(ch1_inp)
        x2 = self.first_bottom_up_ch2(ch2_inp)
        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []

        for i in range(self.n_layers):

            if self.share_bottom_up_starting_idx > i:
                x1, bu_value1 = self.bottom_up_layers_ch1[i](x1, lowres_x=None)
                x2, bu_value2 = self.bottom_up_layers_ch2[i](x2, lowres_x=None)
                bu_values.append((bu_value1 + bu_value2) / 2)
            else:
                if self.share_bottom_up_starting_idx == i:
                    x = (x1 + x2) / 2

                x, bu_value = self.bottom_up_layers[i](x, lowres_x=None)

                bu_values.append(bu_value)

        return bu_values

    def bottomup_pass(self, mix_inp=None, ch1_inp=None, ch2_inp=None):
        # by default it is necessary to feed 0, since in validation step it is required.
        if mix_inp is not None:
            return super().bottomup_pass(mix_inp)
        else:
            return self._bottomup_pass_ch(ch1_inp, ch2_inp)

    def validation_step(self, batch, batch_idx):
        x, target, supervised_mask = batch
        assert supervised_mask.sum() == len(x)
        return super().validation_step((x, target), batch_idx)

    # TODO: TRAINING STEP FOR semi_supervised_v3. I need to use this.
    # def training_step(self, batch, batch_idx, optimizer_idx, enable_logging=True):
    #
    #     x, target, supervised_mask = batch
    #     x_normalized = self.normalize_input(x)
    #     target_normalized = self.normalize_target(target)
    #     if optimizer_idx == 0:
    #         out, td_data = self.forward_ch(x_normalized, optimizer_idx)
    #         if self.mixed_input_type == MixedInputType.ConsistentWithSingleInputs:
    #             if self.skip_disentanglement_for_nonaligned_data:
    #                 if supervised_mask.sum() > 0:
    #                     recons_loss_dict = self._get_reconstruction_loss_vector(out[supervised_mask],
    #                                                                             target_normalized[supervised_mask])
    #                     recons_loss = recons_loss_dict['loss'].mean()
    #                 else:
    #                     recons_loss = 0.0
    #             else:
    #                 recons_loss_dict = self._get_reconstruction_loss_vector(out, target_normalized)
    #                 recons_loss = recons_loss_dict['loss'].mean()
    #         else:
    #             assert self.mixed_input_type == MixedInputType.Aligned
    #             recons_loss = 0
    #             if supervised_mask.sum() > 0:
    #                 recons_loss_dict = self._get_reconstruction_loss_vector(out[supervised_mask],
    #                                                                         target_normalized[supervised_mask])
    #                 recons_loss = recons_loss_dict['loss'].sum()
    #             if (~supervised_mask).sum() > 0:
    #                 # todo: check if x_normalized does not have any extra pre-processing.
    #                 recons_loss += self._get_mixed_reconstruction_loss_vector(out[~supervised_mask],
    #                                                                           x_normalized[~supervised_mask]).sum()
    #             N = len(x)
    #             recons_loss = recons_loss / N
    #     else:
    #         out, td_data = self.forward_ch(target_normalized[:, optimizer_idx - 1:optimizer_idx], optimizer_idx)
    #         recons_loss_dict = self._get_reconstruction_loss_vector(out, target_normalized)
    #         if optimizer_idx == 1:
    #             recons_loss = recons_loss_dict['ch1_loss'].mean()
    #         elif optimizer_idx == 2:
    #             recons_loss = recons_loss_dict['ch2_loss'].mean()
    #
    def training_step(self, batch, batch_idx, optimizer_idx, enable_logging=True):
        x, target, _ = batch
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)
        if optimizer_idx == 0:
            out, td_data = self._forward_mix(x_normalized)
            assert self.mixed_input_type == MixedInputType.ConsistentWithSingleInputs
            recons_loss_dict = self._get_reconstruction_loss_vector(out, target_normalized)
            recons_loss = recons_loss_dict['loss'].mean()
        else:
            out, td_data = self._forward_separate_ch(target_normalized[:, :1], target_normalized[:, 1:2])
            recons_loss_dict = self._get_reconstruction_loss_vector(out, target_normalized)
            recons_loss = recons_loss_dict['loss'].mean()

        kl_loss = self.get_kl_divergence_loss(td_data)

        net_loss = recons_loss + self.get_kl_weight() * kl_loss
        if enable_logging:
            self.log(f'reconstruction_loss_ch{optimizer_idx}', recons_loss, on_epoch=True)
            self.log(f'kl_loss_ch{optimizer_idx}', kl_loss, on_epoch=True)

        output = {
            'loss': net_loss,
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if torch.isnan(net_loss).any():
            return None

        # skipping inf loss
        if torch.isinf(net_loss).any():
            return None

        return output
