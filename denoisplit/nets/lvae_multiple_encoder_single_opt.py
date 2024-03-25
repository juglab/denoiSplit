"""
here, using a single optimizer we want to train the model.
"""
import torch
import torch.optim as optim

from denoisplit.core.mixed_input_type import MixedInputType
from denoisplit.nets.lvae_multiple_encoders import LadderVAEMultipleEncoders


class LadderVAEMulEncoder1Optim(LadderVAEMultipleEncoders):
    def configure_optimizers(self):
        encoder_params = self.get_encoder_params()
        decoder_params = self.get_decoder_params()
        encoder_ch1_params = self.get_ch1_branch_params()
        encoder_ch2_params = self.get_ch2_branch_params()
        optimizer = optim.Adamax(encoder_params + decoder_params + encoder_ch1_params + encoder_ch2_params, lr=self.lr,
                                 weight_decay=0)

        scheduler = self.get_scheduler(optimizer)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.lr_scheduler_monitor}

    def training_step(self, batch, batch_idx, enable_logging=True):

        x, target, supervised_mask = batch
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)
        recons_loss = 0
        kl_loss = 0
        if supervised_mask.sum() > 0:
            out, td_data = self._forward_mix(x_normalized[supervised_mask])
            recons_loss_dict = self._get_reconstruction_loss_vector(out, target_normalized[supervised_mask])
            recons_loss = recons_loss_dict['loss'].sum()
            kl_loss = self.get_kl_divergence_loss(td_data) * supervised_mask.sum()
            # todo: one can also apply mixed reconstruction loss here. input mix and reconstruct mix.

        if (~supervised_mask).sum() > 0:
            target_indep = target_normalized[~supervised_mask]
            out_ch0, td_data0 = self._forward_separate_ch(target_indep[:, :1], None)
            out_ch1, td_data1 = self._forward_separate_ch(None, target_indep[:, 1:2])
            recons_loss_ch0 = self._get_reconstruction_loss_vector(out_ch0, target_indep)['ch1_loss']
            recons_loss_ch1 = self._get_reconstruction_loss_vector(out_ch1, target_indep)['ch2_loss']

            kl_loss0 = self.get_kl_divergence_loss(td_data0)
            kl_loss1 = self.get_kl_divergence_loss(td_data1)

            kl_loss_mix = None
            recons_loss_mix = None
            if self.mixed_input_type == MixedInputType.Aligned:
                out_mix, td_datamix = self._forward_mix(x_normalized[~supervised_mask])
                recons_loss_mix = self._get_mixed_reconstruction_loss_vector(out_mix, x_normalized[~supervised_mask])
                kl_loss_mix = self.get_kl_divergence_loss(td_datamix)
                recons_loss += (recons_loss_ch0.sum() + recons_loss_ch1.sum() + recons_loss_mix.sum()) / 3
                kl_loss += (kl_loss0 + kl_loss1 + kl_loss_mix) / 3 * len(target_indep)
            else:
                recons_loss += (recons_loss_ch0.sum() + recons_loss_ch1.sum()) / 2
                kl_loss += (kl_loss0 + kl_loss1) / 2 * len(target_indep)

            if enable_logging:
                self.log(f'reconstruction_loss_ch0', recons_loss_ch0.mean(), on_epoch=True)
                self.log(f'reconstruction_loss_ch1', recons_loss_ch1.mean(), on_epoch=True)
                self.log(f'kl_loss_ch0', kl_loss0, on_epoch=True)
                self.log(f'kl_loss_ch1', kl_loss1, on_epoch=True)
                if self.mixed_input_type == MixedInputType.Aligned:
                    self.log(f'reconstruction_loss_mix', recons_loss_mix.mean(), on_epoch=True)
                    self.log(f'kl_loss_mix', kl_loss0, on_epoch=True)

        recons_loss = recons_loss / len(x)
        kl_loss = kl_loss / len(x)
        net_loss = recons_loss + self.get_kl_weight() * kl_loss
        if enable_logging:
            self.log('kl_loss', kl_loss, on_epoch=True)
            self.log('reconstruction_loss', recons_loss, on_epoch=True)

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
