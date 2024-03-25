from denoisplit.nets.lvae_with_stitch import LadderVAEwithStitching
import torch.optim as optim
import torch
import torch.nn.functional as F
import os


class LadderVAEwithStitching2Stage(LadderVAEwithStitching):
    def __init__(self, data_mean, data_std, config, use_uncond_mode_at=[], target_ch=2):
        super().__init__(data_mean, data_std, config, use_uncond_mode_at, target_ch)
        assert config.training.pre_trained_ckpt_fpath and os.path.exists(config.training.pre_trained_ckpt_fpath)

    def configure_optimizers(self):
        params = self.offset_predictor.parameters()
        optimizer = optim.Adamax(params, lr=self.lr, weight_decay=0)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         self.lr_scheduler_mode,
                                                         patience=self.lr_scheduler_patience,
                                                         factor=0.5,
                                                         min_lr=1e-12,
                                                         verbose=True)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': self.lr_scheduler_monitor}

    def training_step(self, batch: tuple, batch_idx: int, enable_logging=True):
        x, target, grid_sizes = batch
        x_normalized = self.normalize_input(x)
        target_normalized = self.normalize_target(target)

        out, td_data = self.forward(x_normalized)
        offset = self.compute_offset(td_data['z'])
        if self.encoder_no_padding_mode and out.shape[-2:] != target_normalized.shape[-2:]:
            target_normalized = F.center_crop(target_normalized, out.shape[-2:])

        recons_loss_dict, imgs = self.get_reconstruction_loss(out, target_normalized, offset, return_predicted_img=True)

        if self.skip_nboundary_pixels_from_loss:
            pad = self.skip_nboundary_pixels_from_loss
            target_normalized = target_normalized[:, :, pad:-pad, pad:-pad]

        recons_loss = recons_loss_dict['loss']

        net_loss = recons_loss
        self.log('reconstruction_loss', recons_loss_dict['loss'], on_epoch=True)

        nbr_cons_loss = self.nbr_consistency_loss.get(imgs, grid_sizes=grid_sizes)
        offset_reg_loss = 0.0
        if self.regularize_offset:
            offset_reg_loss = torch.norm(offset)
            offset_reg_loss = self._offset_reg_w * offset_reg_loss
            self.log('offset_reg_loss', offset_reg_loss.item(), on_epoch=True)

        if nbr_cons_loss is not None:
            nbr_cons_loss = self.nbr_consistency_w * nbr_cons_loss
            self.log('nbr_cons_loss', nbr_cons_loss.item(), on_epoch=True)
            net_loss += nbr_cons_loss + offset_reg_loss

        output = {
            'loss': net_loss,
        }
        # https://github.com/openai/vdvae/blob/main/train.py#L26
        if net_loss is None or torch.isnan(net_loss).any():
            return None

        return output
