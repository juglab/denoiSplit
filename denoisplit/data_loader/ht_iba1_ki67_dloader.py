from denoisplit.core.loss_type import LossType
from denoisplit.data_loader.ht_iba1_ki67_rawdata_loader import SubDsetType
from denoisplit.data_loader.two_dset_dloader import TwoDsetDloader


class IBA1Ki67DataLoader(TwoDsetDloader):

    def get_loss_idx(self, dset_idx):
        if self._subdset_types[dset_idx] == SubDsetType.OnlyIba1:
            loss_idx = LossType.Elbo
        elif self._subdset_types[dset_idx] == SubDsetType.Iba1Ki64:
            loss_idx = LossType.ElboMixedReconstruction
        else:
            raise Exception("Invalid subdset type")
        return loss_idx


if __name__ == '__main__':
    from denoisplit.configs.ht_iba1_ki64_config import get_config
    config = get_config()
    fpath = '/group/jug/ashesh/data/Stefania/20230327_Ki67_and_Iba1_trainingdata'
    dloader = IBA1Ki67DataLoader(
        config.data,
        fpath,
        datasplit_type=DataSplitType.Train,
        val_fraction=0.1,
        test_fraction=0.1,
        normalized_input=True,
        use_one_mu_std=True,
        enable_random_cropping=False,
        max_val=[1000, 2000],
    )
    mean_val, std_val = dloader.compute_mean_std()
    dloader.set_mean_std(mean_val, std_val)
    inp, tar, dset_idx, loss_idx = dloader[0]
    len(dloader)
    print('This is working')
