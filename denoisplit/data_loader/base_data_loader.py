class BaseDataLoader:

    def per_side_overlap_pixelcount(self):
        raise NotImplementedError("Implement this for running it on notebooks")

    def get_idx_manager(self):
        raise NotImplementedError("Implement this for running it on notebooks")

    def get_grid_size(self):
        raise NotImplementedError("Implement this for running it on notebooks")
