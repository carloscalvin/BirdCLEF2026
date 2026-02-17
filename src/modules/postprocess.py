import numpy as np

class PostProcessor:
    def __init__(self, cfg):
        self.apply_postprocess = cfg.apply_postprocess
        self.post_top_k = cfg.post_top_k
        self.post_exponent = cfg.post_exponent

    def apply_power_to_low_ranked_cols(self, probs):
        if not self.apply_postprocess:
            return probs

        p = probs.copy()
        max_per_species = p.max(axis=0)
        tail_cols = np.argsort(-max_per_species)[self.post_top_k:]
        p[:, tail_cols] = p[:, tail_cols] ** self.post_exponent
        return p

    def run(self, probs):
        probs = self.apply_power_to_low_ranked_cols(probs)
        return probs