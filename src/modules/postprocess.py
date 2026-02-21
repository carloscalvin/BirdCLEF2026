import numpy as np
import pandas as pd
from tqdm import tqdm

class PostProcessor:
    def __init__(self, cfg):
        self.apply_postprocess = cfg.apply_postprocess
        self.post_top_k = cfg.post_top_k
        self.post_exponent = cfg.post_exponent
        self.apply_smoothing = cfg.apply_smoothing
        self.smoothing_weights = cfg.smoothing_weights
        self.use_sliding = cfg.use_sliding_window_infer
    
    def aggregate_sliding_windows(self, probs, row_ids):
        if not self.use_sliding:
            return probs, row_ids

        print("[*] Agregando inferencias de sliding window...")
        real_row_ids = [rid.split('|')[0] if '|' in rid else rid for rid in row_ids]
        
        num_classes = probs.shape[1]
        col_names = [f"c{i}" for i in range(num_classes)]
        
        df = pd.DataFrame(probs, columns=col_names)
        df['row_id'] = real_row_ids
        
        grouped = df.groupby('row_id')[col_names].mean().reset_index()
        
        unique_ids = list(dict.fromkeys(real_row_ids))
        grouped['row_id_cat'] = pd.Categorical(grouped['row_id'], categories=unique_ids, ordered=True)
        grouped = grouped.sort_values('row_id_cat').drop(columns=['row_id_cat'])
        
        return grouped[col_names].values.astype(np.float32), grouped['row_id'].tolist()
    
    def apply_temporal_smoothing(self, probs, row_ids):
        if not self.apply_smoothing:
            return probs

        print(f"[*] Aplicando suavizado temporal con pesos {self.smoothing_weights}...")
        num_classes = probs.shape[1]
        col_names = [f"c{i}" for i in range(num_classes)]
        
        df = pd.DataFrame(probs, columns=col_names)
        df['row_id'] = row_ids
        df['group_key'] = df['row_id'].apply(lambda x: x.rsplit('_', 1)[0])
        df['end_sec'] = df['row_id'].apply(lambda x: int(x.rsplit('_', 1)[1]))

        df['orig_index'] = df.index
        df = df.sort_values(['group_key', 'end_sec'])

        w_prev, w_curr, w_next = self.smoothing_weights
        groups = df.groupby('group_key')
        results_list = []
        
        for _, group in tqdm(groups, desc="Smoothing", leave=False):
            if len(group) < 2:
                results_list.append(group)
                continue

            mat = group[col_names].values.copy()
            new_mat = mat.copy()

            new_mat[1:-1] = (mat[0:-2] * w_prev) + (mat[1:-1] * w_curr) + (mat[2:] * w_next)
            
            new_mat[0] = (mat[0] * (w_curr + w_prev)) + (mat[1] * w_next)
            new_mat[-1] = (mat[-1] * (w_curr + w_next)) + (mat[-2] * w_prev)
            
            group_res = group.copy()
            group_res.loc[:, col_names] = new_mat
            results_list.append(group_res)

        df_smoothed = pd.concat(results_list)
        df_smoothed = df_smoothed.sort_values('orig_index')
        
        return df_smoothed[col_names].values.astype(np.float32)

    def apply_power_to_low_ranked_cols(self, probs):
        if not self.apply_postprocess:
            return probs

        p = probs.copy()
        max_per_species = p.max(axis=0)
        tail_cols = np.argsort(-max_per_species)[self.post_top_k:]
        p[:, tail_cols] = p[:, tail_cols] ** self.post_exponent
        return p

    def run(self, probs, row_ids):
        probs, row_ids = self.aggregate_sliding_windows(probs, row_ids)
        probs = self.apply_temporal_smoothing(probs, row_ids)        
        probs = self.apply_power_to_low_ranked_cols(probs)

        return probs, row_ids