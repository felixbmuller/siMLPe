
from pathlib import Path
import sys
from einops import rearrange
import numpy as np
from bamp.data import PersonSequences
from bamp.data.kitchen import Kitchen, KitchenObjectType
from tqdm import tqdm
from typing import List, Dict
from torch.utils.data import Dataset
from typing import Dict
from bamp.transforms.transforms import normalize, apply_normalization_to_points3d
from bamp.data import frames2segments

from scipy.spatial import KDTree


class BasisPointSet:
    def __init__(self):
        self.basis_points = np.array(
            [
                (-0.831, 0.057, 0.699),
                (-0.675, -0.430, 0.379),
                (0.898, 0.176, 0.060),
                (0.372, 0.819, 0.038),
                (-0.578, -0.217, 0.285),
                (-0.723, 0.536, 0.225),
                (-0.059, 0.014, 0.881),
                (0.233, 0.720, 0.524),
                (0.986, 0.611, 0.334),
                (0.191, -0.207, 0.497),
                (-0.509, -0.409, 0.668),
                (0.804, 0.301, 0.561),
                (-0.116, 0.038, 0.628),
                (-0.863, -0.618, 0.589),
                (0.804, 0.351, 0.780),
                (-0.394, -0.637, 1.085),
                (0.261, 0.971, 0.271),
                (-0.072, 0.902, 0.402),
                (0.431, -0.364, 0.473),
                (-0.144, -0.185, 0.704),
                (-1.548, 0.571, 0.090),
                (0.798, -0.587, 0.160),
                (-0.303, -0.315, 0.237),
                (0.237, 0.026, 0.794),
                (-0.307, 0.284, 0.358),
                (-0.994, -0.213, 0.497),
                (-0.342, -0.497, 0.106),
                (0.072, -0.265, 0.197),
                (0.738, 0.012, 0.250),
                (-0.295, -0.199, 0.183),
                (0.264, -0.059, 0.240),
                (0.225, -0.352, 0.843),
                (0.306, 0.143, 0.264),
                (0.174, -0.298, 0.248),
                (-0.348, 0.207, 0.437),
                (-0.097, -1.099, 0.193),
                (0.012, -0.469, 0.757),
                (1.097, 0.510, 0.073),
                (0.719, 0.268, 0.005),
                (-0.294, 0.450, 0.020),
                (-0.377, 0.636, 0.457),
                (-0.087, 0.017, 0.447),
                (-0.451, 0.669, 0.809),
                (-0.062, -0.159, 0.694),
                (0.364, -0.878, 0.237),
                (0.054, 0.979, 0.004),
                (-0.549, 0.569, 0.328),
                (0.209, 0.143, 0.161),
                (0.008, 0.033, 0.001),
                (-0.678, -0.741, 0.611),
                (-0.115, 0.247, 0.389),
                (0.195, 0.136, 0.630),
                (0.211, -0.044, 0.213),
                (-0.228, 0.174, 1.230),
                (0.225, 0.424, 0.955),
                (-0.537, 0.143, 0.561),
                (-0.354, 0.249, 0.246),
                (-0.285, 1.234, 0.090),
                (0.167, 0.283, 0.574),
                (-0.298, -0.193, 0.003),
                (-0.545, 0.271, 0.835),
                (-0.484, 0.134, 0.323),
                (-0.901, 0.378, 0.065),
                (-0.526, 0.671, 0.273),
                (-0.432, 0.742, 0.144),
                (0.054, -0.474, 0.480),
                (-0.505, 0.809, 0.210),
                (0.410, -0.361, 0.348),
                (0.519, 0.216, 0.292),
                (-0.343, 0.245, 0.171),
                (-0.222, 1.705, 0.165),
                (0.560, 0.156, 0.148),
                (0.692, 0.571, 0.285),
                (-0.550, 0.316, 1.073),
                (0.534, -0.486, 0.447),
                (-0.550, 0.326, 0.010),
                (-0.620, -0.750, 0.047),
                (-0.842, 0.124, 0.008),
                (-0.276, -0.573, 0.074),
                (0.679, 0.580, 0.455),
                (0.396, -0.706, 0.603),
                (0.399, -0.077, 0.324),
                (1.438, -1.033, 0.021),
                (0.214, 0.437, 0.100),
                (-0.138, 0.219, 0.454),
                (-0.531, 0.135, 0.472),
                (-0.180, -0.230, 0.498),
                (-0.170, -1.094, 0.575),
                (-0.515, -1.355, 0.133),
                (0.620, -0.752, 1.101),
                (-0.977, 0.726, 0.585),
                (0.476, -0.376, 0.540),
                (0.368, -0.450, 0.346),
                (0.313, -0.079, 0.152),
                (-0.827, -0.673, 0.377),
                (0.052, -0.337, 0.607),
                (0.016, -0.322, 0.672),
                (-1.009, -1.134, 0.922),
                (0.284, 0.874, 0.085),
                (0.461, -0.098, 0.345),
                (0.770, -0.261, 0.043),
                (0.331, 0.161, 0.040),
                (-0.168, 0.211, 0.085),
                (0.381, 0.319, 0.506),
                (0.338, 0.160, 0.468),
                (-0.281, 0.824, 0.097),
                (-0.955, 0.346, 0.195),
                (-0.892, -0.672, 0.401),
                (0.675, -0.180, 0.390),
                (-0.010, -0.244, 0.123),
                (0.099, -0.410, 0.464),
                (-0.289, 0.060, 0.788),
                (-0.342, 0.349, 1.001),
                (-0.789, -0.334, 0.684),
                (-0.115, -0.202, 0.105),
                (-0.134, 0.403, 0.160),
                (-0.438, -0.248, 0.180),
                (0.530, -0.923, 0.047),
                (0.432, 0.399, 0.331),
                (0.390, -0.708, 0.298),
                (-0.828, -0.259, 0.020),
                (0.128, 0.515, 0.092),
                (0.369, -1.121, 0.096),
                (0.045, -0.565, 0.183),
                (-0.053, -0.210, 0.168),
                (-0.390, -0.224, 0.063),
                (-0.704, -0.082, 0.772),
                (0.144, 0.436, 0.717),
            ]
        )
        print("self.BPS", self.basis_points.shape)

    def query(self, pts3d):
        """
        :param pts3d: {n_points x 3}
        """
        basis_points = self.basis_points
        tree = KDTree(pts3d)
        i, _ = tree.query(basis_points)
        return np.array(i, dtype=np.float32)


BASIS_POINT_SET = BasisPointSet()


person_seqs = PersonSequences(
    person_path=str(Path(__file__).parent.parent.parent.parent / "bamp/data/poses")
)

kitchens = {}

for dataset in tqdm(["A", "B", "C", "D"]):
    kitchens[dataset] = Kitchen.load_for_dataset(
        dataset=dataset,
        data_location=str(Path(__file__).parent.parent.parent.parent / "bamp/data/scenes")
    )




def get_normalized_sequences(
    seq, actions: List[str], n_in: int, n_out: int, skips: int
):
    All_seqs = []
    
    frames = seq.get_frames_where_action(actions=actions)
    indices = [seq.frame2index[frame] for frame in frames]
    poses = seq.poses3d   
    for start, end in frames2segments(indices):
        for index in range(start, end-(n_in + n_out), skips):
            seq3d = poses[index:index+n_in + n_out].copy()
            seq3d, (mu, R) = normalize(
                seq3d,
                frame=n_in-1,
                return_transform=True
            )
            All_seqs.append({
                "seq3d": seq3d,
                "mu": mu,
                "R": R,
                "dataset": seq.dataset,
                "frame": seq.index2frame[index] + n_in -1
            })
    return All_seqs


class SittingDataset(Dataset):
    
    def __init__(
        self, 
        person_seqs: PersonSequences, 
        kitchens: Dict[str, Kitchen], 
        datasets: List[str],
        n_in: int,
        n_out: int
    ):
        """
        """
        self.n_in = n_in
        sitting_data = []
        not_sitting_data = []
        self.kitchens = kitchens
        
        self.basis_point_set = BasisPointSet()
        
        for dataset in datasets:
            for seq in person_seqs.get_sequences(dataset):
                norm_seqs = get_normalized_sequences(
                    seq,
                    actions=["sitting down", "sitting"], 
                    n_in=n_in,
                    n_out=n_out,
                    skips=10,
#                     skips=10000,
                )
                sitting_data += norm_seqs
                
                norm_seqs = get_normalized_sequences(
                    seq,
                    actions=["walking", "standing", "leaning"], 
                    n_in=n_in,
                    n_out=n_out,
                    skips=17
#                     skips=17000
                )
                not_sitting_data += norm_seqs
        
        print("#sitting_data", len(sitting_data))
        print("#not_sitting_data", len(not_sitting_data))
        self.all_data = sitting_data + not_sitting_data
        
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        out = self.all_data[index]
        
        seq3d = rearrange(out["seq3d"], "t j d -> t (j d)")
        mu = out["mu"]
        R = out["R"]
        dataset = out['dataset']
        frame = out['frame']
        
        kitchen = self.kitchens[dataset]
        
        objs = []
        
        all_sittable3d = []
        all_tables3d = []
        all_nonsittable3d = []
        
        for obj in kitchen.get_environment(frame=frame):
            samples = obj.query()
            if obj.obj_type == KitchenObjectType.SITTABLE:
                all_sittable3d.append(samples)
            elif obj.obj_type == KitchenObjectType.TABLE:
                all_tables3d.append(samples)
            else:
                all_nonsittable3d.append(samples)
        
        all_sittable3d = np.concatenate(all_sittable3d, axis=0)
        all_tables3d = np.concatenate(all_tables3d, axis=0)
        all_nonsittable3d = np.concatenate(all_nonsittable3d, axis=0)
        
        
        all_sittable3d = apply_normalization_to_points3d(
            pts3d=all_sittable3d,
            mu=mu,
            R=R
        )
        all_tables3d = apply_normalization_to_points3d(
            pts3d=all_tables3d,
            mu=mu,
            R=R
        )
        all_nonsittable3d = apply_normalization_to_points3d(
            pts3d=all_nonsittable3d,
            mu=mu,
            R=R
        )
        
        sittable_feature = self.basis_point_set.query(all_sittable3d)
        tables_feature = self.basis_point_set.query(all_tables3d)
        otherobjects_feature = self.basis_point_set.query(all_nonsittable3d)
        
        seq_in = seq3d[:self.n_in]
        seq_out = seq3d[self.n_in:]

        return seq_in, seq_out
        
        # return {
        #     "seq_in": seq_in,
        #     "seq_out": seq_out,
        #     "sittable_feature": sittable_feature,
        #     "tables_feature": tables_feature,
        #     "otherobjects_feature": otherobjects_feature,
        # }
        
    
    
    

