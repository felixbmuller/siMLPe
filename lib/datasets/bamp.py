from collections import defaultdict
from pathlib import Path
import pickle
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
from bamp.data.constants import activity2index
from multipool import MultiPool

from scipy.spatial import KDTree

OBJECT_FRAME = 24
OBJ_TYPE_COUNT = 12  # or 12?


def encode_objects_direction(poses, masks, start_frames, kitchen):
    """_summary_

    Parameters
    ----------
    poses : Tensor(batch, persons, frames, joints, dim)
        _description_
    masks : ndarray(batch, persons, frames)
    start_frames : ndarray(batch)
        _description_
    kitchen : Kitchen
        _description_

    Returns
    -------
    _type_
        _description_
    """

    object_embed = np.full((*poses.shape[:3], OBJ_TYPE_COUNT, 3), 1000)

    #print(f"Created target array of shape {object_embed.shape}")

    for iscene in range(poses.shape[0]):
        # print("iscene", iscene)

        objs_raw = kitchen.get_environment(start_frames[iscene] + OBJECT_FRAME)
        objs_labels = [np.argmax(obj.label).item() for obj in objs_raw]
        objs_trees = [KDTree(obj.query()) for obj in objs_raw]

        # print("objs_labels", objs_labels)

        for iperson in range(poses.shape[1]):
            if np.mean(masks[iscene, iperson]) <= 0.999:
                continue

            # print(f"person {iperson} not skipped")

            embed_per_label = defaultdict(lambda: (float("inf"), None))

            for tree, label in zip(objs_trees, objs_labels):
                dist, indices = tree.query(poses[iscene, iperson, :, 0])

                best_dist, _ = embed_per_label[label]

                if dist[OBJECT_FRAME] < best_dist:
                    embed_per_label[label] = (dist[OBJECT_FRAME], tree.data[indices])

            # print(f"embed_per_label results {[(k, d) for k, (d, i) in embed_per_label.items()]}")

            for label, (dist, points) in embed_per_label.items():
                # print("points.shape", points.shape)

                direction = poses[iscene, iperson, :, 0] - points  # (frames, 3)

                object_embed[iscene, iperson, :, label, :] = direction

    return object_embed


person_seqs = PersonSequences(
    person_path=str(Path(__file__).parent.parent.parent.parent / "bamp/data/poses")
)

kitchens = {}

for dataset in tqdm(["A", "B", "C", "D"]):
    kitchens[dataset] = Kitchen.load_for_dataset(
        dataset=dataset,
        data_location=str(
            Path(__file__).parent.parent.parent.parent / "bamp/data/scenes"
        ),
    )


def get_normalized_sequences(
    data
):
    seq = data["seq"]
    actions = data["actions"]
    n_in = data["n_in"]
    n_out = data["n_out"]
    skips = data["skips"]
    kitchen = data["kitchen"]

    All_seqs = []

    frames = seq.get_frames_where_action(actions=actions)
    indices = [seq.frame2index[frame] for frame in frames]
    poses = seq.poses3d
    for start, end in frames2segments(indices):
        for index in range(start, end - (n_in + n_out), skips):
            seq3d = poses[index : index + n_in + n_out].copy()

            if kitchen is not None:
                obj_enc = encode_objects_direction(
                    rearrange(seq3d, "t j d -> 1 1 t j d"),
                    np.ones((1, 1, seq3d.shape[0])), [index], kitchen,
                ).astype(np.float32)

                seq3d = np.concatenate([seq3d, obj_enc[0, 0]], axis=1)

            seq3d, (mu, R) = normalize(
                seq3d,
                frame=n_in - 1,
                return_transform=True,
                check_shape=False,
            )
            All_seqs.append(
                {
                    "seq3d": seq3d,
                    "mu": mu,
                    "R": R,
                    "dataset": seq.dataset,
                    "frame": seq.index2frame[index] + n_in - 1,
                }
            )
    print(".", end="", flush=True)

    
    return All_seqs


class SittingDataset(Dataset):
    def __init__(
        self,
        person_seqs: PersonSequences,
        kitchens: Dict[str, Kitchen],
        datasets: List[str],
        n_in: int,
        n_out: int,
        encode_objects=False,
    ):
        """ """
        self.n_in = n_in
        sitting_data = []
        # not_sitting_data = []
        self.kitchens = kitchens

        # self.basis_point_set = BasisPointSet()

        for dataset in datasets:

            requests = []
            for seq in person_seqs.get_sequences(dataset):
                requests.append(dict(
                    seq=seq,
                    actions=list(activity2index.keys()),
                    n_in=n_in,
                    n_out=n_out,
                    skips=100,
                    kitchen=kitchens[dataset] if encode_objects else None
                    #                     skips=10000,
                ))

            with MultiPool(20, "mock") as pool:

                results = pool.imap(get_normalized_sequences, requests, 20)

            for norm_seqs in results:

                sitting_data += norm_seqs

        print("#data", len(sitting_data))
        # print("#not_sitting_data", len(not_sitting_data))
        self.all_data = sitting_data  # + not_sitting_data

        with open("bamp_objects_processed.pkl", "wb") as fp:

            pickle.dump(sitting_data, fp)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        out = self.all_data[index]

        seq3d = rearrange(out["seq3d"], "t j d -> t (j d)")
        mu = out["mu"]
        R = out["R"]
        dataset = out["dataset"]
        frame = out["frame"]

        kitchen = self.kitchens[dataset]

        seq_in = seq3d[: self.n_in]
        seq_out = seq3d[self.n_in :]

        return seq_in, seq_out

        # return {
        #     "seq_in": seq_in,
        #     "seq_out": seq_out,
        #     "sittable_feature": sittable_feature,
        #     "tables_feature": tables_feature,
        #     "otherobjects_feature": otherobjects_feature,
        # }
