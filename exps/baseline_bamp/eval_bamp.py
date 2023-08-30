import argparse
from genericpath import isfile
import os, sys
from pathlib import Path
from einops import rearrange
from scipy.spatial.transform import Rotation as R

import numpy as np
from bamp.bamp.eval import save_results
from bamp.bamp.eval.evaluator import Evaluator
from config  import config
from model import siMLPe as Model
from datasets.h36m_eval import H36MEval
from utils.misc import rotmat2xyz_torch, rotmat2euler_torch

import torch
from torch.utils.data import DataLoader

results_keys = ['#2', '#4', '#8', '#10', '#14', '#18', '#22', '#25']

def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

dct_m,idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)


def create_callback_fn(model):

    def smpl_callback(data):
        """
        :param data: {
            "Poses3d_in": n_in x n_person x 29 x 3
            "Masks_in": n_in x n_person
            "kitchen": kitchen
            "frames_in": n_in List[int]
            "n_out": how many output frames are supposed to be generated
        }
        """ 

        motion_input = rearrange(data["persons_in"][-50:], "t b j d -> b t (j d)")
        motion_input = torch.from_numpy(motion_input).cuda()

        outputs = []
        step = 10*25
        if step == 25:
            num_step = 1
        else:
            num_step = 25 // step + 1
        for idx in range(num_step):
            with torch.no_grad():
                if config.deriv_input:
                    motion_input_ = motion_input.clone()
                    motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input_.cuda())
                else:
                    motion_input_ = motion_input.clone()
                output = model(motion_input_)
                output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
                if config.deriv_output:
                    output = output + motion_input[:, -1:, :].repeat(1,step,1)

            outputs.append(output)
            motion_input = torch.cat([motion_input[:, step:], output], axis=1)
        motion_pred = torch.cat(outputs, axis=1)

        return rearrange(motion_pred.detach().cpu().numpy(), "b t (j d) -> t b j d", d=3)
    
    return smpl_callback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-pth', type=str, default=None, help='=encoder path')
    args = parser.parse_args()

    model = Model(config)

    state_dict = torch.load(args.model_pth)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.cuda()

    fn = create_callback_fn(model)

    data_path = Path(__file__).parent.parent.parent.parent / "bamp/data/"

    eval_fname = str(data_path / "testdata/test.json")
    assert isfile(eval_fname), eval_fname

    ev = Evaluator(eval_fname, dataset="D", data_path=data_path)

    results = ev.execute3d(fn)

    save_results("eval.pkl", results)

    

