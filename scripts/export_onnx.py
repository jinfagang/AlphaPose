import numpy as np
import torch
import alphapose
from alphapose.models import builder
from alphapose.utils.config import update_config
from onnxsim import simplify
import os
import onnx
from alfred.dl.torch.common import device


# cfg_path = 'configs/coco/resnet/256x192_res18_lr1e-3_2x_onnx.yaml'
# weight = 'weights/final_DPG.pth'
cfg_path = 'configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml'
weight = 'weights/halpe26_fast_res50_256x192.pth'
# cfg_path = 'configs/halpe_26/resnet/256x192_res18_lr1e-3_2x-regression.yaml'
# weight = 'exp/alphapose-256x192_res18_lr1e-3_2x-regression.yaml/final_DPG.pth'
onnx_model_name = weight.replace('.pth', '.onnx')

if __name__ == "__main__":
    cfg = update_config(cfg_path)
    input = torch.rand(1, 256, 192, 3).to(device)

    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    pose_model.load_state_dict(torch.load(weight, map_location=device))
    pose_model.to(device)
    pose_model.eval()

    torch.onnx.export(pose_model,               # model being run
                      # model input (or a tuple for multiple inputs)
                      input,
                      # where to save the model (can be a file or file-like object)
                      onnx_model_name,
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=12,
                      verbose=True,        # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      # the model's output names
                      output_names=['coords', 'maxvals'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'coords': {
                          0: 'batch_size'}, 'maxvals': {0: 'batch_size'}}
                      )
    print("Finish!")
    print('Simplifying model...')
    model = onnx.load(onnx_model_name)
    model_simp, check = simplify(
        model, input_shapes={'input': [2, 256, 192, 3]}, dynamic_input_shape=True)
    onnx.save(model_simp, onnx_model_name)
    print('model saved into: ', onnx_model_name)
