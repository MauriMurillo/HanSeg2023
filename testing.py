import SimpleITK as sitk
import numpy as np
import logging

from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

if torch.cuda.is_available():
    device = torch.device("cuda")
    all_device = True
else:
    all_device = False
    device = torch.device("cpu")

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=all_device,
    device=device,
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True,
)


# initializes the network architecture, loads the checkpoint
predictor.initialize_from_trained_model_folder(
    join(nnUNet_results, "Dataset002_OnlyCT/nnUNetTrainer__nnUNetPlans__3d_fullres"),
    checkpoint_name="checkpoint_final.pth",
    use_folds=(0,),
)

img, props = SimpleITKIO().read_images(
    [join(nnUNet_raw, "HaN-Seg/set_1/case_01/case_01_IMG_CT.nrrd")]
)
print(props)
print(img.shape)
print(img.dtype)
ret = predictor.predict_single_npy_array(img, props, None, None, False)
print(ret.shape)

retSITK = sitk.GetImageFromArray(ret)
sitk.WriteImage(
    retSITK, "/Users/mauriciomurillogonzales/Documents/GrandChallenge/HanSeg2023/output"
)

def center_crop(self, arr, shape):
        return arr[:,80:84,arr.shape[-1]//2-shape//2:arr.shape[-1]//2+shape//2, arr.shape[-1]//2-shape//2:arr.shape[-1]//2+shape//2]