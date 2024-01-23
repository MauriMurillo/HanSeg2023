import SimpleITK as sitk
import numpy as np
import logging

from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

logger = logging.getLogger(__name__)

from custom_algorithm import Hanseg2023Algorithm

LABEL_dict = {
    "background": 0,
    "A_Carotid_L": 1,
    "A_Carotid_R": 2,
    "Arytenoid": 3,
    "Bone_Mandible": 4,
    "Brainstem": 5,
    "BuccalMucosa": 6,
    "Cavity_Oral": 7,
    "Cochlea_L": 8,
    "Cochlea_R": 9,
    "Cricopharyngeus": 10,
    "Esophagus_S": 11,
    "Eye_AL": 12,
    "Eye_AR": 13,
    "Eye_PL": 14,
    "Eye_PR": 15,
    "Glnd_Lacrimal_L": 16,
    "Glnd_Lacrimal_R": 17,
    "Glnd_Submand_L": 18,
    "Glnd_Submand_R": 19,
    "Glnd_Thyroid": 20,
    "Glottis": 21,
    "Larynx_SG": 22,
    "Lips": 23,
    "OpticChiasm": 24,
    "OpticNrv_L": 25,
    "OpticNrv_R": 26,
    "Parotid_L": 27,
    "Parotid_R": 28,
    "Pituitary": 29,
    "SpinalCord": 30,
}

nnUnetDict = {
    "background": 0,
    "A_Carotid_L": 1,
    "SpinalCord": 2,
    "Brainstem": 3,
    "Parotid_L": 4,
    "BuccalMucosa": 5,
    "Cricopharyngeus": 6,
    "OpticChiasm": 7,
    "Esophagus_S": 8,
    "OpticNrv_R": 9,
    "Eye_PL": 10,
    "A_Carotid_R": 11,
    "Larynx_SG": 12,
    "Glnd_Lacrimal_L": 13,
    "Glnd_Lacrimal_R": 14,
    "Glnd_Thyroid": 15,
    "Eye_AL": 16,
    "Glnd_Submand_L": 17,
    "Bone_Mandible": 18,
    "Pituitary": 19,
    "Arytenoid": 20,
    "Parotid_R": 21,
    "Cochlea_L": 22,
    "Lips": 23,
    "OpticNrv_L": 24,
    "Glnd_Submand_R": 25,
    "Glottis": 26,
    "Cochlea_R": 27,
    "Eye_PR": 28,
    "Cavity_Oral": 29,
    "Eye_AR": 30,
}


class MyHanseg2023Algorithm(Hanseg2023Algorithm):
    def __init__(self):
        # instantiate the nnUNetPredictor
        if torch.cuda.is_available():
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=device,
            verbose=True,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        # initializes the network architecture, loads the checkpoint
        self.predictor.initialize_from_trained_model_folder(
            join(
                nnUNet_results,
                "Dataset002_OnlyCT/nnUNetTrainer__nnUNetPlans__3d_fullres",
            ),
            checkpoint_name="checkpoint_best.pth",
            use_folds=(0,),
        )

        super().__init__()

    def predict(self, *, image_ct: sitk.Image, image_mrt1: sitk.Image) -> sitk.Image:
        # create an empty segmentation same size as ct image
        images = []
        spacings = []
        spacings_for_nnunet = []

        npy_image = sitk.GetArrayFromImage(image_ct)
        spacings.append(image_ct.GetSpacing())
        npy_image = npy_image[None]
        spacings_for_nnunet.append(list(spacings[-1])[::-1])

        images.append(npy_image)
        spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        dict = {
            "sitk_stuff": {
                "spacing": spacings[0],
            },
            "spacing": spacings_for_nnunet[0],
        }

        img = np.vstack(images).astype(np.float32)
        npy_image = None
        del npy_image

        image_mrt1 = None
        output_seg = self.predictor.predict_single_npy_array(
            img, dict, None, None, False
        )

        output_seg = output_seg * 100
        for key, number in nnUnetDict.items():
            output_seg[output_seg == (number * 100)] = LABEL_dict[key]
        output_seg = sitk.GetImageFromArray(output_seg)
        output_seg.CopyInformation(image_ct)


        # output should be a sitk image with the same size, spacing, origin and direction as the original input image_ct
        output_seg = sitk.Cast(output_seg, sitk.sitkUInt8)
        return output_seg


if __name__ == "__main__":
    MyHanseg2023Algorithm().process()
 