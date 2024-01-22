from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class Hanseg2023Algorithm:
    def __init__(
        self,
        input_path="/Users/mauriciomurillogonzales/Documents/GrandChallenge/HanSeg2023/input/images/ct",
        output_path="/Users/mauriciomurillogonzales/Documents/GrandChallenge/HanSeg2023/output",
    ):
        self.input_path = input_path
        self.output_path = output_path
        # instantiate the nnUNetPredictor
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device("cpu"),
            verbose=False,
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

    def predict(self):
        # variant 1: give input and output folders
        self.predictor.predict_from_files(
            [
                [join(self.input_path, "case_01_IMG_CT.nrrd")],
                [join(self.input_path, "case_02_IMG_CT.nrrd")],
            ],
            [
                join(self.output_path, "case_01_IMG_CT_seg.nrrd"),
                join(self.output_path, "case_02_IMG_CT_seg.nrrd"),
            ],
            save_probabilities=False,
            overwrite=False,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )


if __name__ == "__main__":
    Hanseg2023Algorithm().predict()
