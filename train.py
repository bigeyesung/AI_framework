import torch
import os
import numpy as np
import cv2
import time
import torchvision
import hydra
import wandb
import logging
from PIL import Image
from torchvision import transforms as v2
from src.data.datamodule import projDataModule
from src.model.SegmentationTrainer import SegmentorTrainer
from src.logger.img_logger import ImageLogger
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)

torchvision.disable_beta_transforms_warning()

def get_model_mean(model):
	mean = 0
	named_params_model = dict(model.model.named_parameters())
	for name, param1 in named_params_model.items():
		if 'class_head' not in name:
			mean += param1.data.mean()
	print(mean)
 
def copy_params(model1, model2):
	named_params_model1 = dict(model1.model.named_parameters())
	named_params_model2 = dict(model2.model.named_parameters())

	state_dict_model2 = model2.model.state_dict()
	for name, param1 in named_params_model1.items():
		if name in named_params_model2:
			param2 = named_params_model2[name]
			if param1.shape == param2.shape:
				state_dict_model2[name].copy_(param1.data)

	model2.model.load_state_dict(state_dict_model2)
	return model2

@hydra.main(version_base=None, config_path="configs", config_name="train_dino")
def main(cfg):
	log_dir = os.path.join(cfg.log_dir, cfg.experiment_name, cfg.run_name + '_' + time.strftime("%Y-%m-%d-%H:%M:%S"))
	cfg.log_dir = log_dir
	config_dict = OmegaConf.to_container(cfg, resolve=True)
	print(config_dict, type(config_dict))
	ckpt_dir = os.path.join(log_dir, "ckpt")
	os.makedirs(ckpt_dir, exist_ok=True)

	dm =  projDataModule(
		image_size=cfg.image_size,
		class_mapping=cfg.class_mapping,
		**cfg.datamodule,
	)
	dm.setup()
	print('Setup DataModule')
	print(cfg)


	#construct model
	model = SegmentorTrainer(
		model_type=cfg.model_type,
		class_mapping=cfg.class_mapping,
		image_size=cfg.image_size,
		class_weights=torch.Tensor(dm.res_class_ratio_train),
		**cfg.segmentation_trainer,
	)

	for ind in range(200):
		#do sth
		pass
 
if __name__ == "__main__":
    main()