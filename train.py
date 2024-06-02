import torch
import os
import numpy as np
import cv2
import time
import torchvision
import hydra
import lightning as L
import wandb
import logging
from PIL import Image
from pytorch_lightning.loggers.wandb import WandbLogger
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
# @hydra.main(version_base=None, config_path="configs", config_name="pretrain_dino")
# @hydra.main(version_base=None, config_path="configs", config_name="train_smp")
def main(cfg):
	log_dir = os.path.join(cfg.log_dir, cfg.experiment_name, cfg.run_name + '_' + time.strftime("%Y-%m-%d-%H:%M:%S"))
	cfg.log_dir = log_dir
	config_dict = OmegaConf.to_container(cfg, resolve=True)
	print(config_dict, type(config_dict))
	wandb.init(project=cfg.experiment_name, name=cfg.run_name, config=config_dict)
	wandb_dir = os.path.join(log_dir, "wandb")
	ckpt_dir = os.path.join(log_dir, "ckpt")
	os.makedirs(wandb_dir, exist_ok=True)
	os.makedirs(ckpt_dir, exist_ok=True)

	dm =  projDataModule(
		image_size=cfg.image_size,
		class_mapping=cfg.class_mapping,
		**cfg.datamodule,
	)
	dm.setup()
	print('Setup DataModule')
	print(cfg)
	
	if 'resume_ckpt' in cfg:
		print('Resuming from checkpoint')
		model = SegmentorTrainer.load_from_checkpoint(cfg.resume_ckpt)
		if cfg.resume_mode == 'weights_only':
			model2 = SegmentorTrainer(
				model_type=cfg.model_type,
				class_mapping=cfg.class_mapping,
				image_size=cfg.image_size,
				class_weights=torch.Tensor(dm.res_class_ratio_train),
				**cfg.segmentation_trainer,
			)
			get_model_mean(model2)
			model = copy_params(model, model2)
			get_model_mean(model)
		print('Resuming from checkpoint completed')
      
	else:
		model = SegmentorTrainer(
			model_type=cfg.model_type,
			class_mapping=cfg.class_mapping,
			image_size=cfg.image_size,
			class_weights=torch.Tensor(dm.res_class_ratio_train),
			**cfg.segmentation_trainer,
		)

	ckpt_cb = ModelCheckpoint(
		monitor=cfg.segmentation_trainer.monitor_metric,
		dirpath=ckpt_dir,
		save_top_k=2,
		mode=cfg.segmentation_trainer.monitor_mode,
		save_last=True,
	)

	es_cb = EarlyStopping(
		monitor=cfg.segmentation_trainer.monitor_metric, 
  		patience=11, 
    	mode="min",
	)

	image_log_callback = ImageLogger(
		cfg,
	)
	wandb_logger = WandbLogger(
		project=cfg.experiment_name,
		name= cfg.run_name,
		log_model=False,
		save_dir=wandb_dir,
	)

	trainer = L.Trainer(
		accelerator="gpu",
		# callbacks=[ckpt_cb, es_cb, image_log_callback],
		callbacks=[ckpt_cb, image_log_callback],
		max_epochs=cfg.segmentation_trainer.num_epochs,
		devices=[int(cfg.gpu_ids)],
		logger=wandb_logger,
		precision=cfg.precision,
	)

	trainer.fit(model, dm)
 
if __name__ == "__main__":
    main()