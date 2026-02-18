import os
from os import PathLike
from pathlib import Path
from typing import Union, List
import monai
import torch
from monai.bundle import ConfigParser
import yaml
import re
import json
import sys
from monai.transforms import AsDiscrete

def create_image_list(datalist,keys=None):
    """
    Create a new list of dictionaries with only the image key from the original datalist.
    
    Args:
        datalist (list): A list of dictionaries containing image data.
        
    Returns:
        list: A new list of dictionaries with only the image key.
    """
    new_datalist = []

    for i in range(len(datalist)):
        data = datalist[i]
        new_data = {}
        if "image" in data:
            new_data["image"] = data["image"]
        else:
            new_data["image"] = data["ct"][:-len("_0000.nii.gz")] + "_image.nii.gz"
        if keys is not None:
            for key in keys:
                new_data[key] = data[key]
        new_datalist.append(new_data)
    
    return new_datalist

def filter_datalist(datalist, filtering_label="label", keys=None):
    filtered_datalist = []
    for data in datalist:
        if len(data[filtering_label]) > 0:
            filtered_datalist.append(data)
    filtered_datalist = create_image_list(filtered_datalist, keys=keys)
    return filtered_datalist

def change_key(datalist, old_key, new_key):
    """
    Change the key of a dictionary in a list of dictionaries.

    Args:
        datalist (list): A list of dictionaries.
        old_key (str): The key to be changed.
        new_key (str): The new key.

    Returns:
        list: A new list of dictionaries with the updated key.
    """
    new_datalist = []
    for data in datalist:
        if old_key in data:
            data[new_key] = data.pop(old_key)
        new_datalist.append(data)
    return new_datalist


def threshold_CT(x):
    # threshold at 1
    return x > 0.1

def mlflow_transform_det(state_output):
    return state_output["loss"]

def mlflow_transform(state_output):
    return state_output[0]["loss"]

def prepare_batch(batch_data, device, non_blocking=False):
    inputs, inputs_2, gt_input = (
            batch_data["image"].to(device),
            batch_data["image_2"].to(device),
            batch_data["gt_image"].to(device),
        )
    
    return inputs, inputs_2, gt_input

def prepare_batch_seg(batch_data, device, non_blocking=False):
    inputs, gt_input = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
    
    return inputs, gt_input

def prepare_val_batch(batch_data, device, non_blocking=False):
    inputs, gt_input = (
            batch_data["image"].to(device),
            batch_data["gt_image"].to(device),
        )
    
    return inputs, gt_input

def prepare_val_batch_seg(batch_data, device, non_blocking=False):
    inputs, gt_input = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
    
    return inputs, gt_input

def subfiles(
        folder: Union[str, PathLike], join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True
) -> List[str]:
    """
    Given a folder path, returns a list with all the files in the folder.

    Parameters
    ----------
    folder :
        Folder path.
    join :
        Flag to return the complete file paths or only the relative file names.
    prefix :
        Filter the files with the specified prefix.
    suffix :
        Filter the files with the specified suffix.
    sort :
        Flag to sort the files in the list by alphabetical order.

    Returns
    -------
        Filename list.
    """
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E741, E731
    res = [
        l(folder, i.name)
        for i in Path(folder).iterdir()
        if i.is_file() and (prefix is None or i.name.startswith(prefix)) and (suffix is None or i.name.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def get_checkpoint(epoch, ckpt_dir):
    """
    Retrieves the checkpoint for a given epoch from the checkpoint directory.

    Parameters
    ----------
    epoch : int or str
        The epoch number to retrieve. If 'latest', the function will return the latest checkpoint.
    ckpt_dir : str
        The directory where checkpoints are stored.

    Returns
    -------
    int
        The epoch number of the checkpoint to be retrieved. If 'latest', returns the latest epoch number.
    """
    if epoch == "latest":

        latest_checkpoints = subfiles(ckpt_dir, prefix="checkpoint_epoch", sort=True,
                                      join=False)
        epochs = []
        for latest_checkpoint in latest_checkpoints:
            epochs.append(int(latest_checkpoint[len("checkpoint_epoch="):-len(".pt")]))

        epochs.sort()
        latest_epoch = epochs[-1]
        return latest_epoch
    else:
        return epoch

def reload_checkpoint(trainer, epoch, num_train_batches_per_epoch, ckpt_dir, lr_scheduler=None):
    """
    Reloads the checkpoint for a given epoch and updates the trainer's state.

    Parameters
    ----------
    trainer : object
        The trainer object whose state needs to be updated.
    epoch : int
        The epoch number to load the checkpoint from.
    num_train_batches_per_epoch : int
        The number of training batches per epoch.
    ckpt_dir : str
        The directory where the checkpoints are stored.
    lr_scheduler : object, optional
        The learning rate scheduler to be updated (default is None).

    Returns
    -------
    None
    """

    epoch_to_load = get_checkpoint(epoch, ckpt_dir)
    trainer.state.epoch = epoch_to_load
    trainer.state.iteration = (epoch_to_load* num_train_batches_per_epoch) +1
    
    if lr_scheduler is not None:
        lr_scheduler.ctr = epoch_to_load
        lr_scheduler.step(epoch_to_load)

def tb_batch_transform(batch):
    return [batch_id["image"] for batch_id in batch], [batch_id["gt_image"] for batch_id in batch]

def tb_output_transform(output):
    return [out["pred"] for out in output]

def tb_batch_transform_seg(batch):
    return [batch_id["image"] for batch_id in batch], [batch_id["label"] for batch_id in batch]

def dice_output_transform(output):
    return [out['pred'] for out in output], [out['label'] for out in output]

def dice_output_transform_det(output):
    return [AsDiscrete(argmax=True)(out['logits']) for out in output['pred']],[AsDiscrete(to_onehot=2)(out['instance_seg']) for out in output['label']]

def create_mlflow_experiment_params(params_file, custom_params=None):
    params_dict = {}
    config_values = monai.config.deviceconfig.get_config_values()
    for k in config_values:
        params_dict[re.sub("[()]"," ",str(k))] = config_values[k]
    optional_config_values = monai.config.deviceconfig.get_optional_config_values()

    for k in optional_config_values:
        params_dict[re.sub("[()]"," ",str(k))] = optional_config_values[k]

    gpu_info = monai.config.deviceconfig.get_gpu_info()
    for k in gpu_info:
        params_dict[re.sub("[()]"," ",str(k))] = str(gpu_info[k])

    yaml_config_files = [params_file]
    # %%
    monai_config = {}
    for config_file in yaml_config_files:
        with open(config_file, 'r') as file:
            monai_config.update(yaml.safe_load(file))

    monai_config["bundle_root"] = str(Path(Path(params_file).parent).parent)

    parser = ConfigParser(monai_config, globals={"os": "os",
                                                 "pathlib": "pathlib",
                                                 "json": "json",
                                                 "ignite": "ignite"
                                                 })

    parser.parse(True)

    for k in monai_config:
        params_dict[k] = parser.get_parsed_content(k,instantiate=True)

    if custom_params is not None:
        for k in custom_params:
            params_dict[k] = custom_params[k]
    return params_dict


def get_lightning_checkpoint(epoch, ckpt_dir):
    if epoch == "latest":

        latest_checkpoints = subfiles(ckpt_dir, prefix="epoch=", sort=True,
                                      join=False)
        epochs = []
        for latest_checkpoint in latest_checkpoints:
            try:
                epochs.append(int(latest_checkpoint[len("epoch="):-len(".ckpt")]))
            except:
                ...

        epochs.sort()
        latest_epoch = epochs[-1]
        return latest_epoch
    else:
        return epoch
    
def load_ssl_checkpoint(ssl_ckpt_dir, epoch, model):
    """
    Load the SSL checkpoint for a given epoch.

    Parameters
    ----------
    ssl_ckpt_dir : str
        The directory where the SSL checkpoints are stored.
    epoch : int or str
        The epoch number to load. If 'latest', the function will return the latest checkpoint.

    Returns
    -------
    dict
        The loaded SSL checkpoint.
    """
    ssl_epoch_to_load = get_checkpoint(epoch, ssl_ckpt_dir)
    
    ckpt_path = os.path.join(ssl_ckpt_dir, f"checkpoint_epoch={ssl_epoch_to_load}.pt")
    
    torch_ckpt = torch.load(ckpt_path)
    model_state_dict = torch_ckpt["model"]
    # Check if model weights start with 'swinViT.' and if the keys after 'swinViT.' are in the model's state_dict
    missing_keys = []
    matched_keys = []
    for key in model.state_dict():
        if key.startswith("swinViT."):
            subkey = key[len("swinViT."):]
            if subkey not in model_state_dict:
                missing_keys.append(subkey)
            else:
                matched_keys.append(subkey)
                # Build a new state dict with the correct keys
    new_state_dict = model.state_dict()
    for key in new_state_dict:
        if key.startswith("swinViT."):
            subkey = key[len("swinViT."):]
            if subkey in model_state_dict:
                new_state_dict[key] = model_state_dict[subkey]
    model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print(f"Warning: The following swinViT subkeys are missing in the model's state_dict: {missing_keys}", file=sys.stderr)
    if matched_keys:
        print(f"Warning: The following swinViT subkeys are matched in the model's state_dict: {matched_keys}", file=sys.stderr)


class PrepareBatch:

    def __init__(self, image_key, label_key, box_key, validation=False, whole_image_validation= False):
        self.image_key = image_key
        self.label_key = label_key
        self.box_key = box_key
        self.validation = validation
        self.whole_image_validation = whole_image_validation

    def __call__(self, batch, device, non_blocking):

        if not self.whole_image_validation:
            img = [batch_element[self.image_key].to(device, non_blocking=non_blocking) for batch_i in batch for batch_element in batch_i]

            targets = [
                {
                    self.label_key: batch_element[self.label_key].to(device, non_blocking=non_blocking),
                    self.box_key: batch_element[self.box_key].to(device, non_blocking=non_blocking)
                }
                for batch_i in batch for batch_element in batch_i]
        else:
            img = [batch_element[self.image_key].to(device, non_blocking=non_blocking) for
                   batch_element in batch]
            
            targets = [
                {
                    self.label_key: batch_element[self.label_key].to(device, non_blocking=non_blocking),
                    self.box_key: batch_element[self.box_key].to(device, non_blocking=non_blocking)
                }
                for batch_element in batch
            ]

        if self.validation:
            if self.whole_image_validation:
                return {"input_images":img, "use_inferer": True}, targets
            return {"input_images":img}, targets
        else:
            return {"input_images":img,"targets": targets}, targets

class PrepareSegBatch:

    def __init__(self, image_key, label_key, box_key, seg_key,validation=False,whole_image_validation= False):
        self.image_key = image_key
        self.label_key = label_key
        self.box_key = box_key
        self.seg_key = seg_key
        self.validation = validation
        self.whole_image_validation = whole_image_validation

    def __call__(self, batch, device, non_blocking):

        if not self.whole_image_validation:
            img = [batch_element[self.image_key].to(device, non_blocking=non_blocking) for batch_i in batch for batch_element in batch_i]

            targets = [
                {
                    self.label_key: batch_element[self.label_key].to(device, non_blocking=non_blocking),
                    self.box_key: batch_element[self.box_key].to(device, non_blocking=non_blocking),
                    self.seg_key: batch_element[self.seg_key].to(device, non_blocking=non_blocking)
                }
                for batch_i in batch for batch_element in batch_i]
        else:
            img = [batch_element[self.image_key].to(device, non_blocking=non_blocking) for
                   batch_element in batch]

            targets = [
                {
                    self.label_key: batch_element[self.label_key].to(device, non_blocking=non_blocking),
                    self.box_key: batch_element[self.box_key].to(device, non_blocking=non_blocking),
                    self.seg_key: batch_element[self.seg_key].to(device, non_blocking=non_blocking)
                }
                for batch_element in batch
            ]

        if self.validation:
            if self.whole_image_validation:
                return {"input_images":img, "use_inferer": True}, targets
            return {"input_images":img}, targets
        else:
            return {"input_images":img,"targets": targets}, targets
        

def configure_seg_detector(detector,hard_negative_sampler_kwargs,box_selector_kwargs,val_roi_size,atss_matcher_kwargs):
    print("Configuring Seg Detector")
    detector.set_atss_matcher(
        **atss_matcher_kwargs)
    
    detector.set_hard_negative_sampler(
        **hard_negative_sampler_kwargs
    )

    detector.set_target_keys(box_key="box", label_key="label", seg_logits_key="instance_seg")

    detector.set_box_selector_parameters(
        **box_selector_kwargs
    )

    detector.set_sliding_window_inferer(
        roi_size=val_roi_size,
        overlap=0.25,
        sw_batch_size=1,
        mode="constant",
        device="cuda:0",
        progress=True,
    )


def load_pretrained_swinunetr_seg(detector, pretrained_swinunetr_ckpt_dir):
    """
    Load the pretrained SwinUNETR weights into the detector.

    Parameters
    ----------
    detector : object
        The detector object to load the weights into.
    pretrained_swinunetr_ckpt : str
        The path to the pretrained SwinUNETR checkpoint file.

    Returns
    -------
    None
    """
    print("Loading Pretrained SwinUNETR Segmentation Weights")
    epoch_to_load = get_checkpoint("latest", pretrained_swinunetr_ckpt_dir)
    pretrained_swinunetr_ckpt = str(Path(pretrained_swinunetr_ckpt_dir).joinpath(f"checkpoint_epoch={epoch_to_load}.pt"))
    state_dict = torch.load(pretrained_swinunetr_ckpt, weights_only=False)["detector"]
    detector.load_state_dict(state_dict, strict=False)