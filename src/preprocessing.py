
from tqdm import tqdm
from pathlib import Path
import json
from monai.transforms import Compose, AsDiscrete, Lambda
from monai.apps.detection.transforms.array import MaskToBox
import numpy as np
import monai
import shutil
from pathlib import Path

MAX_INSTANCES = 30

def create_autopet_decathlon_datalist(root_dir):
    """
    Create a data list for the AutoPET Decathlon dataset.
    This function reads the final splits from a JSON file and generates a data list
    containing paths to CT, PET, label, and instance segmentation files for training.
    
    Parameters
    ----------
    root_dir : str
        The root directory where the dataset and splits_final.json file are located.
    
    Returns
    -------
    None
        The function saves the generated data list as a JSON file named 
        'Autopet_v1.1_Decathlon.json' in the root directory.
    """
    
    data_list = {"testing": [], "training": []}
    
    with open(Path(root_dir).joinpath("splits_final.json"), "r") as f:
        final_splits = json.load(f)
    

    for idx, _ in enumerate(final_splits):
        for case in final_splits[idx]["val"]:
            data_list["training"].append({
                "ct": str(Path("imagesTr").joinpath(case+"_0000.nii.gz")),
                "pet": str(Path("imagesTr").joinpath(case+"_0001.nii.gz")),
                "label": str(Path("labelsTr").joinpath(case+".nii.gz")),
                "instance_seg": str(Path("labelsInstTr").joinpath(case+".nii.gz")),
                "fold": idx
            })
    
    with open(Path(root_dir).joinpath("Autopet_Decathlon.json"),"w") as f:
        json.dump(data_list,f)        
    

def run_planning(data_loaders, modality_conf, image_modalities, plan_filename):
    """
    Generates a planning file containing metadata for each case in the dataset.
    
    Parameters
    ----------
    data_loaders : list
        List of data loaders, each providing batches of data.
    modality_conf : dict
        Dictionary containing configuration for each modality. Each key is a modality name, and the value is a dictionary with modality-specific settings, including 'suffix'.
    image_modalities : list
        List of modality names to be included in the planning.
    plan_filename : str
        Path to the output JSON file where the planning data will be saved.
    
    Returns
    -------
    None
    """
    dataset_dict = {}

    for data_loader in data_loaders:
        for data_batch in tqdm(data_loader):

            for data in data_batch:
                idx = 0
                for modality_key in modality_conf:
                    if modality_key in image_modalities:
                        modality_suffix = modality_conf[modality_key]['suffix']
                        if idx == 0:
                            dataset_dict[Path(data[modality_key].meta['filename_or_obj']).name[:-len(modality_suffix)]] = {}
                            idx +=1
                        case_id = data[modality_key].meta['filename_or_obj']
                        
                        dataset_dict[Path(data[modality_key].meta['filename_or_obj']).name[:-len(modality_suffix)]][modality_key] = {}


                        dataset_dict[Path(data[modality_key].meta['filename_or_obj']).name[:-len(modality_suffix)]][modality_key]['spacing'] = data[modality_key].meta['pixdim'][1:4].tolist()
                        dataset_dict[Path(data[modality_key].meta['filename_or_obj']).name[:-len(modality_suffix)]][modality_key]['size'] = data[modality_key].meta['dim'][1:4].tolist()
                        dataset_dict[Path(data[modality_key].meta['filename_or_obj']).name[:-len(modality_suffix)]][modality_key]['space'] = data[modality_key].meta['space']

    with open(plan_filename,"w") as f:
        json.dump(dataset_dict, f)


class PreprocessNameFormatter:
    """
    A class to format the name of preprocessed files based on metadata and a SaveImage transform.

    Parameters
    ----------
    filename_key : str
        The key used to identify the filename in the metadata.

    Methods
    -------
    __call__(metadict: dict, saver: monai.transforms.Transform) -> dict
        Returns a dictionary with formatted filename and patch index based on the input metadata and SaveImage transform.
    """
    def __init__(self, filename_key):
        self.filename_key = filename_key


    def __call__(self, metadict: dict, saver: monai.transforms.Transform) -> dict:
        subject = (
            metadict.get(monai.utils.ImageMetaKey.FILENAME_OR_OBJ, getattr(saver, "_data_index", 0))
            if metadict
            else getattr(saver, "_data_index", 0)
        )
        patch_index = metadict.get(monai.utils.ImageMetaKey.PATCH_INDEX, None) if metadict else None
        subject = subject[:-len(self.filename_key)]+".nii.gz"
        return {"subject": f"{subject}", "idx": patch_index}
    
def preprocess(
    preprocess_datasets,
    preprocess_transform,
    modality_conf,
    box_key,
    label_key,
    instance_segmentation_key,
    preprocess_file_path,
    save_preprocess_transforms,
    preprocess_dir
    ):
    """
    Preprocess datasets for object detection.
    Parameters
    ----------
    preprocess_datasets : dict
        Dictionary containing datasets for different phases (e.g., "training", "validation").
    preprocess_transform : callable
        Function to apply preprocessing transformations to the data.
    modality_conf : dict
        Configuration dictionary for different modalities.
    box_key : str
        Key to access bounding box information in the data.
    label_key : str
        Key to access label information in the data.
    instance_segmentation_key : str
        Key to access instance segmentation information in the data.
    preprocess_file_path : str
        Path to save the final preprocessed data dictionary.
    save_preprocess_transforms : callable
        Function to save the preprocessed transformations.
    preprocess_dir : str
        Directory to save the preprocessed files.
    Returns
    -------
    None
    """  
    data_dict = {}
    
    for phase in preprocess_datasets: 
        if phase not in ["training","validation"]:
            continue
        data_dict[phase] = []
        for data in tqdm(preprocess_datasets[phase],desc=f"Processing {phase}"):
            instance_segmentation_suffix = modality_conf[instance_segmentation_key]['suffix']
            name = Path(data[instance_segmentation_key]).name[:-len(instance_segmentation_suffix)]
            if Path(preprocess_dir).joinpath(name, name+".json").exists():
                with open(Path(preprocess_dir).joinpath(name, name+".json"), "r") as f:
                    data_dict[phase].append(json.load(f))
            else:
                print(name)
                filename = data[instance_segmentation_key]
                data = preprocess_transform(data)
                n_instances = np.max(data[instance_segmentation_key])
                if n_instances > MAX_INSTANCES:
                    print(f"Too many instances {n_instances} in {filename}")
                    continue
                preprocess_box_transforms = Compose(
                    [
                        AsDiscrete(to_onehot=int(n_instances)+1),
                        Lambda(func= lambda x: x[1:,:]),
                        Lambda(func= lambda x: x-1),
                        MaskToBox(bg_label=-1)
                    ]
                )

                boxes, labels = preprocess_box_transforms(data[instance_segmentation_key])
                save_preprocess_transforms(data)


                instance_segmentation_suffix = modality_conf[instance_segmentation_key]['suffix']
                name = Path(data[instance_segmentation_key].meta['filename_or_obj']).name[:-len(instance_segmentation_suffix)]
                data_dict[phase].append(
                    {   modality: str(Path(name).joinpath(name+modality_conf[modality]['suffix']))
                        for modality in modality_conf


                    })
                for modality in modality_conf:
                    if not Path(preprocess_dir).joinpath(name, name+modality_conf[modality]['suffix']).exists():
                        #print(f"Missing {name+modality_conf[modality]['suffix']}")
                        shutil.copy(Path(data[modality].meta['filename_or_obj']), Path(preprocess_dir).joinpath(name, name+modality_conf[modality]['suffix']))
                data_dict[phase][-1][box_key] = boxes.tolist()
                data_dict[phase][-1][label_key] = labels.tolist()
                with open(Path(preprocess_dir).joinpath(name,name+".json"), "w") as f:
                    json.dump(data_dict[phase][-1], f)

        
    with open(preprocess_file_path,"w") as f:
        json.dump(data_dict,f)