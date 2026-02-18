# Anatomy-Aware Lymphoma Lesion Detection in Whole-Body PET/CT

This repository contains the official source code for the paper **"Anatomy-Aware Lymphoma Lesion Detection in Whole-Body PET/CT"**.  
It provides a comprehensive overview of the project and detailed instructions to reproduce all experiments described in the paper, including data preparation, model training, and evaluation steps.

## Dataset Description

This project utilizes two publicly available, expertly annotated whole-body PET/CT datasets from The Cancer Imaging Archive (TCIA):

1. **FDG-PET-CT-Lesions**  
   This dataset, published by Gatidis and Kuestner (2022), includes whole-body [<sup>18</sup>F]FDG PET/CT scans from 900 patients, primarily with lymphoma. Each scan features comprehensive, high-quality manual 3D lesion segmentations by clinical experts. The dataset also provides corresponding metadata, making it ideal for developing and evaluating algorithms for lesion detection and segmentation.

   - **Details:**  
     - Modality: [<sup>18</sup>F]FDG PET/CT  
     - Subjects: 900  
     - Annotations: Manual segmentation of all lymphoma lesions  
     - DOI: [10.7937/gkr0-xv29](https://doi.org/10.57754/FDAT.wf9fy-txq84)

2. **PSMA-PET-CT-Lesions**  
   Described by Jeblick et al. (2024), this resource contains whole-body [<sup>68</sup>Ga]PSMA-11 PET/CT scans from 365 prostate cancer patients, also with expert manual 3D lesion segmentations. Similar to the FDG dataset, this resource supports training and testing of deep learning models for lesion detection and segmentation across a different cancer type and tracer.

   - **Details:**  
     - Modality: [<sup>68</sup>Ga]PSMA-11 PET/CT  
     - Subjects: 365  
     - Annotations: Manual segmentation of all tumor lesions  
     - DOI: [10.7937/r7ep-3x37](https://doi.org/10.57754/FDAT.6gjsg-zcg93)

### Data Access and Usage

Both datasets are freely available to the research community via TCIA, and their use in this project complies with the corresponding terms and conditions. Please cite the original dataset publications when using the data in your own research.

## Environment Setup

### Setting Up the Environment with Docker

To simplify environment setup and ensure reproducibility, you can use the provided Docker image. This project is compatible with the `nndetection` environment:

```bash
docker pull ghcr.io/minnelab/nndetection:1.0
```
or, to use the same image with Singularity, you can use the following command:
```bash
singularity pull nndetection_1.0.sif docker://ghcr.io/minnelab/nndetection:1.0
```

### Local Environment Setup

To set up the environment locally, you can create a virtual environment and install the dependencies using the following command:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Experimental Workflow

The experimental workflow consists of the following main stages:

1. Data Preparation
   - Download the required datasets from TCIA.
   - Convert the semantic lesion segmentations to instance segmentations.
   - Create the datalist for the AutoPET Decathlon dataset.
   - [Optional] Generate anatomical segmentations using TotalSegmentator. These anatomical masks can be incorporated as an additional input channel to the detection network if anatomy awareness is desired.
   - Experimental planning.
   - Data preprocessing.

2. Model Training and Evaluation
   - [Optional] Train a self-supervised learning (SSL) model, such as the Swin Retina UneTR, to initialize weights for downstream tasks.
   - Train and evaluate lymphoma lesion detection models. Two main architectures are considered:
     - **Retina U-Net**: Used as the baseline model, trained without anatomical (TotalSegmentator) input.
     - **Swin Retina UneTR**: Can be trained either with or without the additional anatomical input channel (from TotalSegmentator masks).
   - For anatomy-aware experiments, both the Retina U-Net and Swin Retina UneTR models can be augmented with the anatomical channel; performance comparisons should be made to assess the impact of providing anatomical context.
   - Evaluate all models according to standardized metrics.


## Download the Dataset

To download the PSMA-FDG-PET-CT-Lesions dataset, you can use the following command:
```bash
wget https://fdat.uni-tuebingen.de/records/6gjsg-zcg93/files/psma-fdg-pet-ct-lesions.zip -O psma-fdg-pet-ct-lesions.zip
unzip psma-fdg-pet-ct-lesions.zip
```
The dataset has the following Decathlon structure:
```
psma-fdg-pet-ct-lesions/
├── dataset.json
├── imagesTr/
│   ├── <patient_id>_0000.nii.gz  # CT image
│   ├── <patient_id>_0001.nii.gz  # PET image
│   └── ...
├── labelsTr/
   ├── <patient_id>.nii.gz  # Semantic segmentation
   └── ...

```

## Convert the Semantic Segmentations to Instance Segmentations
For object detection, each individual lesion must be treated as a distinct object. To achieve this, the semantic segmentation masks, which label all lesions with the same value, need to be converted into instance segmentation masks, where each separate lesion is assigned a unique integer label.
To convert the semantic segmentations to instance segmentations, you can use the following command:
```bash
PyMAIA_convert_semantic_to_instance_segmentation --data-folder $DATA_DIR/labelsTr --sem-seg-suffix .nii.gz --inst-seg-suffix .nii.gz --inst-seg-folder $DATA_DIR/labelsInstTr --output-json-path $DATA_DIR/Autopet_Inst-Seg.json  --decathlon-format yes
```
A separate folder for the instance segmentations will be created:
```
psma-fdg-pet-ct-lesions/
├── labelsInstTr/
│   ├── <patient_id>.nii.gz  # Instance segmentation
│   └── ...
```
Additionaly, a JSON file (`Autopet_Inst-Seg.json`) will be created, including the number of lesions for each patient.
```
{
    "<patient_id>" : <number_of_lesions>

}
```
## Create the Decathlon Datalist
Before proceeding with the data preprocessing, we need to create the Decathlon datalist. This is a dictionary that maps the patient IDs to the corresponding image and segmentation files, following this structure:
```json
{
    "testing": [], 
    "training": [
        {
            "ct": "imagesTr/<patient_id>_0000.nii.gz",
            "pet": "imagesTr/<patient_id>_0001.nii.gz", 
            "label": "labelsTr/<patient_id>.nii.gz",
            "instance_seg": "labelsInstTr/<patient_id>.nii.gz",
            "fold": 0
        },
        ...
    ],
}
```
The single cases are assigned to the corresponding folds, according to the `splits_final.json` file, already provided in the dataset. 
To create the datalist, you can use the following command:
```bash
python scripts/create_Autopet_Decathlon_Datalist.py --root-dir $DATA_DIR
```
The datalist is saved as `Autopet_Decathlon.json` in the root directory of the dataset.

## Run Planning
The experimen planning step is required to create a dataset fingerprint (including values such as voxel spacing, image size, modalities, etc.) that is used to preprocess the data. To run the planning, you can use the following command:
```bash
export BUNDLE=Bundles/Detection
python -m monai.bundle run plan \
--bundle_root $BUNDLE \
--decathlon_data_list $DATA_DIR/Autopet_Decathlon.json \
--data_dir $DATA_DIR \
--config_file "['$BUNDLE/Detection/dataset.yaml','$BUNDLE/Detection/transforms.yaml', '$BUNDLE/Detection/plan.yaml','$BUNDLE/Detection/params.yaml','$BUNDLE/Detection/imports.yaml']" \
--meta_file $BUNDLE/Detection/metadata.json
```
The command will analyze the dataset and create a `plan.json` file in the dataset directory:
```json
{
  "<PATIENT_ID>": {
    "ct": 
    {
      "spacing": [x_spacing, y_spacing, z_spacing],
      "size": [x_size, y_size, z_size],
      "space": "RAS"
    },
    "pet":
    {
      "spacing": [x_spacing, y_spacing, z_spacing],
      "size": [x_size, y_size, z_size],
      "space": "RAS"
    }
  }
}
```

## Run Preprocessing
The preprocessing step is performing the following transformations:
- Loading CT, PET and instance segmentations images
- Orienting the images to the given orientation axes
- Resampling the images to the given voxel spacing
- Scaling the single channel images to the range [0, 1]
- Concatenating the PET and CT images into a single image

To run the preprocessing, you can use the following command:
```bash
python -m monai.bundle run preprocess \
--bundle_root $BUNDLE \
--decathlon_data_list $DATA_DIR/Autopet_Decathlon.json \
--data_dir $DATA_DIR \
--config_file "['$BUNDLE/Detection/dataset.yaml','$BUNDLE/Detection/transforms.yaml', '$BUNDLE/Detection/preprocess.yaml','$BUNDLE/Detection/params.yaml','$BUNDLE/Detection/imports.yaml']" \
--meta_file $BUNDLE/Detection/metadata.json
```

You will get a folder with the preprocessed images in the `<data_dir>/preprocessed/` directory:
```
psma-fdg-pet-ct-lesions/
├── preprocessed/
│   ├── <patient_id> # Folder with the preprocessed images
│       ├── <patient_id>_0000.nii.gz # Preprocessed CT image
│       ├── <patient_id>_0001.nii.gz # Preprocessed PET image
│       ├── <patient_id>.json # Object information
│       ├── <patient_id>_image.nii.gz # Preprocessed concatenated image
│       └── <patient_id>.nii.gz # Preprocessed instance segmentation
```
Where the `<patient_id>.json` file contains the object information, including the bounding box, class label, and instance ID:
```json
{
    "ct": <path_to_ct_image>,
    "pet": <path_to_pet_image>,
    "instance_seg": <path_to_instance_seg_image>,
    "box": [
        [BOX_0_X_MIN, BOX_0_Y_MIN, BOX_0_Z_MIN, BOX_0_X_MAX, BOX_0_Y_MAX, BOX_0_Z_MAX],
        [BOX_1_X_MIN, BOX_1_Y_MIN, BOX_1_Z_MIN, BOX_1_X_MAX, BOX_1_Y_MAX, BOX_1_Z_MAX],
        ...
    ],
    "label": [
        <CLASS_0>,
        <CLASS_1>,
        ...
    ]
}
```
Where the `box` is a list of bounding boxes (in image coordinates) and the `label` is a list of class labels for each object.

To visualize the preprocessed images with the corresponding instance segmentations and the bounding boxes, you can follow the instructions in the [Detection Notebook](Detection.ipynb).

## TotalSegmentator
## Train a self-supervised learning (SSL) model
## Train nnDetection model
## References

1. Gatidis S, Kuestner T. A whole-body FDG-PET/CT dataset with manually annotated tumor lesions (FDG-PET-CT-Lesions) [Dataset]. The Cancer Imaging Archive, 2022. DOI: [10.7937/gkr0-xv29](https://doi.org/10.57754/FDAT.wf9fy-txq84)
2. Jeblick, K., Boning, G., Battistel, L., et al. A whole-body PSMA-PET/CT dataset with manually annotated tumor lesions (PSMA-PET-CT-Lesions) (Version 1) [Dataset]. The Cancer Imaging Archive, 2024. DOI: [10.7937/r7ep-3x37](https://doi.org/10.57754/FDAT.6gjsg-zcg93)


