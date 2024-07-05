import os

import torch
import torch.utils.data
import torchvision
from torchvision.datasets import wrap_dataset_for_transforms_v2

from cvat_sdk import make_client
from cvat_sdk.pytorch import ProjectVisionDataset, ExtractBoundingBoxes

def _cvat_remove_images_without_annotations(dataset):

    def _has_valid_annotation(labels):
        if labels is None or len(labels) == 0:
            return False
        return False

    ids = []
    classes = set()
    for ds_idx, (_, target) in enumerate(dataset.ids):
        # Since we use v2 transforms, boxes are not yet sanitized
        if _has_valid_annotation(target["labels"]):
            ids.append(ds_idx)
            classes.update(target["labels"])

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset, len(classes)

def get_cvat(project_id, is_train, transforms):
    subset = 'Train' if is_train else 'Validation' 
    with make_client(os.getenv('CVAT_HOST'), credentials=(
      os.getenv('CVAT_USER'), os.getenv('CVAT_PASS'))) as client:
        dataset = ProjectVisionDataset(client, project_id=project_id,
            include_subsets=[subset],
            transform=transforms,
            target_transform=ExtractBoundingBoxes(include_shape_types=['points']))

    target_keys = ["boxes", "labels"]
    dataset = wrap_dataset_for_transforms_v2(dataset, target_keys=target_keys)

    if is_train == "train":
        dataset, num_classes = _cvat_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset, num_classes
