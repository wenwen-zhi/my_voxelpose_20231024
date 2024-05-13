# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from lib.dataset.mydataset import MyDataset as mydataset
from lib.dataset.mydatasetv2 import MyDatasetv2 as mydatasetv2

from lib.dataset.panoptic import Panoptic as panoptic
from lib.dataset.shelf_synthetic import ShelfSynthetic as shelf_synthetic
from lib.dataset.campus_synthetic import CampusSynthetic as campus_synthetic
from lib.dataset.shelf import Shelf as shelf
from lib.dataset.test_shelf import Shelf as test_shelf
from lib.dataset.campus import Campus as campus
from lib.dataset.association4d import Association4D as association4d
from lib.dataset.association4d_v2 import Association4DV2 as association4d_v2
from lib.dataset.shelf_end_to_end import ShelfEndToEnd as shelf_end_to_end
from lib.dataset.campus_end_to_end import CampusEndToEnd as campus_end_to_end
from lib.dataset.ue_dataset import UEDataset as ue_dataset
from lib.dataset.ue_dataset_synthetic import UEDatasetSynthetic as ue_dataset_synthetic
from lib.dataset.ue_dataset_synthetic_v2 import UEDatasetSyntheticV2 as ue_dataset_synthetic_v2
from lib.dataset.ue_dataset_synthetic_v2_test import UEDatasetSyntheticV2Test as ue_dataset_synthetic_v2_test




dataset_factory = {
    'mydataset': mydataset,
    'mydatasetv2': mydatasetv2,
    'panoptic': panoptic,
    'shelf_synthetic': shelf_synthetic,
    'campus_synthetic': campus_synthetic,
    'shelf': shelf,
    'test_shelf': test_shelf,
    'campus': campus,
    'association4d': association4d,
    'association4d_v2': association4d_v2,
    'shelf_end_to_end': shelf_end_to_end,
    "campus_end_to_end": campus_end_to_end,
    "ue_dataset": ue_dataset,
    'ue_dataset_synthetic':ue_dataset_synthetic,
    'ue_dataset_synthetic_v2':ue_dataset_synthetic_v2,
    'ue_dataset_synthetic_v2_test':ue_dataset_synthetic_v2_test,
}


def get_dataset(name):
    return dataset_factory[name]
