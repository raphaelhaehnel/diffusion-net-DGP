# How to run

Run every command from inside its experiment folder (the scripts find `src/` through a relative path). Activate the venv first:

```
.venv\Scripts\activate
```

`--input_features` takes `xyz` or `hks`. Operators are cached in each experiment's `data/op_cache/`, so the first run is slower than later ones. `--visualize` (in classification, human segmentation and RNA segmentation) opens polyscope with the test-set predictions after the final evaluation.

---

## 1. SHREC11 classification

```
cd experiments/classification_shrec11
```

Train and evaluate (each run draws a new random train/test split):

```
python classification_shrec11.py --dataset_type=original   --input_features=hks --visualize
python classification_shrec11.py --dataset_type=original   --input_features=xyz --visualize
python classification_shrec11.py --dataset_type=simplified --input_features=hks --visualize
python classification_shrec11.py --dataset_type=simplified --input_features=xyz --visualize
```

Optional: `--split_size=N` sets the number of training shapes per class (default 10).

There is no pretrained model or `--evaluate` mode for this task, because the split changes on every run. Results vary between runs, so average over several.

---

## 2. Human segmentation

```
cd experiments/human_segmentation_original
```

Evaluate the pretrained models:

```
python human_segmentation_original.py --evaluate --input_features=hks --visualize
python human_segmentation_original.py --evaluate --input_features=xyz --visualize
```

Train (saves to `data/saved_models/human_seg_{features}_4x128.pth`):

```
python human_segmentation_original.py --input_features=hks --visualize
python human_segmentation_original.py --input_features=xyz --visualize
```

---

## 3. Functional correspondence (FAUST / SCAPE)

```
cd experiments/functional_correspondence
```

Evaluate the pretrained models. `--load_model` is required with `--evaluate`. You can also test across datasets, e.g. a FAUST-trained model on SCAPE:

```
python functional_correspondence.py --evaluate --test_dataset=faust --input_features=xyz --load_model=pretrained_models/faust_xyz.pth
python functional_correspondence.py --evaluate --test_dataset=faust --input_features=hks --load_model=pretrained_models/faust_hks.pth
python functional_correspondence.py --evaluate --test_dataset=scape --input_features=xyz --load_model=pretrained_models/scape_xyz.pth
python functional_correspondence.py --evaluate --test_dataset=scape --input_features=hks --load_model=pretrained_models/scape_hks.pth

# cross-dataset
python functional_correspondence.py --evaluate --test_dataset=scape --input_features=xyz --load_model=pretrained_models/faust_xyz.pth
python functional_correspondence.py --evaluate --test_dataset=faust --input_features=xyz --load_model=pretrained_models/scape_xyz.pth
```

Train (saves to `saved_models/{train_dataset}_{features}.pth`):

```
python functional_correspondence.py --train_dataset=faust --test_dataset=faust --input_features=xyz
python functional_correspondence.py --train_dataset=faust --test_dataset=faust --input_features=hks
python functional_correspondence.py --train_dataset=scape --test_dataset=scape --input_features=xyz
python functional_correspondence.py --train_dataset=scape --test_dataset=scape --input_features=hks
```

The first run computes all-pairs geodesic distances for the error metric. It takes minutes per dataset and is cached afterwards.

There is no `--visualize` flag here.

---

## 4. RNA mesh segmentation

```
cd experiments/rna_mesh_segmentation
```

Train (default features: xyz; saves to `data/saved_models/rna_mesh_seg_{features}_4x128.pth`):

```
python rna_mesh_segmentation.py --input_features=xyz
python rna_mesh_segmentation.py --input_features=hks
```

No pretrained model ships with the repo. To use `--evaluate`, first train, then copy the saved model into `pretrained_models/`:

```
mkdir pretrained_models
copy data\saved_models\rna_mesh_seg_xyz_4x128.pth pretrained_models\
python rna_mesh_segmentation.py --evaluate --input_features=xyz
```

Add `--visualize` to any of these commands to open polyscope after the final evaluation. It shows all 136 test meshes in a grid with the per-vertex prediction, ground truth and errors (class 0 is the dataset's unlabeled class):

```
python rna_mesh_segmentation.py --evaluate --input_features=xyz --visualize
```

---

## 5. Sampling invariance (data not downloaded)

```
cd experiments/sampling_invariance
```

Evaluate the pretrained model (xyz only):

```
python sampling_invariance.py --evaluate --input_features=xyz
```

Train:

```
python sampling_invariance.py --input_features=xyz
```

**Note:** the data for this task is not downloaded. The script loads the training set (`data/train/MPI-FAUST/training/registrations/`) even with `--evaluate`, so **both** the train and test data (`data/test/`) are needed before even the pretrained evaluation will run. See `experiments/sampling_invariance/README.md` for the download steps. The real metric is the geodesic error printed at the end; the per-epoch "accuracy" is only training feedback.
