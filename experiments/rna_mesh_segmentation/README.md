This da 


### Data

The dataset is available and documented at https://github.com/nmwsharp/RNA-Surface-Segmentation-Dataset. Clone this repo in to the `data` subdirectory, so the meshes are stored in `data/RNA-Surface-Segmentation-Dataset/off/`.

```
cd data
git clone https://github.com/nmwsharp/RNA-Surface-Segmentation-Dataset.git
```

### Training from scratch

To train the models, use

```python
python rna_mesh_segmentation.py --input_features=xyz  
```
or, with heat kernel signature features
```python
python rna_mesh_segmentation.py --input_features=hks  
```

### Visualization

Add `--visualize` to open a polyscope viewer after the final evaluation. It shows every test mesh with the predicted per-vertex segmentation, the ground truth, and the per-vertex errors. The accuracy of each mesh is also printed to the console.
