# Capstone_LDS_U_Net
Comparative Evaluation of U-Net and LDS U-Net for Ultrasound Bony Feature Segmentation in Scoliosis Patients

## How to Run

1. Import 3D ultrasound images data into ./data/images directory

2. Import 3D ultrasound masks data into ./data/masks directory

3. Otherwise specify path with --images and --masks

4. Run one of the following options in terminal:

**LDS gate only:**<br>
```
python ldsunet.py --images data/images --masks data/masks --no-mskip --no-dense --use-dice
```
**LDS skip only:**<br>
```
python ldsunet.py --images data/images --masks data/masks --no-gate --no-dense --use-dice
```
**LDS dense only:**<br>
```
python ldsunet.py --images data/images --masks data/masks --no-mskip --no-gate --use-dice
```
**LDS dense + skip:**<br>
```
python ldsunet.py --images data/images --masks data/masks --no-gate --use-dice
```
**LDS gate + dense:**<br>
```
python ldsunet.py --images data/images --masks data/masks --no-mskip --use-dice
```
**LDS gate + skip:**<br>
```
python ldsunet.py --images data/images --masks data/masks --no-dense --use-dice
```
**Full LDS U-Net:**<br>
```
python ldsunet.py --images data/images --masks data/masks  --use-dice
```
**Basic U-Net:**<br>
```
python ldsunet.py --images data/images --masks data/masks --no-mskip --no-dense --no-gate --use-dice
```


## CLI commands
| Argument | Type / Action | Default | Description |
|-----------|----------------|----------|--------------|
| `--images` | str | `data/images` | Folder with input images |
| `--masks` | str | `data/masks` | Folder with corresponding masks |
| `--width` | int | `64` | Resize width |
| `--height` | int | `256` | Resize height |
| `--batch-size` | int | `4` | Batch size for training |
| `--epochs` | int | `10` | Number of training epochs |
| `--lr` | float | `1e-3` | Learning rate |
| `--threshold` | float | `0.5` | Segmentation threshold |
| `--use-dice` | action: store_true | — | Use 0.5 × BCEWithLogits + 0.5 × Dice loss |
| `--out` | str | `runs/lds_unet_best.pt` | Path to save best checkpoint |
| `--save-vis` | str | `runs/vis_lds` | Folder to save visualisation panels |
| `--max-vis` | int | `16` | Number of panels to save |
| `--weights` | str | `""` | Load weights before training/evaluation |
| `--eval-only` | action: store_true | — | Evaluate only, no training |
| `--show-vis` | action: store_true | — | Show figures as they are generated |
| `--open-folder` | action: store_true | — | Automatically open output folder |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--no-dense` | action: store_true | — | Disable DenseBlocks |
| `--no-gate` | action: store_true | — | Disable selection gates on skip connections |
| `--no-mskip`, `--no-multiscale-skip` | action: store_true | — | Disable multi-scale skip concatenations |

