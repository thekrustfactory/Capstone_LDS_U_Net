# Capstone_LDS_U_Net
Comparative Evaluation of U-Net and LDS U-Net for Ultrasound Bony Feature Segmentation in Scoliosis Patients

Run one of the following options in terminal:

**LDS gate only:**<br>
```
python ldsunet_model.py --images data/images --masks data/masks --no-mskip --no-dense --use-dice
```
**LDS skip only:**<br>
```
python ldsunet_model.py --images data/images --masks data/masks --no-gate --no-dense --use-dice
```
**LDS dense only:**<br>
```
python ldsunet_model.py --images data/images --masks data/masks --no-mskip --no-gate --use-dice
```
**LDS dense + skip:**<br>
```
python ldsunet_model.py --images data/images --masks data/masks --no-gate --use-dice
```
**LDS gate + dense:**<br>
```
python ldsunet_model.py --images data/images --masks data/masks --no-mskip --use-dice
```
**LDS gate + skip:**<br>
```
python ldsunet_model.py --images data/images --masks data/masks --no-dense --use-dice
```
**Full LDS U-Net:**<br>
```
python ldsunet_model.py --images data/images --masks data/masks  --use-dice
```
**Basic U-Net:**<br>
```
python ldsunet_model.py --images data/images --masks data/masks --no-mskip --no-dense --no-gate --use-dice
```


