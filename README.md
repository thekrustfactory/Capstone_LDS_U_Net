# Capstone_LDS_U_Net
Comparative Evaluation of U-Net and LDS U-Net for Ultrasound Bony Feature Segmentation in Scoliosis Patients

call in terminal
LDS gate only:
python ldsunet_model.py --images data/images --masks data/masks --no-mskip --no-dense --use-dice

LDS skip only:
python ldsunet_model.py --images data/images --masks data/masks --no-gate --no-dense --use-dice

LDS dense only:
python ldsunet_model.py --images data/images --masks data/masks --no-mskip --no-gate --use-dice

LDS dense + skip:
python ldsunet_model.py --images data/images --masks data/masks --no-gate --use-dice

LDS gate + dense:
python ldsunet_model.py --images data/images --masks data/masks --no-mskip --use-dice

LDS gate + skip:
python ldsunet_model.py --images data/images --masks data/masks --no-dense --use-dice

Full LDS U-Net:
python ldsunet_model.py --images data/images --masks data/masks  --use-dice

basic U-Net:
python ldsunet_model.py --images data/images --masks data/masks --no-mskip --no-dense --no-gate --use-dice
