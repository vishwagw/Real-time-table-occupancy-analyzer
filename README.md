# Table occupancy detection model

## model structure:

table-occupancy-app/
│
├── backend/
│   └── app.py
│
├── templates/
│   └── index.html
│
├── main.py
├── requirements.txt
└── README.md

## integrate GPU:
### For CUDA (NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

### For CPU only
pip install torch torchvision torchaudio


## training structure:
### dataset.yaml
path: /path/to/your/dataset
train: images/train
val: images/val
test: images/test

nc: 4  # number of classes
names: ['table', 'occupied_table', 'vacant_table', 'person', 'chair']

## 