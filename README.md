# Visuomotor Navigation for Embodied Robots With Spatial Memory and Semantic Reasoning Cognition
[IRMV Lab](https://irmv.sjtu.edu.cn/)

Official Github repository for "Visuomotor Navigation for Embodied Robots With Spatial Memory and Semantic Reasoning Cognition".



## YOLOv3

#### 1.Model Training
```
cd objDetect/yolov3

python train.py --img 640 --batch 16 --epochs 2000 --data gibson.yaml --weights yolov3.pt
```
results：runs/train

#### 2.Model Detection Accuracy Validating
```
python val.py --img 640 --data gibson.yaml --weights yolov3.pt
```
results：runs/val

## Imitation Learning
#### 1.Training Data Obtaining
```
cd Gibson_Dataset_Sample

python sample_training_data.py
```

#### 2.Model Training
```
cd ImitationLearning_gibson/train/IL_topo_semantic

python main.py
```

#### 3.Testing
```
cd ImitationLearning_gibson/test/rl_topo_semantic

python main.py
```

