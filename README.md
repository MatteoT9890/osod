# Open Set Object Detection

## Setup

```
git clone <link detectron2> ## Download detectron2 code
rm detectron2/detectron2/config/defaults.py ## Remove default configuration file of detectron2
mv defaults.py detectron2/detectron2/config ## Copy our configuration file inside detectron library
rm detectron2/detectron2/modeling/backbone/resnet.py ## Remove default resnet of detectron2
mv resnet.py detectron2/detectron2/modeling/backbone ## Copy our resnet inside detectron2
unzip data.zip
rm -r detectron2/detectron2/data #Delete detectron2 data folder
mv data detectron2/detectron2 #Move ours data folder inside detectron2
```

## Run

```
cd start
bash start_all_bg.sh ## Run all methods in background: if you want to run specific method, comment some lines inside start_all.sh file 
```

