# AVT2-DWF: Improving Deepfake Detection with Audio-Visual Fusion and Dynamic Weighting Strategies

This repository contains the implementation of AVT2-DWF method proposed in the paper 

Links: [[PDF]]() 

![Network Architecture](https://github.com/raining-dev/AVT2-DWF/blob/main/model_architecture.pdf)
  
# Dependencies
1) torch  

  
# Prepare data
1) Data directory
   ```
   /data/real/{videoname}.mp4  
   /daata/fake/{videoname}.mp4  
   ```
2) Once the videos have been placed at the above mentioned paths, run `python pre-process.py --out_dir train` and `python pre-process.py --out_dir test` for pre-processing the videos.  
  
3) After the above step, you can delete `pyavi`, `pywork`, `pyframes` and `pycrop` directories under `train` and `test` folders. (Do not delete `pytmp` folder please!)  
  
4) Collect video paths in csv files by running `python write_csv.py --out_dir . ` command.  

Thanks to the code available at https://github.com/abhinavdhall/deepfake/tree/main/ACM_MM_2020

# Training
```
python all_process.py --out_dir "train dataset path" --gpu 1 --resume false
```

# Testing
```
python all_process.py --test /checkpoints --out_dir "test dataset path"
```

