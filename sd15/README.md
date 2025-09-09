# SD1.5 COCO Val2014 Evaluation

Evaluate Stable Diffusion 1.5 on COCO validation 2014 captions and calculate FID scores.


## Setup
```bash
conda create -n sd15-eval python=3.10
conda activate sd15-eval
```

```bash
pip install torch torchvision transformers diffusers accelerate

pip install pandas pillow tqdm

pip install pytorch-fid clean-fid

pip install scipy numpy matplotlib
```

### Run
```bash
chmod +x run_eval.sh

./run_eval.sh
```
