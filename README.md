This repo contains our code and configurations for the Kaggle - LLM Science Exam competition. A detailed summary of the solution is posted [here](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/447647). Please refer to the following sections for details on training and dependencies. 

## Section 1: Setup
### 1.1 Hardware
Computing resources from Jarvislabs.ai were used. Specifically, models were trained on the following instance:

Ubuntu 20.04.5 LTS (128 GB boot disk)
Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz (7 vCPUs)
1 x NVIDIA A100 40GB GPU OR 1 x NVIDIA A6000 48GB GPU

### 1.2 Software
I used PyTorch-2.0 image from Jarvislabs.ai, which comes with:

* Python 3.10.11
* CUDA 11.8
* Python packages installation: pip install -r requirements.txt

### 1.3 Datasets
Please make sure Kaggle API is installed. Then run the following script to download the required datasets:

```
chmod +x ./setup.sh
./setup.sh
```

Please note that the above script will create a datasets folder in the directory located one level above the current directory. The external datasets will be downloaded in the datasets folder.

## Section 2: Training

### 2.1 Retriever Training
```
python ./code/train_e_topic.py \
--config-name conf_e_topic_bge \
use_wandb=false \
all_data=false
```

### 2.2 Ranker Training
```
python ./code/train_e_ranker.py \
--config-name conf_e_ranker \
use_wandb=false \
all_data=false
```

### 2.3 Reader: Spanwise model

#### Step 1: training with large number of MCQs
```
python ./code/train_r_delta.py \
--config-name conf_r_delta_k1 \
use_wandb=false \
all_data=false
```

#### Step 2: specialization with difficult MCQs
```
python ./code/train_r_delta.py \
--config-name conf_r_delta_k2_resumed \
use_wandb=false \
all_data=false
```