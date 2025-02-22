# BPD-DSRL

Code and supplementary material for our AAAI 2025 paper: **Enhancing Question Generation through Diversity-Seeking Reinforcement Learning with Bilevel Policy Decomposition**

## Checkpoints Available Here!

We provide supervised fine-tuning checkpoints (including cold-start policies and reward models) as well as reinforcement learning checkpoints to facilitate reimplementation and further experimentation.

**BPD-DSRL checkpoints are** [here](https://qubstudentcloud-my.sharepoint.com/my?id=%2Fpersonal%2F40414335_ads_qub_ac_uk%2FDocuments%2FBPD-DSRL-RL-Checkpoints%2FBPD-DSRL-CKPT-SQUAD2&ga=1)

### Evaluating BPD-DSRL

1. Download the checkpoints from the provided [link]([https://qubstudentcloud-my.sharepoint.com/my?id=%2Fpersonal%2F40414335%5Fads%5Fqub%5Fac%5Fuk%2FDocuments%2FBPD%2DDSRL%2DRL%2DCheckpoints%2FBPD%2DDSRL%2DCKPT%2DSQUAD2&ga=1)

2. To generate outputs using our model checkpoints, run the following command:

   ```bash
   python Eval/generation.py \
       --model_path PATH_TO_YOUR_MODEL \
       --n_test_dataset SELECTED_BENCHMARK_NAME \
       --output_file_name OUTPUT_FILE_NAME
   ```

3. To evaluate the generated outputs using the Bleu-based metrics in our main experiments, run the following command:

   ```bash
   python Eval/main_experiments.py \
       --output_file_path OUTPUT_FILE_PATH
   ```

**We release all test results generated by our reinforcement learning checkpoints using the NVIDIA L20 GPU in Eval/Test_Results/BPD_DSRL. Note that results may vary across different GPUs.**

## Training From Sratch

### Step 1: Supervised Fine-tuning (SFT)

We first use SFT to initalize the polciy, and learn the outcome reward model (question fluency and answerability).

1. Download the training and validation data from this [link](https://qubstudentcloud-my.sharepoint.com/:f:/g/personal/40414335_ads_qub_ac_uk/EjwlIB1oyNdNqDzq37Xm0IIB-RVwC4-6esWEP1KNlv4Z3g?e=RwuaF1) and place them in the appropriate folder. For example, to reimplement on the SQuAD 1.1/1 benchmark, download the squad1 dataset and move train.json and dev.json to Datasets/squad1.

2. Execute the following command with our default parameter settings.

**Policy Warm up (change the <u>*task*</u> varaible for different benchmarks)** 

```bash
source SFT/run_policy.sh
```

**Outcome Reward Model Learning (change the <u>*task*</u> varaible for different benchmarks)** 

```bash
source SFT/run_reward.sh
```

### Step 2: Reinforcement Learning (RL)

Using the model checkpoints from SFT, we further use RL to fine-tune the policy.

We provide a demo using the SQuAD 1.1/1 benchmark.

1. Download our pre-trained SFT policy and outcome reward model checkpoints from this [link](https://qubstudentcloud-my.sharepoint.com/:f:/g/personal/40414335_ads_qub_ac_uk/EkzKu244C81Iiv972374J_IB8bL5nZOSv6ycUumqEJuoGw?e=UWqc2D) and place them in the project root directory (./).

2. Execute the following command with our default parameter settings.

   ```
   python RL/main.py
   ```

3. Run the following command to generate questions using the fine-tuned RL model:

   ```bash
   python Eval/generation.py \
       --model_path RL/DSRL-checkpoint-squad1-step-100 \
       --n_test_dataset squad1 \
       --output_file_name test_squad1
   ```

4. Use BLEU-based metrics to evaluate the generated questions:

   ```bash
   python Eval/main_experiments.py \
       --output_file_path Eval/test_squad1.jsonl
   ```

   





