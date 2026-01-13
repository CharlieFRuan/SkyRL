<div align="center">

# SkyRL: A Modular Full-stack RL Library for LLMs

<p align="center">
| <a href="https://skyrl.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://x.com/NovaSkyAI"><b>Twitter/X</b></a> | <a href="https://huggingface.co/NovaSky-AI"><b>Huggingface</b></a> | <a href="https://join.slack.com/t/skyrl/shared_invite/zt-3f6ncn5b8-QawzK3uks6ka3KWoLwsi5Q"><b>Slack Workspace</b></a> |
</p>

</div>

# Overview of this fork

This is a fork of SkyRL for the [OpenThoughts-Agent project](https://github.com/open-thoughts/OpenThoughts-Agent).

We will soon merge the changes to the main SkyRL branch.

For the time being, we list the steps to run SkyRL+Harbor for reproducing the RL training of our first release, i.e.:
- Using [open-thoughts/OpenThinker-Agent-v1-SFT](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1-SFT) as base
- GRPO with the data [open-thoughts/OpenThoughts-Agent-v1-RL](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL), while
- Evaluating with [open-thoughts/OpenThoughts-TB-dev](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TB-dev), and 
- Getting the final [open-thoughts/OpenThinker-Agent-v1](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1)

### Environment

Install SkyRL

```bash
conda create -n otagent python=3.12
conda activate otagent
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

git clone https://github.com/mlfoundations/SkyRL
cd SkyRL/skyrl-train/
pip install -e .
pip install "vllm==0.10.1.1"
cd ../..
```

Install Harbor
```bash
git clone https://github.com/CharlieFRuan/harbor
cd harbor
git checkout 112425-terminus2-messages
pip install -e .
```

Remainings
```bash
pip install fastapi uvicorn
```

We will soon make things uv-syncable.

### Data preparation

```bash
conda activate otagent
# Download the eval dataset (OTTB-dev)
hf download open-thoughts/OpenThoughts-TB-dev --repo-type=dataset
# Download the train dataset
hf download open-thoughts/OpenThoughts-Agent-v1-RL --repo-type=dataset
# cd into the downloaded folder, say /path/to/.cache/huggingface/hub/datasets--open-thoughts--OpenThoughts-Agent-v1-RL/snapshots/hash_code
cd /path/to/.cache/huggingface/hub/datasets--open-thoughts--OpenThoughts-Agent-v1-RL/snapshots/hash_code
python extract_parquet_tasks.py tasks_new.parquet ./extracted_tasks
```

### Launch

Then configure the paths and API keys at the top of the script, and run:

```bash
cd SkyRL/skyrl-train
bash run_otagent.sh
```

The script is designed to run on 8 GPUs single-node. If that is not your setup, modify these configs correspondingly:

```bash
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.num_inference_engines=8 \
  generator.inference_engine_tensor_parallel_size=1 \
```
