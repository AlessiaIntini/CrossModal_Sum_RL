# Video summarization on VideoXum with Qwen-VL: SFT, DPO and GRPO

Code for training and evaluating vision-language models on the three tasks of
VideoXum:

- **V2V**: select the frames of a video summary, answered as a JSON array of
  frame intervals;
- **V2T**: write a text summary of the video;
- **V2VT**: select the frames and write the summary in a single JSON object.

The models are Qwen3-VL (Qwen2.5-VL is also supported), fine-tuned with LoRA
adapters through [TRL](https://github.com/huggingface/trl). Training proceeds in
three stages, supervised fine-tuning, Direct Preference Optimization and GRPO
with a composite reward. Zero-shot baselines cover local models (Qwen3-VL,
InternVL3.5, VideoLLaMA3) and API models through OpenRouter (Gemini).

## Repository layout

```text
videoxum/                 shared package
  prompts.py              prompts of every task and of the zero-shot baselines
  data.py                 VideoXum records and datasets
  video.py                1 fps frame decoding and the video encoding of checkpoints
  inputs.py               chat messages, processor inputs and generation
  parsing.py              strict and lenient parsers of model responses
  metrics.py              VideoXum selection metrics, collapse diagnostics, text metrics
  rewards.py              GRPO reward
  baselines.py            random and positional selections of equal size
  models.py               loading of base models and trained adapters
  inference.py            test-set inference loops
  training/               SFT, DPO and GRPO trainers and their validation
  zero_shot/              zero-shot backends (local models, OpenRouter)
train_sft.py              SFT entry point (--task v2v | v2t | v2vt)
train_dpo.py              DPO entry point
train_grpo.py             GRPO entry point
infer.py                  test-set inference of a trained adapter (--task v2v | v2t | v2vt)
infer_score_sampling.py   V2VT inference with per-frame scores from sampled responses
zero_shot.py              zero-shot baselines (--backend local | openrouter)
evaluation/               evaluation and paper tables, run as python -m evaluation.<script>
configuration/            consensus targets and DPO preference pairs
tools/                    conversion of VideoLLaMA3 to the transformers format
script/                   SLURM launchers of every stage
```

## Installation

```bash
pip install -r requirements.txt
pip install -r requirements-evaluation.txt
```

`trl` is pinned to 1.13.0 because the trainers override private TRL methods.
`evaluation/eval_tsum_coco.py` also needs Java.

## Data

The default paths are in `videoxum/paths.py`:

- **metadata**: `train/val/test_videoxum.json`, with `sampled_frames`,
  `vsum_onehot` (10 binary annotations at 1 fps), `vsum` and `tsum`;
- **videos**: one directory per split;
- **BLIP frame features**: used by the GRPO reward;
- **consensus targets and DPO pairs**: `configuration/create_consensus_dpo_data.py`;
  its docstring describes how the consensus targets and the rejected
  selections are built.

Labels can be expressed on frames (`--timeline_unit frame`) or on pairs of
consecutive frames (`--timeline_unit pair`, matching the temporal patches of
Qwen3-VL). Metrics are always computed on the original 1 fps frames.

## Pipeline

```bash
# 0. Consensus targets and DPO preference pairs of the training set
python configuration/create_consensus_dpo_data.py --timeline-unit pair

# 1. Supervised fine-tuning (V2V, V2T or V2VT target)
python train_sft.py --task v2vt --timeline_unit pair --max_pixels 100352 --min_pixels 3136 \
    --saveroot outputs/sft_v2vt/checkpoints --resultsroot outputs/sft_v2vt/results

# 2. DPO on chosen/rejected selections, starting from a V2V SFT adapter
python train_dpo.py --init_from outputs/sft_v2v/checkpoints/Qwen3-VL-8B-Instruct/checkpoint-best \
    --no-length-normalize --saveroot outputs/dpo/checkpoints --resultsroot outputs/dpo/results

# 3. GRPO on the joint task, starting from an SFT (or DPO) adapter
python train_grpo.py --init_from outputs/sft_v2vt/checkpoints/Qwen3-VL-8B-Instruct/checkpoint-best \
    --saveroot outputs/grpo/checkpoints --resultsroot outputs/grpo/results

# Test-set inference
python infer.py --task v2vt --checkpoint outputs/grpo/checkpoints/Qwen3-VL-8B-Instruct/checkpoint-best

# Zero-shot baselines
python zero_shot.py --backend local --model models/Qwen3-VL-8B-Instruct --task v2vt
OPENROUTER_API_KEY=... python zero_shot.py --backend openrouter --model google/gemini-3.5-flash --task v2vt
```

Every checkpoint stores the LoRA adapter and `video_encoding.json`, which
records the timeline unit, temporal markers, pixel budget and task. Later
stages and inference inherit this encoding from the checkpoint they start
from. GRPO checkpoints also record the merged starting adapter in
`grpo_init.json`, and DPO checkpoints record it in `dpo_init.json`.

With `--time_budget_hours`, SFT and DPO save a resumable `checkpoint-N` and exit
with code 85 before the SLURM time limit; the launchers in `script/` then
requeue themselves with `--resume_from`.

## Evaluation

```bash
python -m evaluation.eval_vsum --predictions <predictions.jsonl> --annotations <test_videoxum.json> \
    --output <out.jsonl> --index-base 0                        # F1, Kendall, Spearman
python -m evaluation.eval_tsum_coco --candidates-json <predictions.jsonl> --references-json <test.json> \
    --output-json <out.json>                                   # BLEU, METEOR, ROUGE-L, CIDEr (COCO protocol)
python -m evaluation.eval_cross_fclip --predictions <predictions.jsonl> --annotations <test_videoxum.json> \
    --feature-dir <clip features> --clip-model <clip checkpoint> --output <out.jsonl>
python -m evaluation.results_table                             # summary table of all experiments
```

The other scripts in `evaluation/` cover:

- F1 against equal-size random selections: `eval_vsum_por`,
  `matched_baselines_table`;
- frame-level F_CLIP: `eval_vsum_fclip`;
- output-collapse diagnostics and run comparisons: `collapse_report`,
  `compare_runs`;
- paired bootstrap significance tests: `significance_tests`.
