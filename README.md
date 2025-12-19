# LLM-Benchmark-ChildCaregiver
Repo for Benchmarking LLMs for Mimicking Child-Caregiver Language in Interaction

## Single-turn Response Generation
This script generates single-turn responses from fixed prompts for a target role (CHI or ADULT). 

```
python gen_single_turn.py \
  --prompt_path data/CHI.csv \
  --gen_path outputs/single_turn \
  --test_model_lst "['llama3','mistral']" \
  --month_lst "[24,36]" \
  --dialogue_num 10 \
  --max_length 15 
```


## Multi-turn Response Generation
This script generates multi-turn CHI–ADULT dialogues by simulating interactions between a reference model and one or more test models. It supports few-shot prompting, age control (in months), LoRA or fully fine-tuned models, silence proportion control, and dialogue-level CSV outputs.

```
python gen_multi_turn.py \
  --gen_path data/gen_multi.csv \
  --ref_path data/ref_dialogues.csv \
  --demo_path data/demo.csv \
  --test_role ADULT \
  --test_model_lst "['chatgpt-4o-latest']" \
  --ref_model_name chatgpt-4o-latest \
  --month_lst "[24,36]" \
  --dialogue_num 5
```



## MFeature Extraction & Analysis
TThis script extracts lexical, syntactic, alignment, diversity, repetition, and speech-act features from generated data (single-turn, multi-turn, or combined). 
```
python gen_multi_turn.py \
  --gen_path data/gen_multi.csv \
  --ref_path data/ref_dialogues.csv \
  --demo_path data/demo.csv \
  --test_role ADULT \
  --test_model_lst "['chatgpt-4o-latest']" \
  --ref_model_name chatgpt-4o-latest \
  --month_lst "[24,36]" \
  --dialogue_num 5
```