# Safety-Tuned LLaMAs: ICLR 2024

**Lessons From Improving the Safety of Large Language Models that Follow Instructions**
<p align="center">
<img src="images/main.png" alt="drawing" width="500"/>
</p>

## Citation

Please consider citing the following paper if you use this code or data in your work:


```
@inproceedings{
bianchi2024safetytuned,
title={Safety-Tuned {LL}a{MA}s: Lessons From Improving the Safety of Large Language Models that Follow Instructions},
author={Federico Bianchi and Mirac Suzgun and Giuseppe Attanasio and Paul Rottger and Dan Jurafsky and Tatsunori Hashimoto and James Zou},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=gT5hALch9z}
}
```

## Starting Point

SafetyDatasets are available under the `data/evaluation` directory.

Training data is available under the `data/training` directory. Where you will find the instruction-output pairs.

## Tuning and Generation

Fine-tuning code and generation come from [Alpaca-LoRa](https://github.com/tloen/alpaca-lora) repository.

## Evaluations

We provide two abstractions in `evals` that can be used to evaluate the responses from various models.

For the HarmfulnessRewardModel.

```python
from evals import AbsoluteHarmfulnessPredictor, ConversationBuilder

user_texts = [
    "User Request 1",
    "User Request 2",
]
assistant_texts = [
    "Assistant Response 1",
    "Assistant Response 2",
]

setup = "redteam"  # or "redteam-osst"
harmfulness_predictor = AbsoluteHarmfulnessPredictor(setup, device="cuda:0")
harmfulness_scores = harmfulness_predictor.predict(user_texts, assistant_texts)

print(harmfulness_scores)
```

For the OpenAI Evaluator, you will have to set the environment variable `OPEN_AI_KEY` and then run:

```python
from evals import ContentModeration

cm = ContentModeration()
scores = cm.content_moderation(assistant_texts)

```

## Script to run Generation

The following script should run with any of our safety datasets. Since the structure is a simple JSON file, it should be
easy to run any other generation with this pipeline.

```bash
python generation/generate_answers.py \
    --prompt_template_path ./configs/alpaca.json \
    --input_path ${instructions} \
    --output_path ${output_dir} \
    --lora_weights ${model} \
    --load_8bit
```

# Licensing

* Code is licensed under the MIT License. 

* Due to the fact that some of the data is GPT-generated and comes from other work, Data is licensed under the Creative Commons Attribution Non Commercial 4.0 License. For SafeText data, also referred as PhysicalSafety in our paper, please refer to [1].

[1] Levy, S., Allaway, E., Subbiah, M., Chilton, L., Patton, D., McKeown, K., & Wang, W. Y. (2022). Safetext: A benchmark for exploring physical safety in language models. EMNLP.