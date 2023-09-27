# Safety-Tuned LLaMAs


**Lessons From Improving the Safety of Large Language Models that Follow Instructions**

<img src="images/main.png" alt="drawing" width="500"/>

## Starting Point

SafetyDatasets are available under the `data/evaluation` directory.

Training data is available under the `data/training` directory. Where you will find the instruction-output pairs.

## Evaluations

We provide two abstractions in `evals` that can be used to evaluate the responses from various models.

For the HarmfulnessRewardModel.

```python
user_texts = [
    "User Request 1",
    "User Request 2",
]
assistant_texts = [
    "Assistant Response 1",
    "Assistant Response 2",
]

# Build the conversation with the correct template.
conversation = ConversationBuilder()
inputs = [conversation.build(u, a) for u, a in zip(user_texts, assistant_texts)]
print(inputs)

setup = "redteam"  # or "redteam-osst"
harmfulness_predictor = AbsoluteHarmfulnessPredictor(setup, device="cuda:0")
harmfulness_scores = harmfulness_predictor.predict(inputs)

print(harmfulness_scores)
```

For the OpenAI Evaluator:

```python

cm = ContentModeration()
scores = cm.content_moderation(assistant_texts)

```

## Generation

```bash
python generation/generate_answers.py \
    --prompt_template_path ./configs/alpaca.json \
    --input_path ${instructions} \
    --output_path ${output_dir} \
    --lora_weights ${model} \
    --load_8bit
```