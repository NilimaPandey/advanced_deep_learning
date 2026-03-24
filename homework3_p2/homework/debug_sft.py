"""
Diagnostic script: paste this into a Colab cell AFTER training SFT.
It shows exactly what the model generates so you can see what's happening.

Usage in Colab:
    %cd /content/homework3_p2
    %run homework/debug_sft.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from homework.sft import load as load_sft, SFTModel, tokenize, format_example
from homework.data import Dataset, is_answer_valid
from homework.base_llm import BaseLLM

print("=" * 60)
print("STEP 1: Check tokenization alignment")
print("=" * 60)

tokenizer = BaseLLM().tokenizer
ex = format_example("How many gram are there per 6 kg?", 6000.0)
print(f"format_example output: {ex}")

tok_result = tokenize(tokenizer, ex["question"], ex["answer"])
decoded_ids = tokenizer.decode(tok_result["input_ids"], skip_special_tokens=False)
print(f"Full tokenized+decoded: {decoded_ids!r}")

prompt_text = "How many gram are there per 6 kg? "
prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
print(f"Prompt token IDs: {prompt_ids}")
print(f"Prompt decoded: {tokenizer.decode(prompt_ids, skip_special_tokens=False)!r}")

labels = tok_result["labels"]
supervised_ids = [tok_result["input_ids"][i] for i in range(len(labels)) if labels[i] != -100]
print(f"Supervised token IDs: {supervised_ids}")
print(f"Supervised decoded: {tokenizer.decode(supervised_ids, skip_special_tokens=False)!r}")

num_supervised = sum(1 for l in labels if l != -100)
print(f"Num supervised tokens: {num_supervised}, total tokens: {sum(tok_result['attention_mask'])}")

print()
print("=" * 60)
print("STEP 2: Load trained SFT model and generate")
print("=" * 60)

try:
    llm = load_sft()
    print("SFT model loaded successfully!")
except Exception as e:
    print(f"ERROR loading SFT model: {e}")
    sys.exit(1)

dataset = Dataset("valid")
test_questions = [dataset[i][0] for i in range(10)]
test_answers = [dataset[i][1] for i in range(10)]

print("\nGenerating answers for 10 validation questions...")
prompts = [llm.format_prompt(q) for q in test_questions]
generations = llm.batched_generate(prompts)

correct = 0
answered = 0
for i in range(10):
    parsed = llm.parse_answer(generations[i])
    is_nan = parsed != parsed
    is_correct = not is_nan and is_answer_valid(parsed, test_answers[i])
    if not is_nan:
        answered += 1
    if is_correct:
        correct += 1
    status = "CORRECT" if is_correct else ("NO_ANSWER" if is_nan else "WRONG")
    print(f"\n  Q: {test_questions[i]}")
    print(f"  Expected: {test_answers[i]}")
    print(f"  Raw generation: {generations[i]!r}")
    print(f"  Parsed: {parsed}  [{status}]")

print(f"\n  Accuracy: {correct}/10 = {correct/10:.1%}")
print(f"  Answer rate: {answered}/10 = {answered/10:.1%}")

print()
print("=" * 60)
print("STEP 3: Full benchmark on 100 questions")
print("=" * 60)
from homework.data import benchmark
result = benchmark(llm, dataset, 100)
print(f"  accuracy={result.accuracy:.3f}  answer_rate={result.answer_rate:.3f}")
if result.accuracy >= 0.4:
    print("  --> PASSES SFT grader threshold (0.4)")
else:
    print(f"  --> BELOW SFT grader threshold (need 0.4, got {result.accuracy:.3f})")
