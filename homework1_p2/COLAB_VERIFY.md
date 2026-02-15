# Colab Verification for Online Grader

The **online grader** uses a different "safe_grader" that replaces the base (frozen) part of LoRA/QLoRA layers with **identity** in certain configurations. This makes the backward accuracy test harder—the LoRA adapters must learn with less help from the pretrained base.

## Quick Verify in Colab

1. **Upload** your homework folder to Colab (or clone from your repo).

2. **Run the online grader simulation:**
   ```bash
   !cd /content/your_homework_folder && python -m grader homework -v
   ```
   If `grader/safe_grader.py` exists, you'll see "Testing grader loaded" and the identity replacement simulation runs.

3. **Check for 105/105** (100 + 5 extra credit). If QLoRA backward gets less than 10/10, the LoRA scaling may need adjustment.

## What the Online Grader Does Differently

From official grader logs:
- Replaces base with identity at **blocks [0, 2, 4], layer 4** (3rd linear in blocks 0,2,4)
- Replaces base with identity at **blocks [6, 8, 10], layer 0** (1st linear in blocks 3,4,5)
- Runs fit with **each** configuration and uses the **worst** accuracy for scoring
- QLoRA needs to reach 0.5–0.8 accuracy in 20 steps under these harder conditions

## Local Simulation

With `grader/safe_grader.py` in place, running:
```bash
python -m grader homework -v
```
will use the safe_grader (Testing grader) and simulate the online behavior. Your local Val grader (when safe_grader is missing) does NOT do identity replacement, which is why local can pass but online may not.
