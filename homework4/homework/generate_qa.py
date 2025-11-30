import json, glob, os, random
from pathlib import Path

"""
Automatic question-answer generation for SuperTuxKart.
This version works with the dataset structure:

    stk_root/
        data/
            train/
                00000_info.json
                00000_00_im.jpg
                00000_01_im.jpg
                00000_02_im.jpg
                ...
"""

def load_info(info_path):
    with open(info_path, "r") as f:
        return json.load(f)

def generate_qa_for_info(info, image_path):
    qas = []

    # 1. Track name
    if "track" in info:
        qas.append({
            "image_path": image_path,
            "question": "What track is this?",
            "answer": info["track"]
        })

    # 2. Speed
    if "speed" in info:
        qas.append({
            "image_path": image_path,
            "question": "How fast am I moving?",
            "answer": str(info["speed"])
        })

    # 3. Objects visible
    if "objects" in info:
        for obj in info["objects"]:
            qas.append({
                "image_path": image_path,
                "question": f"Do I see a {obj}?",
                "answer": "yes"
            })

    # 4. Opponents
    if "opponents" in info:
        qas.append({
            "image_path": image_path,
            "question": "How many opponents do I see?",
            "answer": str(len(info["opponents"]))
        })

    # 5. Weapon
    if "weapon" in info and info["weapon"]:
        qas.append({
            "image_path": image_path,
            "question": "What weapon do I have?",
            "answer": info["weapon"]
        })

    return qas


def create(stk_root, output_file):
    """
    Create QA dataset from SuperTuxKart files.

    stk_root should be folder containing:
        stk_root/data/train/*.json + *_im.jpg
    """
    train_dir = os.path.join(stk_root, "data", "train")
    info_files = sorted(glob.glob(os.path.join(train_dir, "*_info.json")))

    all_qas = []

    for info_file in info_files:
        base = info_file.replace("_info.json", "")
        # Match ANY *_XX_im.jpg
        image_files = sorted(glob.glob(base + "_*_im.jpg"))

        info = load_info(info_file)

        # Generate QA for each image
        for img in image_files:
            all_qas.extend(generate_qa_for_info(info, img))

    random.shuffle(all_qas)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(all_qas, f, indent=2)

    print(f"Generated {len(all_qas)} QA pairs → {output_file}")


if __name__ == "__main__":
    import fire
    fire.Fire({"create": create})
