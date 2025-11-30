import json, glob, os, random
from pathlib import Path


def load_info(info_path):
    with open(info_path, "r") as f:
        return json.load(f)


def generate_qa_for_info(info, image_file):
    qas = []

    # 1. Track
    if "track" in info:
        qas.append({
            "image_file": image_file,
            "question": "What track is this?",
            "answer": info["track"]
        })

    # 2. Speed
    if "speed" in info:
        qas.append({
            "image_file": image_file,
            "question": "How fast am I moving?",
            "answer": str(info["speed"])
        })

    # 3. Objects
    if "objects" in info:
        for obj in info["objects"]:
            qas.append({
                "image_file": image_file,
                "question": f"Do I see a {obj}?",
                "answer": "yes"
            })

    # 4. Opponents
    if "opponents" in info:
        qas.append({
            "image_file": image_file,
            "question": "How many opponents do I see?",
            "answer": str(len(info["opponents"]))
        })

    # 5. Weapon
    if "weapon" in info and info["weapon"]:
        qas.append({
            "image_file": image_file,
            "question": "What weapon do I have?",
            "answer": info["weapon"]
        })

    return qas


def create(stk_root, output_file):
    train_dir = os.path.join(stk_root, "data", "train")
    info_files = sorted(glob.glob(os.path.join(train_dir, "*_info.json")))

    all_qas = []

    for info_file in info_files:
        base = info_file.replace("_info.json", "")
        image_files = sorted(glob.glob(base + "_*_im.jpg"))

        info = load_info(info_file)

        for img in image_files:
            # Convert absolute → relative path
            img_rel = os.path.relpath(img, start=stk_root)
            all_qas.extend(generate_qa_for_info(info, img_rel))

    random.shuffle(all_qas)

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(all_qas, f, indent=2)

    print(f"Generated {len(all_qas)} QA pairs → {output_file}")


if __name__ == "__main__":
    import fire

    fire.Fire({"create": create})
