import json, glob, os, random
from pathlib import Path

def load_info(info_path):
    with open(info_path, "r") as f:
        return json.load(f)

def generate_qa_for_info(info, image_path):
    return [
        {"image_path": image_path, "question": "What track is this?", "answer": info.get("track", "")},
        {"image_path": image_path, "question": "How fast am I moving?", "answer": str(info.get("speed", ""))},
        {"image_path": image_path, "question": "How many opponents do I see?", "answer": str(len(info.get("opponents", [])))}
    ] + [
        {"image_path": image_path, "question": f"Do I see a {obj}?", "answer": "yes"}
        for obj in info.get("objects", [])
    ] + (
        [{"image_path": image_path, "question": "What weapon do I have?", "answer": info["weapon"]}]
        if info.get("weapon") else []
    )


def create(stk_root, output_file):
    train_dir = os.path.join(stk_root, "data", "train")
    info_files = sorted(glob.glob(os.path.join(train_dir, "*_info.json")))

    all_qas = []

    for info_file in info_files:
        base = info_file.replace("_info.json", "")
        imgs = sorted(glob.glob(base + "_*_im.jpg"))
        info = load_info(info_file)

        for img in imgs:
            # ABSOLUTE PATH required for finetune.py
            img_abs = os.path.abspath(img)
            all_qas.extend(generate_qa_for_info(info, img_abs))

    random.shuffle(all_qas)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_qas, f, indent=2)

    print(f"Generated {len(all_qas)} QA → {output_file}")


if __name__ == "__main__":
    import fire
    fire.Fire({"create": create})
