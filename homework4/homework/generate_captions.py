import json, glob, os, random
from pathlib import Path

def load_info(path):
    with open(path, "r") as f:
        return json.load(f)

def build_caption(info):
    parts = []

    if "track" in info:
        parts.append(f"A scene on the {info['track']} track")

    if "speed" in info:
        parts.append(f"moving at speed {info['speed']}")

    if "objects" in info and len(info["objects"]) > 0:
        objs = ", ".join(info["objects"])
        parts.append(f"with objects {objs}")

    if "opponents" in info:
        parts.append(f"and {len(info['opponents'])} opponents")

    if "weapon" in info and info["weapon"]:
        parts.append(f"holding a {info['weapon']} weapon")

    return ", ".join(parts)

def create(stk_root, output_file):
    train_dir = os.path.join(stk_root, "data", "train")
    info_files = sorted(glob.glob(os.path.join(train_dir, "*_info.json")))

    all_caps = []

    for info_file in info_files:
        base = info_file.replace("_info.json", "")
        image_files = sorted(glob.glob(base + "_*_im.jpg"))

        info = load_info(info_file)
        caption = build_caption(info)

        for img in image_files:
            image_file = os.path.basename(img)   # <-- IMPORTANT FIX
            all_caps.append({"image_file": image_file, "caption": caption})

    random.shuffle(all_caps)

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(all_caps, f, indent=2)

    print(f"Generated {len(all_caps)} captions → {output_file}")


if __name__ == "__main__":
    import fire
    fire.Fire({"create": create})
