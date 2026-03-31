import json
from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    with open(info_path) as f:
        info = json.load(f)

    kart_names = info["karts"]
    ego_name = kart_names[view_index]
    track_name = extract_track_info(info_path)
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)

    if not karts:
        return []

    center_x = img_width / 2
    center_y = img_height / 2

    captions = []

    # 1. Ego car
    captions.append(f"{ego_name} is the ego car.")

    # 2. Counting
    captions.append(f"There are {len(karts)} karts in the scene.")

    # 3. Track name
    captions.append(f"The track is {track_name}.")

    # 4. Relative position for each non-ego kart
    for kart in karts:
        if kart["instance_id"] == view_index:
            continue
        kx, ky = kart["center"]
        name = kart["kart_name"]

        lr = "left" if kx < center_x else "right"
        fb = "in front" if ky < center_y else "behind"

        captions.append(f"{name} is {fb} of the ego car.")
        captions.append(f"{name} is {lr} of the ego car.")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


def generate_all_captions(data_split: str = "train", output_name: str = "example_captions.json"):
    """
    Generate captions for all frames and views in a data split.
    """
    data_dir = Path(__file__).parent.parent / "data" / data_split
    info_files = sorted(data_dir.glob("*_info.json"))

    all_captions = []
    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")
        for view_index in range(10):
            image_file = f"{data_split}/{base_name}_{view_index:02d}_im.jpg"
            image_path = data_dir / f"{base_name}_{view_index:02d}_im.jpg"
            if not image_path.exists():
                continue
            try:
                captions = generate_caption(str(info_file), view_index)
                for caption in captions:
                    all_captions.append({
                        "image_file": image_file,
                        "caption": caption,
                    })
            except Exception:
                continue

    output_path = data_dir / output_name
    with open(output_path, "w") as f:
        json.dump(all_captions, f, indent=2)

    print(f"Generated {len(all_captions)} captions, saved to {output_path}")


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption, "generate": generate_all_captions})


if __name__ == "__main__":
    main()
