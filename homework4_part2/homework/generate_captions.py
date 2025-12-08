from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate SINGLE unique caption per image with maximum specificity.
    Returns exactly 1 caption per image to maximize diversity across dataset.
    """
    import random
    import hashlib
    from pathlib import Path

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    ego_car = next((k for k in kart_objects if k["is_center_kart"]), None)
    if not ego_car:
        return []

    ego_name = ego_car['kart_name']
    num_karts = len(kart_objects)
    other_karts = [k for k in kart_objects if not k["is_center_kart"]]

    # Unique seed per image ensures deterministic but varied caption selection
    unique_seed = int(hashlib.md5(f"{info_path}_{view_index}".encode()).hexdigest()[:8], 16)
    rng = random.Random(unique_seed)

    # Camera view descriptions for uniqueness
    views = ["front camera", "side camera", "rear camera", "angled camera"]
    view_desc = views[view_index % 4]

    # Count spatial positions for detailed descriptions
    if other_karts:
        front = len([k for k in other_karts if k['center'][1] < ego_car['center'][1]])
        behind = len([k for k in other_karts if k['center'][1] >= ego_car['center'][1]])
        left = len([k for k in other_karts if k['center'][0] < ego_car['center'][0]])
        right = len([k for k in other_karts if k['center'][0] >= ego_car['center'][0]])

        # Pick one specific kart with detailed position
        picked_kart = rng.choice(other_karts)
        kx, ky = picked_kart['center']
        ex, ey = ego_car['center']

        # Very detailed position descriptions
        if kx < ex - 30:
            h_pos = "far left"
        elif kx < ex - 10:
            h_pos = "left"
        elif kx > ex + 30:
            h_pos = "far right"
        elif kx > ex + 10:
            h_pos = "right"
        else:
            h_pos = "center"

        if ky < ey - 30:
            v_pos = "far ahead"
        elif ky < ey - 10:
            v_pos = "ahead"
        elif ky > ey + 30:
            v_pos = "far behind"
        elif ky > ey + 10:
            v_pos = "behind"
        else:
            v_pos = "level"

    # Generate ONE unique caption using multiple specific details
    # More templates = more diversity even with duplicate scenarios
    template_choice = rng.randint(0, 9)

    if num_karts == 1:
        # Solo racing - include view and track details
        templates = [
            f"{view_desc} shows {ego_name} racing alone on {track_name}",
            f"solo {ego_name} kart on {track_name} from {view_desc}",
            f"{track_name} circuit: only {ego_name} visible via {view_desc}",
            f"{ego_name} driving solo through {track_name} ({view_desc})",
            f"empty {track_name} track except {ego_name} in {view_desc}",
            f"{view_desc} perspective of lone {ego_name} at {track_name}",
            f"isolated {ego_name} on {track_name} captured by {view_desc}",
            f"{track_name}: single kart {ego_name} shown from {view_desc}",
            f"{ego_name} has {track_name} alone as seen from {view_desc}",
            f"no opponents for {ego_name} on {track_name} in this {view_desc}",
        ]
    else:
        # Multi-kart racing - include specific positional details
        templates = [
            f"{view_desc}: {ego_name} on {track_name} with {picked_kart['kart_name']} {v_pos} and {h_pos}",
            f"{track_name} race - {ego_name} has {front} ahead, {behind} behind (via {view_desc})",
            f"{ego_name} at {track_name}: {left} to left, {right} to right, total {num_karts} karts",
            f"from {view_desc}, {ego_name} sees {picked_kart['kart_name']} {v_pos}-{h_pos} on {track_name}",
            f"{track_name}: {ego_name} racing, {picked_kart['kart_name']} positioned {v_pos} {h_pos}",
            f"{num_karts} kart race on {track_name} - {ego_name} vs others ({view_desc})",
            f"{view_desc} of {ego_name} on {track_name}: {front}F {behind}B {left}L {right}R",
            f"{track_name} - {ego_name} with {picked_kart['kart_name']} {v_pos} and to the {h_pos}",
            f"tactical {view_desc}: {ego_name} at {track_name}, {num_karts} racers, {front} in front",
            f"{ego_name} navigates {track_name} ({num_karts} total), {picked_kart['kart_name']} is {v_pos}-{h_pos}",
        ]

    # Return ONLY ONE caption per image
    return [templates[template_choice]]


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


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption})


if __name__ == "__main__":
    main()