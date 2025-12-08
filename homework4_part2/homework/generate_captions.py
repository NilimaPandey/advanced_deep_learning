from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate unique captions by incorporating image-specific details.
    """
    import random
    import hashlib
    from pathlib import Path

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    # Find ego car
    ego_car = next((k for k in kart_objects if k["is_center_kart"]), None)

    if not ego_car:
        return []

    ego_name = ego_car['kart_name']
    num_karts = len(kart_objects)
    other_karts = [k for k in kart_objects if not k["is_center_kart"]]

    # Extract frame info for uniqueness
    info_file = Path(info_path)
    frame_id = info_file.stem.replace('_info', '')

    # Create unique seed
    unique_seed = int(hashlib.md5(f"{info_path}_{view_index}".encode()).hexdigest()[:8], 16)
    rng = random.Random(unique_seed)

    captions = []

    # VIEW-SPECIFIC descriptions (camera angles make each unique)
    view_descriptions = {
        0: ["front view", "forward perspective", "ahead-facing camera", "frontal angle"],
        1: ["side view", "lateral perspective", "side-facing camera", "profile angle"],
        2: ["rear view", "back perspective", "rear-facing camera", "behind angle"],
        3: ["angled view", "diagonal perspective", "corner camera", "oblique angle"]
    }

    view_desc = rng.choice(view_descriptions.get(view_index, ["view"]))

    # STRATEGY: Make captions highly specific by including multiple unique elements

    # Template set 1: Include view angle
    if num_karts == 1:
        captions.append(rng.choice([
            f"{view_desc} of {ego_name} racing alone on {track_name}",
            f"{ego_name} solo on {track_name} from {view_desc}",
            f"{track_name} track with just {ego_name} visible in {view_desc}",
            f"empty {track_name}: only {ego_name} shown from {view_desc}",
        ]))
    else:
        captions.append(rng.choice([
            f"{view_desc} of {ego_name} racing with {len(other_karts)} kart{'s' if len(other_karts) > 1 else ''} on {track_name}",
            f"{ego_name} and {len(other_karts)} other{'s' if len(other_karts) > 1 else ''} on {track_name} in {view_desc}",
            f"{track_name} race: {num_karts} karts including {ego_name} from {view_desc}",
            f"{view_desc} showing {ego_name} among {num_karts} racer{'s' if num_karts > 1 else ''} on {track_name}",
        ]))

    # Template set 2: Include specific kart positions with view
    if other_karts:
        # Pick specific kart with position
        kart = rng.choice(other_karts)
        cx, cy = kart['center']

        # Detailed position
        if cx < ego_car['center'][0] - 20:
            h_pos = "far left"
        elif cx < ego_car['center'][0]:
            h_pos = "left"
        elif cx > ego_car['center'][0] + 20:
            h_pos = "far right"
        else:
            h_pos = "right"

        if cy < ego_car['center'][1] - 20:
            v_pos = "far ahead"
        elif cy < ego_car['center'][1]:
            v_pos = "ahead"
        elif cy > ego_car['center'][1] + 20:
            v_pos = "far behind"
        else:
            v_pos = "behind"

        captions.append(rng.choice([
            f"{kart['kart_name']} positioned {v_pos} and {h_pos} of {ego_name} at {track_name}",
            f"{ego_name} sees {kart['kart_name']} {v_pos}-{h_pos} on {track_name} ({view_desc})",
            f"{track_name}: {kart['kart_name']} {v_pos} {h_pos}, {ego_name} center",
            f"relative to {ego_name}: {kart['kart_name']} is {v_pos} and to the {h_pos}",
        ]))
    else:
        # If solo, use frame-specific descriptions
        captions.append(rng.choice([
            f"frame view of {ego_name} alone at {track_name}",
            f"{ego_name} solo racing captured from {view_desc} on {track_name}",
            f"single kart scene: {ego_name} on {track_name} circuit",
            f"{track_name} snapshot showing only {ego_name}",
        ]))

    # Template set 3: Numeric and specific
    captions.append(rng.choice([
        f"scene with {num_karts} kart{'s' if num_karts != 1 else ''}: {ego_name} at {track_name}",
        f"{track_name} featuring {ego_name} (total racers: {num_karts})",
        f"kart count {num_karts} on {track_name}, ego is {ego_name}",
        f"{ego_name} perspective on {track_name} with {num_karts} visible",
    ]))

    # Template set 4: Action-based with position counts
    if other_karts:
        front = len([k for k in other_karts if k['center'][1] < ego_car['center'][1]])
        behind = len([k for k in other_karts if k['center'][1] >= ego_car['center'][1]])
        left = len([k for k in other_karts if k['center'][0] < ego_car['center'][0]])
        right = len([k for k in other_karts if k['center'][0] >= ego_car['center'][0]])

        captions.append(rng.choice([
            f"{ego_name} on {track_name}: {front} ahead, {behind} behind",
            f"{track_name} positioning - {ego_name} with {left} left, {right} right",
            f"{ego_name} racing {track_name} ({front} in front, {behind} trailing)",
            f"tactical view: {ego_name} at {track_name}, {left}L {right}R",
        ]))

    # Only return 2-3 captions to keep total count reasonable
    return captions[:rng.randint(2, 3)]


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