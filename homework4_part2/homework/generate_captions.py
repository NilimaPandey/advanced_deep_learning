from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate diverse captions with forced uniqueness using image file hash.
    """
    import random
    import hashlib

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    # Find ego car
    ego_car = next((k for k in kart_objects if k["is_center_kart"]), None)

    if not ego_car:
        return []

    ego_name = ego_car['kart_name']
    num_karts = len(kart_objects)
    other_karts = [k for k in kart_objects if not k["is_center_kart"]]

    # Create unique seed from file path to ensure different images get different captions
    unique_seed = int(hashlib.md5(f"{info_path}_{view_index}".encode()).hexdigest()[:8], 16)
    rng = random.Random(unique_seed)

    captions = []

    # Generate exactly 2 captions per image with maximum variety
    num_to_generate = 2

    # MEGA TEMPLATE POOL - Each category has many variations

    # Category 1: Solo racing (if alone)
    if num_karts == 1:
        templates = [
            f"A {ego_name} kart racing alone on the {track_name} track",
            f"The {ego_name} racer driving solo through {track_name}",
            f"{ego_name} navigating the {track_name} circuit alone",
            f"Racing scene showing {ego_name} on {track_name} with no other karts",
            f"Solo {ego_name} racer on {track_name} circuit",
            f"{ego_name} races alone at {track_name}",
            f"Empty {track_name} track with just {ego_name}",
            f"Solitary {ego_name} driving through {track_name}",
            f"{ego_name} owns the {track_name} track",
            f"Lone {ego_name} on {track_name}",
            f"{ego_name} has {track_name} to themselves",
            f"Only {ego_name} racing on {track_name}",
            f"{track_name} track: just {ego_name}",
            f"{ego_name} cruising solo on {track_name}",
            f"No competition for {ego_name} at {track_name}",
        ]
    else:
        # Category 2: Multi-kart racing
        templates = [
            f"{ego_name} racing on {track_name} with {len(other_karts)} other{'s' if len(other_karts) > 1 else ''}",
            f"Racing scene on {track_name}: {ego_name} vs {len(other_karts)}",
            f"The {ego_name} kart on {track_name} among {len(other_karts)} competitor{'s' if len(other_karts) > 1 else ''}",
            f"{ego_name} battles {len(other_karts)} on {track_name}",
            f"Racing action: {ego_name} faces {len(other_karts)} at {track_name}",
            f"{len(other_karts)} opponent{'s' if len(other_karts) > 1 else ''} challenge {ego_name} on {track_name}",
            f"{track_name} race: {ego_name} competing against {len(other_karts)}",
            f"{ego_name} fights for position with {len(other_karts)} on {track_name}",
            f"{track_name} circuit: {ego_name} in pack of {num_karts}",
            f"{ego_name} surrounded by {len(other_karts)} racer{'s' if len(other_karts) > 1 else ''} at {track_name}",
            f"Competitive {track_name} race: {ego_name} plus {len(other_karts)}",
            f"{ego_name} navigates {track_name} with {len(other_karts)} nearby",
            f"Close racing on {track_name}: {ego_name} and {len(other_karts)} other{'s' if len(other_karts) > 1 else ''}",
            f"{track_name} showdown: {ego_name} versus {len(other_karts)}",
            f"{ego_name} dueling {len(other_karts)} kart{'s' if len(other_karts) > 1 else ''} on {track_name}",
        ]

    # Add specific spatial descriptions if there are other karts
    if other_karts:
        front = [k for k in other_karts if k['center'][1] < ego_car['center'][1]]
        behind = [k for k in other_karts if k['center'][1] >= ego_car['center'][1]]

        if front:
            kart = rng.choice(front)
            templates.extend([
                f"{kart['kart_name']} ahead of {ego_name} on {track_name}",
                f"{ego_name} chasing {kart['kart_name']} at {track_name}",
                f"{kart['kart_name']} leads {ego_name} through {track_name}",
                f"{ego_name} pursuing {kart['kart_name']} on {track_name}",
                f"{track_name}: {kart['kart_name']} in front, {ego_name} following",
            ])

        if behind:
            kart = rng.choice(behind)
            templates.extend([
                f"{ego_name} leading {kart['kart_name']} on {track_name}",
                f"{ego_name} ahead of {kart['kart_name']} at {track_name}",
                f"{kart['kart_name']} chasing {ego_name} through {track_name}",
                f"{ego_name} outrunning {kart['kart_name']} on {track_name}",
                f"{track_name}: {ego_name} in front, {kart['kart_name']} behind",
            ])

    # Add short descriptive captions
    templates.extend([
        f"{ego_name} at {track_name}",
        f"{track_name} from {ego_name} view",
        f"{ego_name} on {track_name} circuit",
        f"{track_name} race: {ego_name}",
        f"Racing {track_name} as {ego_name}",
        f"{ego_name}'s perspective on {track_name}",
        f"Driver view: {ego_name} at {track_name}",
        f"{track_name} track featuring {ego_name}",
        f"{ego_name} navigates {track_name}",
        f"{track_name} circuit with {ego_name}",
    ])

    # Add question-based captions
    templates.extend([
        f"Which racer? {ego_name} on {track_name}",
        f"Where? {track_name} with {ego_name}",
        f"Ego car: {ego_name} at {track_name}",
        f"Track: {track_name}, Kart: {ego_name}",
        f"Who's driving? {ego_name} on {track_name}",
        f"{track_name} track, {ego_name} racing",
        f"Kart count: {num_karts} at {track_name}",
        f"{num_karts} racer{'s' if num_karts > 1 else ''} on {track_name} including {ego_name}",
    ])

    # Add position-based descriptions for specific karts
    if other_karts and len(other_karts) <= 3:
        for kart in other_karts[:2]:
            cx, cy = kart['center']
            lr = "left" if cx < ego_car["center"][0] else "right"
            fb = "front" if cy < ego_car["center"][1] else "back"

            templates.extend([
                f"{kart['kart_name']} to {ego_name}'s {lr} at {track_name}",
                f"{ego_name} sees {kart['kart_name']} {fb}-{lr} on {track_name}",
                f"{track_name}: {kart['kart_name']} {fb} and {lr}",
                f"{kart['kart_name']} positioned {fb}-{lr} of {ego_name}",
            ])

    # Randomly select exactly 2 captions from the huge pool
    if len(templates) >= num_to_generate:
        captions = rng.sample(templates, num_to_generate)
    else:
        captions = templates

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


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption})


if __name__ == "__main__":
    main()