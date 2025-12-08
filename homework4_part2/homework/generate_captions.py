from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate diverse captions for a specific view to improve CLIP training.
    Returns multiple caption styles for better diversity.
    """
    import random

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    captions = []

    # Find ego car
    ego_car = next((k for k in kart_objects if k["is_center_kart"]), None)

    if not ego_car:
        return captions

    ego_name = ego_car['kart_name']
    num_karts = len(kart_objects)
    other_karts = [k for k in kart_objects if not k["is_center_kart"]]

    # STYLE 1: Descriptive racing scene
    if num_karts == 1:
        # Solo racing
        solo_templates = [
            f"A {ego_name} kart racing alone on the {track_name} track",
            f"The {ego_name} racer driving solo through {track_name}",
            f"{ego_name} navigating the {track_name} circuit",
            f"Racing scene showing {ego_name} on {track_name} with no other karts visible"
        ]
        captions.append(random.choice(solo_templates))
    else:
        # Racing with others
        multi_templates = [
            f"{ego_name} racing on {track_name} with {len(other_karts)} other kart{'s' if len(other_karts) > 1 else ''}",
            f"Racing scene on {track_name}: {ego_name} competing against {len(other_karts)} racer{'s' if len(other_karts) > 1 else ''}",
            f"The {ego_name} kart on {track_name} track surrounded by {len(other_karts)} competitor{'s' if len(other_karts) > 1 else ''}"
        ]
        captions.append(random.choice(multi_templates))

    # STYLE 2: Short direct statements
    short_templates = [
        f"{ego_name} at {track_name}",
        f"{track_name} race featuring {ego_name}",
        f"{num_karts} kart{'s' if num_karts > 1 else ''} racing at {track_name}",
        f"View from {ego_name} on {track_name}"
    ]
    captions.append(random.choice(short_templates))

    # STYLE 3: Question-answer pairs (good for alignment)
    qa_pairs = [
        f"Which kart is this? It's {ego_name}",
        f"What track are they on? {track_name}",
        f"How many karts are visible? {num_karts}",
        f"Who is the ego car? {ego_name}"
    ]
    captions.extend(random.sample(qa_pairs, min(2, len(qa_pairs))))

    # STYLE 4: Spatial relationships (if other karts exist)
    if other_karts:
        # Count positions
        front_karts = [k for k in other_karts if k['center'][1] < ego_car['center'][1]]
        behind_karts = [k for k in other_karts if k['center'][1] >= ego_car['center'][1]]
        left_karts = [k for k in other_karts if k['center'][0] < ego_car['center'][0]]
        right_karts = [k for k in other_karts if k['center'][0] >= ego_car['center'][0]]

        # Add position-based captions
        if front_karts:
            names = ', '.join([k['kart_name'] for k in front_karts[:2]])
            captions.append(f"{len(front_karts)} kart{'s' if len(front_karts) > 1 else ''} ahead: {names}")

        if behind_karts:
            names = ', '.join([k['kart_name'] for k in behind_karts[:2]])
            captions.append(f"{ego_name} leading {len(behind_karts)}: {names}")

        # Find nearest kart
        nearest = min(other_karts, key=lambda k:
        ((k['center'][0] - ego_car['center'][0]) ** 2 +
         (k['center'][1] - ego_car['center'][1]) ** 2) ** 0.5)

        spatial_templates = [
            f"{ego_name} racing near {nearest['kart_name']} on {track_name}",
            f"View from {ego_name} with {nearest['kart_name']} nearby"
        ]
        captions.append(random.choice(spatial_templates))

    # STYLE 5: Detailed individual kart positions (for variety)
    if len(other_karts) <= 3:  # Only if not too many
        for kart in other_karts[:2]:  # Max 2 to avoid repetition
            kart_name = kart["kart_name"]
            cx, cy = kart["center"]

            # Determine position
            lr = "left" if cx < ego_car["center"][0] else "right"
            fb = "ahead" if cy < ego_car["center"][1] else "behind"

            position_templates = [
                f"{kart_name} is {lr} and {fb} of {ego_name}",
                f"{kart_name} positioned {fb} and to the {lr}",
                f"The {kart_name} kart {fb} on the {lr} side"
            ]
            captions.append(random.choice(position_templates))

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