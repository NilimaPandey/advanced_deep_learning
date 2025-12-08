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

    # IMPORTANT: Only generate 2-3 captions per image (not all styles)
    # This forces variety across the dataset

    selected_styles = random.sample(range(5), min(3, 5))

    # STYLE 1: Descriptive racing scene
    if 0 in selected_styles:
        if num_karts == 1:
            solo_templates = [
                f"A {ego_name} kart racing alone on the {track_name} track",
                f"The {ego_name} racer driving solo through {track_name}",
                f"{ego_name} navigating the {track_name} circuit alone",
                f"Racing scene showing {ego_name} on {track_name} with no other karts visible",
                f"Solo {ego_name} racer on {track_name}",
                f"{ego_name} races alone at {track_name}",
                f"Empty track: just {ego_name} on {track_name}",
                f"Solitary {ego_name} driving through {track_name}"
            ]
            captions.append(random.choice(solo_templates))
        else:
            multi_templates = [
                f"{ego_name} racing on {track_name} with {len(other_karts)} other kart{'s' if len(other_karts) > 1 else ''}",
                f"Racing scene on {track_name}: {ego_name} competing against {len(other_karts)} racer{'s' if len(other_karts) > 1 else ''}",
                f"The {ego_name} kart on {track_name} track surrounded by {len(other_karts)} competitor{'s' if len(other_karts) > 1 else ''}",
                f"{ego_name} battles {len(other_karts)} kart{'s' if len(other_karts) > 1 else ''} at {track_name}",
                f"Racing action: {ego_name} vs {len(other_karts)} on {track_name}",
                f"{len(other_karts)} kart{'s' if len(other_karts) > 1 else ''} race against {ego_name} on {track_name}",
                f"{track_name} circuit: {ego_name} among {len(other_karts)} racer{'s' if len(other_karts) > 1 else ''}"
            ]
            captions.append(random.choice(multi_templates))

    # STYLE 2: Short direct statements
    if 1 in selected_styles:
        short_templates = [
            f"{ego_name} at {track_name}",
            f"{track_name} race featuring {ego_name}",
            f"{num_karts} kart{'s' if num_karts > 1 else ''} racing at {track_name}",
            f"View from {ego_name} on {track_name}",
            f"{ego_name} on {track_name} circuit",
            f"{track_name}: {ego_name} perspective",
            f"Racing at {track_name} - {ego_name}",
            f"{ego_name}'s view of {track_name}"
        ]
        captions.append(random.choice(short_templates))

    # STYLE 3: Question-answer pairs (only add ONE, not multiple)
    if 2 in selected_styles:
        qa_pairs = [
            f"Which kart is this? It's {ego_name}",
            f"What track are they on? {track_name}",
            f"How many karts are visible? {num_karts}",
            f"Who is the ego car? {ego_name}",
            f"Where is this race? {track_name}",
            f"Whose perspective is this? {ego_name}",
            f"What's the track name? {track_name}",
            f"How many racers? {num_karts}"
        ]
        captions.append(random.choice(qa_pairs))

    # STYLE 4: Spatial relationships (if other karts exist)
    if 3 in selected_styles and other_karts:
        front_karts = [k for k in other_karts if k['center'][1] < ego_car['center'][1]]
        behind_karts = [k for k in other_karts if k['center'][1] >= ego_car['center'][1]]

        spatial_options = []

        if front_karts:
            names = ', '.join([k['kart_name'] for k in front_karts[:2]])
            spatial_options.extend([
                f"{len(front_karts)} kart{'s' if len(front_karts) > 1 else ''} ahead: {names}",
                f"{names} racing ahead of {ego_name}",
                f"{ego_name} chasing {names}",
                f"{names} in front on {track_name}"
            ])

        if behind_karts:
            names = ', '.join([k['kart_name'] for k in behind_karts[:2]])
            spatial_options.extend([
                f"{ego_name} leading {len(behind_karts)}: {names}",
                f"{ego_name} ahead of {names}",
                f"{names} trailing {ego_name}",
                f"{ego_name} outpacing {names}"
            ])

        if spatial_options:
            captions.append(random.choice(spatial_options))

    # STYLE 5: Specific kart positions with more variety
    if 4 in selected_styles and other_karts:
        # Pick ONE random kart to describe
        kart = random.choice(other_karts)
        kart_name = kart["kart_name"]
        cx, cy = kart["center"]

        lr = "left" if cx < ego_car["center"][0] else "right"
        fb = "ahead" if cy < ego_car["center"][1] else "behind"

        position_templates = [
            f"{kart_name} is {lr} and {fb} of {ego_name}",
            f"{kart_name} positioned {fb} and to the {lr}",
            f"The {kart_name} kart {fb} on the {lr} side",
            f"{ego_name} sees {kart_name} {fb} and {lr}",
            f"{kart_name} to the {lr}, {fb}",
            f"From {ego_name}: {kart_name} {fb}-{lr}",
            f"{kart_name} {fb}, {lr} of {ego_name}"
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