from pathlib import Path
import fire
from matplotlib import pyplot as plt
from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Simple captions that actually work - no fancy stuff.
    """
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    ego_car = next((k for k in kart_objects if k["is_center_kart"]), None)
    if not ego_car:
        return []

    ego_name = ego_car['kart_name']
    num_karts = len(kart_objects)
    other_karts = [k for k in kart_objects if not k["is_center_kart"]]

    # Just make ONE simple caption per image
    if num_karts == 1:
        caption = f"{ego_name} racing alone on {track_name}"
    else:
        caption = f"{ego_name} racing with {len(other_karts)} others on {track_name}"

    return [caption]


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


def main():
    fire.Fire({"check": check_caption})


if __name__ == "__main__":
    main()