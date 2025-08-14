from pathlib import Path
from PIL import Image
from rembg import remove, new_session
from io import BytesIO

INPUT_DIR = "./rembg/120_faces_women"
OUTPUT_DIR = "./rembg/120_faces_women_after"

in_dir = Path(INPUT_DIR)
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(exist_ok=True, parents=True)

valid_ext = {".jpg", ".jpeg", ".png", ".webp"}

# Count all files to process
all_files = [p for p in in_dir.iterdir() if p.suffix.lower() in valid_ext]
total_files = len(all_files)
processed = 0

session = new_session("isnet-general-use")

F_THRESHOLD = 220 
B_THRESHOLD = 10
ERODE_SIZE   = 1

for p in all_files:
    processed += 1
    
    output_path = out_dir / f"{p.stem}.jpg"
    if output_path.exists():
        continue

    print(f"Processing {processed}/{total_files}: {p.name}")

    with open(p, "rb") as f:
        input_bytes = f.read()

    output_bytes = remove(
        input_bytes,
        session=session,
        alpha_matting=True,
        alpha_matting_foreground_threshold=F_THRESHOLD,
        alpha_matting_background_threshold=B_THRESHOLD,
        alpha_matting_erode_size=ERODE_SIZE,
        post_process_mask=False,
    )

    fg = Image.open(BytesIO(output_bytes)).convert("RGBA")
    white_bg = Image.new("RGBA", fg.size, (255, 255, 255, 255))
    composed = Image.alpha_composite(white_bg, fg).convert("RGB")
    composed.save(output_path, quality=95)

print("Processing completed")
