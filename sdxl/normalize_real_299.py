# normalize_to_299.py
from pathlib import Path
from PIL import Image, ImageOps

SRC = Path(r"D:\VSC_research\generative-models\sdxl\generated_image_sdxl_512_no_sc_3_5k")
DST = Path(r"D:\VSC_research\generative-models\sdxl\generated_image_sdxl_512_no_sc_3_5k_299")

DST.mkdir(parents=True, exist_ok=True)

ex = {".jpg", ".jpeg", ".png", ".webp"}
n_bad = 0
for p in SRC.iterdir():
    if p.suffix.lower() not in ex: 
        continue
    try:
        im = Image.open(p)
        im = ImageOps.exif_transpose(im).convert("RGB")  # fix EXIF orientation, ensure RGB
        # resize so short side == 299, then center-crop to 299×299 (matches typical FID preprocessing)
        w, h = im.size
        if min(w, h) == 0:
            raise ValueError("zero dimension")
        scale = 299 / min(w, h)
        im = im.resize((int(round(w*scale)), int(round(h*scale))), Image.BICUBIC)
        w2, h2 = im.size
        left = max(0, (w2 - 299)//2); top = max(0, (h2 - 299)//2)
        im = im.crop((left, top, left+299, top+299))
        im.save(DST / (p.stem + ".jpg"), "JPEG", quality=95)
    except Exception as e:
        n_bad += 1
        # print(f"skip {p.name}: {e}")
print("done. bad files skipped:", n_bad)


