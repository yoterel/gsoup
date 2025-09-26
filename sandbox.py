import gsoup
from pathlib import Path
###############
# images = gsoup.load_images(Path("./bla"))
# text_strings = [f"Iter {i}" for i in range(len(images))]
# images = gsoup.draw_text_on_image(images, text_strings, )
# gsoup.save_video(images, Path("./bla.mp4"), 1)
####################
dst_path = Path("./bla")
dst_path.mkdir(parents=True, exist_ok=True)
gray = gsoup.GrayCode()
# set projector resolution
proj_wh = (800, 600)
# set decoding mode (rows first or columns first)
mode = "ij"  # "ij" is rows first, "xy" is columns first
# generate the gray code patterns for structured light projection
orig_patterns = gray.encode(proj_wh)
gsoup.save_images(orig_patterns, dst_path)