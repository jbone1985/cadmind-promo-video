# CADmind promo video builder

Builds `cadmind_promo_v2.mp4` (35s, 1920x1080, 30fps) from CADmind gallery renders, the CADmind logo, and CC BY 3D-printing b-roll.

## Requirements
- Python 3 with `numpy`, `pillow`, `scipy`
- `ffmpeg` on PATH
- IBM Plex Mono font (or edit the font paths in the script)
- `yt-dlp` to download the b-roll

## Setup
1. Edit `ROOT` near the top of `make_promo_v1.py` to point to this folder.
2. The b-roll clips are not bundled (licensing and size). Download them into `broll/` with yt-dlp using the URLs in `broll/CREDITS.txt`, keeping the filenames the script expects:
   - `broll/Adafruit_Industries_Raspberry_Pi_5_3D_Printed_Case_0ERqsF9wzvo.mp4`
   - `broll/KPrint_3D_Printed_Vertical_Laptop_Stand_tIXM9_x7KPg.mp4`
   - `broll/Uncle_Duncans_Electronics_Shack_ZK4KX_Bench_PSU_Enclosure_Jo6VrJHi-nc.mp4`
   - `broll/YO2BTW_I_Made_My_Own_Headphone_Stand_2f2LZJlDc5A.mp4`
3. Run `python3 make_promo_v1.py`.

All b-roll is Creative Commons Attribution: keep the on-screen credits and credit the creators in the post.
