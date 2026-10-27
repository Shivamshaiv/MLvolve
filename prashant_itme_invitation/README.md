# Prashant Group – India ITME 2026 customer invitation

Two finished videos, rebuilt from the supplied 30-second, 720×1280 invitation.

| File | Format | Use |
|---|---|---|
| `deliverables/PrashantGroup_IndiaITME2026_Invitation_MOBILE_1080x1920.mp4` | 9:16, 1080×1920, 30 fps, H.264 High + AAC, fast-start | WhatsApp, Instagram/LinkedIn stories, phones |
| `deliverables/PrashantGroup_IndiaITME2026_Invitation_WEB_1920x1080.mp4` | 16:9, 1920×1080, 30 fps, H.264 High + AAC, fast-start | Website embeds, desktop, presentations |

Both files run 50.4 s. That is slightly over the 45-second target, because of the extra dwell time on the hall/booth scene and the final hold.

## Event details used (from the source video)
- **India ITME 2026**, with the full name *India International Textile Machinery Exhibition* shown as a subtitle the first time the event appears
- **4–9 December 2026**, **Hall 14**, **Booths B8, B10, B12**
- Contact details: sales@prashantgroup.com · www.prashantgroup.com

## Scene list (same story in both formats, each layout rebuilt for its canvas)

| # | Time (s) | Scene | Mobile layout | Web layout |
|---|---|---|---|---|
| 1 | 0.0–4.6 | **Branded invitation**: Prashant Group logo on a white plate (opens outward like fabric unrolling from a beam), yellow rule, "Invites you", tagline *Excellence in Weaving Preparatory* | Centred stack | Centred stack, wider plate |
| 2 | 3.9–10.0 | **Exhibition introduction** (event branding): India ITME logo card, "Meet us at", **India ITME 2026**, full exhibition name, date pill | Logo card above the text | Logo card on the left, text on the right |
| 3 | 9.3–15.8 | **Hall & booths** (longest content dwell, about 6 s): small ITME chip, "Find us at", **HALL 14**, booth tiles B8 / B10 / B12, dates | Stacked: Hall 14 above, three booth tiles in a row | Hall 14 on the left, gold divider, booth tiles on the right |
| 4 | 15.2–20.8 | **Heritage**: "Since 1975", *Five decades of textile engineering*, factory photo (wipe reveal plus slow push-in), company statement | Headline, photo, then text | Large photo on the left, text on the right |
| 5 | 20.2–25.2 | **Key figures** (numbers count up): 50+ years · 60+ countries · 7,800+ installations · 100,000+ m² | 2 × 2 tiles | 4 tiles in a row |
| 6 | 24.6–31.2 | **Group companies**: Prashant Gamatex (Italy), Prashant West Point (USA), Prashant Ferber (Italy), each with logo, flags, capabilities and partnership line | 3 horizontal cards, stacked | 3 vertical cards, side by side |
| 7 | 30.6–35.8 | **Applications**: one-stop solution for weaving preparatory, 8 segments with photos (staggered cascade) | 2 × 4 grid | 4 × 2 grid |
| 8 | 35.2–40.8 | **Global presence in over 60 countries**: world map (wipe reveal, pulse on India) and the three regions | Map with regions below | Map on the left, regions on the right |
| 9 | 40.2–44.0 | **Regional offices in India** (split out of the global scene): 8 city chips | 2 × 4 | 4 × 2 |
| 10 | 43.4–50.4 | **Closing invitation** (event branding): Prashant logo, **"We look forward to welcoming you"**, event card (ITME logo, India ITME 2026, 4–9 December 2026, Hall 14 · Booths B8, B10, B12), email, website, ISO/CE marks. Fully readable for about 5 s at the end. | Stacked | Invitation and contact on the left, event card on the right |

**Transitions.** Scenes crossfade over 0.7 s with a gentle sideways drift. Content lifts out before each change, and a fine band of white and yellow "threads" sweeps across the frame like a passing shuttle. Navy scenes carry a faint woven texture. Light scenes carry a slow, flowing bundle of warp-thread lines.

**Animation order.** Every scene builds in the same order: eyebrow and headline, then the yellow rule, then visuals, then details. Elements are staggered 0.1–0.35 s apart and use ease-out-quint motion.

## Brand system
- Navy `#0B2A5B` / deep navy `#061B3E`, logo blue `#1F6DB0`, yellow `#F4C21A`, white / paper `#F6F8FB`
- Inter Display (headlines, ExtraBold) and Inter (text)
- The Prashant Group logo always sits on white or very light grounds, in its original proportions.
- The India ITME logo appears only in the event scenes (2, 3 and 10), always on white, in its original colours and proportions.

## Copy edits (meaning unchanged)
- "India-ITME 2026" / "India-ITME-2026" → **India ITME 2026**; "4th–9th December" → **4–9 December 2026**; "Hall:14 – B8,B10,B12" → **Hall 14 · Booths B8, B10, B12**
- "Looking forward to welcome you at" → **We look forward to welcoming you**
- "One Stop Solution" → **One-stop solution**; "Global Presence over 60 countries" → **Global presence in over 60 countries**
- "Sampling Solution" → "sampling solutions"; "Technical Textile" → "Technical Textiles"; "Yarn-Dyed-Shirting" → "Yarn-Dyed Shirting"; "Home furnishing/Silk" → "Home Furnishing / Silk"; sentence case applied consistently
- New short labels (eyebrows) only describe what is already shown: "Meet us at", "Find us at", "Our journey in numbers", "Global partnerships", "Applications", "Close to our customers"

## Audio
The original music track is kept. It is extended to 50.4 s by looping a beat-aligned section from the second half with 0.4 s crossfades, normalised to −16 LUFS (−1.5 dBTP), with a 0.4 s fade-in and a 3.2 s fade-out. **Licensing:** the track came with the supplied video, and its licence could not be checked here. Please confirm that Prashant Group holds the rights, or swap in a licensed track (re-run `build.sh` after replacing the audio step).

## Known limitations
- Every logo, photo and the map is taken from the supplied 720×1280 video, so they were denoised (temporal averaging across frames) and upscaled 2–3×. They are clean but slightly soft at full screen. The ISO badge in particular was illegibly small in the source. For a sharper master, drop official files (vector logos, original photos) into `project/assets/` with the same names and re-run `build.sh`.
- No venue is stated, because the source video does not give one.

## Editable project
`project/` holds the whole edit as code. Anyone can change text, timing or layout and re-render.
- `invitation.html`: all scenes, styles and the animation timeline (`data-start` / `data-end` per scene, `data-in="type,delay,duration"` per element). Open `invitation.html?f=m&play` or `?f=w&play` in Chrome for a live preview, or add `&t=12` to see one moment.
- `render.js`: frame-accurate capture with Playwright, piped to ffmpeg
- `build.sh`: builds the soundtrack and renders and encodes both MP4s
- `extract_assets.py`: how the logos and photos were pulled from the source video
- `assets/`: extracted and cleaned assets; `source/original.mp4`: the supplied video
