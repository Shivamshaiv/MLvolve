#!/usr/bin/env bash
# Builds both deliverables. Requires: node + playwright (chromium), ffmpeg, the original video as SRC.
set -euo pipefail
cd "$(dirname "$0")"
SRC=${SRC:-source/original.mp4}; OUT=${OUT:-../deliverables}; TMP=${TMP:-build}; mkdir -p "$OUT" "$TMP"
DUR=50.4
# Soundtrack: original music extended by looping its last 10.284 s phrase (loop length found by
# envelope cross-correlation, so beats line up), short overlaps at matched points, loudness-normalised.
ffmpeg -y -v error -i "$SRC" -vn -c:a pcm_s16le "$TMP/src_audio.wav"
D=0.1; L=10.284; LS=$(echo "28.0-$L-$D"|bc)
ffmpeg -y -v error -t 28.0 -i "$TMP/src_audio.wav" \
  -ss $LS -t $(echo "$L+$D"|bc) -i "$TMP/src_audio.wav" -ss $LS -t $(echo "$L+$D"|bc) -i "$TMP/src_audio.wav" \
  -ss $LS -t $(echo "$L+$D"|bc) -i "$TMP/src_audio.wav" \
  -filter_complex "[0:a][1:a]acrossfade=d=$D[ab];[ab][2:a]acrossfade=d=$D[abc0];[abc0][3:a]acrossfade=d=$D[abc];\
[abc]atrim=0:${DUR},afade=t=in:d=0.4,afade=t=out:st=$(echo "$DUR-3.2"|bc):d=3.2,\
aresample=48000[out]" -map "[out]" -c:a pcm_s16le "$TMP/music_raw.wav"
# two-pass loudness normalisation to -16 LUFS / -1.5 dBTP
M=$(ffmpeg -hide_banner -i "$TMP/music_raw.wav" -af loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
j(){ echo "$M" | python3 -c "import sys,json;print(json.load(sys.stdin)['$1'])"; }
ffmpeg -y -v error -i "$TMP/music_raw.wav" -af "loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=$(j input_i):measured_TP=$(j input_tp):measured_LRA=$(j input_lra):measured_thresh=$(j input_thresh):offset=$(j target_offset):linear=true,aresample=48000" -c:a pcm_s16le "$TMP/music.wav"
node render.js m "$TMP/mobile_video.mp4" &
node render.js w "$TMP/web_video.mp4" &
wait
enc(){ # $1 video $2 out $3 level
  ffmpeg -y -v error -i "$1" -i "$TMP/music.wav" -map 0:v -map 1:a \
    -c:v libx264 -preset slow -crf 18 -profile:v high -level:v "$3" -pix_fmt yuv420p -tune stillimage \
    -x264-params "keyint=60:min-keyint=30" -r 30 -c:a aac -b:a 192k -ar 48000 -shortest \
    -movflags +faststart -metadata title="Prashant Group – India ITME 2026 Invitation" "$2"; }
enc "$TMP/mobile_video.mp4" "$OUT/PrashantGroup_IndiaITME2026_Invitation_MOBILE_1080x1920.mp4" 4.2
enc "$TMP/web_video.mp4"    "$OUT/PrashantGroup_IndiaITME2026_Invitation_WEB_1920x1080.mp4" 4.2
