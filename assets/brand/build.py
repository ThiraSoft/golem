# Regenerates the brand images from JetBrains Mono (github.com/JetBrains/JetBrainsMono).
# uv run --with fonttools python assets/brand/build.py <dir with JetBrainsMono-*.ttf> assets/brand
# then rsvg-convert -w <width> <name>.svg -o <name>.png
import sys, os
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

FONTDIR, OUT = sys.argv[1], sys.argv[2]
BG, CARD, EDGE = "#11151B", "#1A2029", "#2A3340"
TEAL, WHITE, SOFT, MUTED, GRAY = "#5CCFD0", "#FFFFFF", "#C8D2DE", "#7E8A9C", "#4A5568"
fonts = {w: TTFont(os.path.join(FONTDIR, f"JetBrainsMono-{w}.ttf")) for w in ("ExtraBold", "Medium", "Regular")}

def text(s, x, y, size, weight, color, spacing=0.0):
    f = fonts[weight]; gs = f.getGlyphSet(); cmap = f.getBestCmap(); k = size / f["head"].unitsPerEm
    out, cx = [], x
    for ch in s:
        g = cmap[ord(ch)]; pen = SVGPathPen(gs)
        gs[g].draw(TransformPen(pen, (k, 0, 0, -k, cx, y)))
        if pen.getCommands(): out.append(pen.getCommands())
        cx += gs[g].width * k + spacing
    return f'<path fill="{color}" d="{" ".join(out)}"/>', cx

def mark(x, y, s, p, flat=None):
    """The stone block with the aleph, from assets/logo.svg, s pixels wide."""
    if flat:
        return (f'<g transform="translate({x} {y}) scale({s/140}) translate(-24 -30)">'
                f'<rect x="24" y="30" width="140" height="140" rx="18" fill="{flat}"/></g>')
    return (f'<g transform="translate({x} {y}) scale({s/140}) translate(-24 -30)">'
            f'<defs><linearGradient id="{p}seam" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#7fe3d4"/><stop offset="100%" stop-color="#3aa8c9"/></linearGradient>'
            f'<linearGradient id="{p}stone" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#4a5568"/><stop offset="100%" stop-color="#2d3440"/></linearGradient></defs>'
            f'<rect x="24" y="30" width="140" height="140" rx="18" fill="url(#{p}stone)"/>'
            f'<rect x="24" y="30" width="140" height="140" rx="18" fill="none" stroke="#7fe3d4" stroke-opacity="0.35" stroke-width="2"/>'
            f'<g stroke="url(#{p}seam)" stroke-width="9" stroke-linecap="round" fill="none">'
            f'<path d="M66 74 L122 126"/><path d="M122 74 C104 78 96 92 92 104"/><path d="M66 126 C84 122 92 108 96 96"/></g></g>')

def svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">'
            f'<title>{title}</title>{body}</svg>\n')

def save(name, content):
    open(os.path.join(OUT, name), "w").write(content)

def word(x, y, size):
    return text("golem", x, y, size, "ExtraBold", WHITE, spacing=-size * 0.02)[0]

TITLE = "golem: local AI in one static Go binary"

# avatar 400x400
save("avatar.svg", svg(400, 400, f'<rect width="400" height="400" fill="{BG}"/>' + mark(100, 100, 200, "a"), "golem"))

# README banner 1280x320
b = f'<rect width="1280" height="320" rx="24" fill="{BG}"/>' + mark(90, 70, 180, "a")
b += word(330, 168, 110)
b += text("Local AI in one static Go binary.", 336, 228, 30, "Medium", SOFT)[0]
b += text("LLMs · vision · speech · images · no Python, no cgo", 336, 268, 21, "Regular", TEAL)[0]
save("banner.svg", svg(1280, 320, b, TITLE))

# LinkedIn cover 1584x396
W, H = 1584, 396
c = f'<rect width="{W}" height="{H}" fill="{BG}"/>'
c += f'<g opacity="0.05">{mark(-140, -120, 460, "", WHITE)}</g>'
c += mark(476, 60, 128, "a")
c += word(636, 152, 88)
c += text("Local AI in one", 476, 236, 27, "Medium", SOFT)[0]
c += text("static Go binary.", 476, 272, 27, "Medium", SOFT)[0]
c += text("no Python, no cgo · CPU or Vulkan", 476, 318, 20, "Regular", TEAL)[0]
cx, cy, cw, ch = 1040, 50, 492, 296
c += f'<rect x="{cx}" y="{cy}" width="{cw}" height="{ch}" rx="16" fill="{CARD}" stroke="{EDGE}"/>'
rows = [("Text", "Gemma 4 · Qwen3 · Bonsai 2"), ("Vision", "Gemma 4 · Qwen3.8"),
        ("Speech", "Kyutai STT · Pocket TTS"), ("Images", "Krea 2 · FLUX 3 Klein"),
        ("Music", "YuE2"), ("Serving", "OpenAI-compatible API")]
y = cy + 58
for k, v in rows:
    c += text(k, cx + 30, y, 15, "Medium", TEAL)[0]
    c += text(v, cx + 140, y, 15, "Regular", SOFT)[0]
    y += 38
save("linkedin-cover.svg", svg(W, H, c, TITLE))

# thumbnail 1200x627, LinkedIn Featured and link previews
W, H = 1200, 627
t = f'<rect width="{W}" height="{H}" fill="{BG}"/>'
t += f'<g opacity="0.05">{mark(990, 470, 420, "", WHITE)}</g>'
t += mark(170, 190, 230, "a")
t += word(476, 268, 120)
t += text("Local AI in one", 482, 334, 32, "Medium", SOFT)[0]
t += text("static Go binary.", 482, 376, 32, "Medium", SOFT)[0]
t += text("CPU or Vulkan · no Python, no cgo", 482, 446, 24, "Regular", TEAL)[0]
save("thumbnail.svg", svg(W, H, t, TITLE))
print("ok")
