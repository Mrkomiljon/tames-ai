"""Make a pandoc reference.docx with manuscript styling: Times New Roman, 12 pt, 1.5 spacing,
black bold headings, captions 10 pt, tables 9 pt."""
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.shared import Pt, RGBColor

HERE = Path(__file__).resolve().parent
ref = HERE / "reference.docx"
subprocess.run(["pandoc", "-o", str(ref), "--print-default-data-file", "reference.docx"], check=True)

d = Document(str(ref))

class _St(dict):
    pass

st = _St((s.name, s) for s in d.styles)


def font(style, size, bold=None, italic=None, color=RGBColor(0, 0, 0), name="Times New Roman"):
    f = style.font
    f.name = name
    f.size = Pt(size)
    f.color.rgb = color
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}rFonts')
    if rfonts is None:
        from docx.oxml import OxmlElement
        rfonts = OxmlElement('w:rFonts')
        rpr.append(rfonts)
    for k in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}' + k.split(":")[1], name)


def spacing(style, line=1.5, before=0, after=6):
    pf = style.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)


for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
    if name in st:
        font(st[name], 12); spacing(st[name])
font(st["Title"], 16, bold=True); spacing(st["Title"], 1.2, 0, 12)
for name, size in (("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 12)):
    font(st[name], size, bold=True, italic=(name == "Heading 3")); spacing(st[name], 1.2, 12, 6)
for name in ("Caption", "Image Caption", "Table Caption"):
    if name in st:
        font(st[name], 10, italic=False); spacing(st[name], 1.15, 4, 10)
if "Table" in st:
    font(st["Table"], 9); spacing(st["Table"], 1.0, 0, 0)
if "Source Code" in st:
    font(st["Source Code"], 9, name="Consolas")
d.save(str(ref))
print("reference.docx written")
