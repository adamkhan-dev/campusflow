"""Generate the 12 multimodal demo files (3 per CS subject) in demo_files/."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT_DIR = Path(__file__).resolve().parent
DEMO_DIR = ROOT_DIR / "demo_files"


def _load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.ImageFont:
    if mono:
        candidates = [
            r"C:\Windows\Fonts\consolab.ttf" if bold else r"C:\Windows\Fonts\consola.ttf",
            r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        ]
    else:
        candidates = [
            r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\consola.ttf",
        ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _synthesize_wav(out_path: Path, spoken_script: str) -> None:
    """Use Windows SAPI offline TTS to synthesize realistic lecture audio WAV."""
    escaped_path = str(out_path.resolve()).replace("'", "''")
    escaped_text = spoken_script.replace("'", "''")
    ps_script = (
        "Add-Type -AssemblyName System.Speech; "
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$synth.Rate = 0; "
        f"$synth.SetOutputToWaveFile('{escaped_path}'); "
        f"$synth.Speak('{escaped_text}'); "
        "$synth.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
        check=True,
    )


def _apply_smudge(
    img: Image.Image,
    box: tuple[int, int, int, int],
    radius: int = 8,
    tint: tuple[int, int, int] = (205, 208, 212),
) -> None:
    """Apply a realistic dry-erase whiteboard smudge over a bounding box."""
    x0, y0, x1, y1 = box
    crop = img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(radius=radius))
    img.paste(crop, (x0, y0))
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle([x0 + 4, y0 + 4, x1 - 4, y1 - 4], radius=10, fill=(*tint, 130))
    od.line([(x0 + 10, (y0 + y1) // 2), (x1 - 12, (y0 + y1) // 2 - 4)], fill=(120, 124, 132, 165), width=7)
    od.line([(x0 + 16, (y0 + y1) // 2 + 8), (x1 - 18, (y0 + y1) // 2 + 4)], fill=(140, 144, 150, 145), width=5)
    blended = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img.paste(blended)


def _make_slide_page(
    course_label: str,
    page_num: int,
    total_pages: int,
    slide_title: str,
    sections: list[tuple[str, list[str]]],
    footer_note: str,
) -> Image.Image:
    W, H = 1400, 790
    img = Image.new("RGB", (W, H), (248, 250, 252))
    d = ImageDraw.Draw(img)

    f_header = _load_font(22, bold=True)
    f_title = _load_font(38, bold=True)
    f_sec = _load_font(25, bold=True)
    f_body = _load_font(21)
    f_small = _load_font(17)

    # Top banner
    d.rectangle([0, 0, W, 96], fill=(17, 24, 39))
    d.text((54, 32), f"COMSATS University Islamabad  |  {course_label}", fill=(148, 163, 184), font=f_header)
    d.text((W - 190, 32), f"Slide {page_num} of {total_pages}", fill=(56, 189, 248), font=f_header)

    d.text((54, 128), slide_title, fill=(15, 23, 42), font=f_title)
    d.line([(54, 184), (W - 54, 184)], fill=(203, 213, 225), width=3)

    y = 212
    for heading, bullets in sections:
        d.rounded_rectangle([54, y, W - 54, y + 48 + len(bullets) * 38 + 20], radius=12, fill=(255, 255, 255), outline=(203, 213, 225), width=2)
        d.text((78, y + 14), heading, fill=(15, 118, 110), font=f_sec)
        by = y + 54
        for bullet in bullets:
            d.text((92, by), f"-  {bullet}", fill=(30, 41, 59), font=f_body)
            by += 38
        y += 48 + len(bullets) * 38 + 38

    d.line([(54, H - 56), (W - 54, H - 56)], fill=(226, 232, 240), width=2)
    d.text((54, H - 40), footer_note, fill=(100, 116, 139), font=f_small)
    return img


def _save_two_page_pdf(out_path: Path, page1: Image.Image, page2: Image.Image) -> None:
    page1.save(out_path, "PDF", resolution=150.0, save_all=True, append_images=[page2])


# ---------------------------------------------------------------------------
# 1. CSC322 — Operating Systems (FCFS CPU Scheduling)
# ---------------------------------------------------------------------------
def _draw_csc322_whiteboard(out_path: Path) -> None:
    W, H = 1280, 760
    wb = Image.new("RGB", (W, H), (242, 244, 240))
    d = ImageDraw.Draw(wb)

    f_title = _load_font(34, bold=True)
    f_h2 = _load_font(25, bold=True)
    f_body = _load_font(22)
    f_small = _load_font(18)

    d.rectangle([14, 14, W - 14, H - 14], outline=(156, 163, 175), width=5)
    d.text((48, 36), "CSC322 - Operating Systems | FCFS CPU Scheduling", fill=(20, 44, 92), font=f_title)
    d.line([(48, 84), (920, 84)], fill=(20, 44, 92), width=3)

    d.text((52, 112), "Process Table:", fill=(24, 24, 28), font=f_h2)
    cols_x = [52, 190, 380, 580]
    rows_y = [156, 208, 260, 312, 364]
    for y in rows_y:
        d.line([(52, y), (580, y)], fill=(70, 78, 88), width=2)
    for x in cols_x:
        d.line([(x, 156), (x, 364)], fill=(70, 78, 88), width=2)

    for idx, h in enumerate(["Process", "Arrival (AT)", "Burst (BT)"]):
        d.text((cols_x[idx] + 16, 168), h, fill=(20, 44, 92), font=f_body)

    rows = [("P1", "0 ms", "4 ms"), ("P2", "1 ms", "3 ms"), ("P3", "2 ms", "2 ms")]
    for r_i, (p, at, bt) in enumerate(rows):
        y = rows_y[r_i + 1] + 14
        d.text((cols_x[0] + 28, y), p, fill=(30, 30, 35), font=f_body)
        d.text((cols_x[1] + 45, y), at, fill=(30, 30, 35), font=f_body)
        d.text((cols_x[2] + 48, y), bt, fill=(30, 30, 35), font=f_body)

    # Smudge ONLY P2's Burst cell cleanly (row 1 -> rows_y[2]..rows_y[3])
    _apply_smudge(wb, (cols_x[2] + 6, rows_y[2] + 4, cols_x[3] - 6, rows_y[3] - 4), radius=9)
    d.text((290, 374), "^ Smudged P2 Burst Time! Deduce from Gantt [4..7]", fill=(180, 83, 9), font=f_small)

    # Key formulas box
    d.rounded_rectangle([630, 112, 1225, 364], radius=10, outline=(20, 44, 92), width=2, fill=(248, 250, 247))
    d.text((654, 132), "Key Formulas & Policy", fill=(20, 44, 92), font=f_h2)
    d.text((654, 178), "Policy: Non-preemptive FIFO Ready Queue", fill=(30, 30, 35), font=f_body)
    d.text((654, 218), "Turnaround: TAT = CT - AT", fill=(15, 118, 110), font=f_body)
    d.text((654, 258), "Waiting:    WT  = TAT - BT", fill=(15, 118, 110), font=f_body)
    d.text((654, 304), "Convoy Effect: short bursts stall behind long CPU job", fill=(153, 27, 27), font=f_small)

    # Gantt chart
    d.text((52, 412), "Worked Gantt Chart Timeline (ms):", fill=(24, 24, 28), font=f_h2)
    gx, gy, gh = 56, 464, 84
    segments = [("P1", 0, 4, 360), ("P2", 4, 7, 280), ("P3", 7, 9, 210)]
    cur_x = gx
    for label, start_t, end_t, seg_w in segments:
        d.rectangle([cur_x, gy, cur_x + seg_w, gy + gh], outline=(20, 44, 92), width=3, fill=(232, 239, 248))
        d.text((cur_x + seg_w // 2 - 18, gy + 26), label, fill=(20, 44, 92), font=f_h2)
        d.text((cur_x - 6, gy + gh + 10), f"{start_t}", fill=(30, 30, 35), font=f_body)
        cur_x += seg_w
    d.text((cur_x - 8, gy + gh + 10), "9", fill=(30, 30, 35), font=f_body)

    d.text((52, 608), "CT: P1=4, P2=7, P3=9   |   TAT: P1=4, P2=6, P3=7 (Avg 5.67 ms)   |   WT: P1=0, P2=3, P3=5 (Avg 2.67 ms)", fill=(30, 41, 59), font=f_body)
    d.text((52, 658), "Class Note: jo process pehle ready queue mein aaye gi, CPU pehle usay mile ga.", fill=(71, 85, 105), font=f_small)
    wb.save(out_path)


# ---------------------------------------------------------------------------
# 2. CSC325 — Computer Organization & Assembly (IEEE 754 Single-Precision)
# ---------------------------------------------------------------------------
def _draw_csc325_whiteboard(out_path: Path) -> None:
    W, H = 1280, 760
    wb = Image.new("RGB", (W, H), (242, 244, 240))
    d = ImageDraw.Draw(wb)

    f_title = _load_font(34, bold=True)
    f_h2 = _load_font(25, bold=True)
    f_body = _load_font(22)
    f_mono = _load_font(22, bold=True, mono=True)
    f_small = _load_font(18)

    d.rectangle([14, 14, W - 14, H - 14], outline=(156, 163, 175), width=5)
    d.text((48, 36), "CSC325 - COAL | IEEE 754 Single-Precision (-12.625)", fill=(20, 44, 92), font=f_title)
    d.line([(48, 84), (990, 84)], fill=(20, 44, 92), width=3)

    d.text((52, 114), "Step 1: Binary Conversion & Normalization", fill=(24, 24, 28), font=f_h2)
    d.text((64, 158), "Decimal = -12.625   =>   Sign Bit S = 1 (Negative)", fill=(30, 30, 35), font=f_body)
    d.text((64, 196), "12 = 1100_2   |   0.625 = 0.5 + 0.125 = .101_2   =>   1100.101_2", fill=(30, 30, 35), font=f_body)
    d.text((64, 234), "Normalized:  -1.100101 x 2^3    (Unbiased E = 3)", fill=(15, 118, 110), font=f_body)

    # Step 2 formula card
    d.rounded_rectangle([52, 286, 1225, 422], radius=10, outline=(20, 44, 92), width=2, fill=(248, 250, 247))
    d.text((76, 304), "Step 2: Bias Formula & Mantissa Packing", fill=(20, 44, 92), font=f_h2)
    d.text((76, 344), "Biased Exponent (8 bits) = E + Bias = 3 + 127 = 130  =>  10000010", fill=(30, 30, 35), font=f_body)
    d.text((76, 382), "23-bit Mantissa (Fraction): 10010100000000000000000   (implicit leading 1 dropped)", fill=(30, 30, 35), font=f_body)

    # 32-bit layout grid
    d.text((52, 448), "32-bit IEEE 754 Layout: [ 1 Sign | 8 Exponent | 23 Fraction ]", fill=(24, 24, 28), font=f_h2)
    bx, by, bh = 54, 494, 72
    # Sign cell
    d.rectangle([bx, by, bx + 90, by + bh], outline=(20, 44, 92), width=3, fill=(254, 226, 226))
    d.text((bx + 22, by + 18), "S=1", fill=(153, 27, 27), font=f_mono)
    # Exponent cell (smudged!)
    exp_x0, exp_x1 = bx + 98, bx + 420
    d.rectangle([exp_x0, by, exp_x1, by + bh], outline=(180, 83, 9), width=3, fill=(254, 243, 199))
    d.text((exp_x0 + 30, by + 18), "10000010 (130)", fill=(120, 53, 15), font=f_mono)
    _apply_smudge(wb, (exp_x0 + 6, by + 6, exp_x1 - 6, by + bh - 6), radius=9)
    # Fraction cell
    frac_x0 = bx + 428
    d.rectangle([frac_x0, by, 1222, by + bh], outline=(20, 44, 92), width=3, fill=(232, 239, 248))
    d.text((frac_x0 + 26, by + 18), "10010100000000000000000 (23 bits)", fill=(20, 44, 92), font=f_mono)

    d.text((exp_x0 - 10, by + bh + 12), "^ Smudged 8-bit Exponent (check 127 + 3 = 130)", fill=(180, 83, 9), font=f_small)
    d.text((54, 624), "Final 32-bit Hexadecimal Word:   0xC14A0000", fill=(15, 118, 110), font=f_h2)
    d.text((54, 670), "Class Note: mantissa mein decimal ke baad sirf fraction store hota hai, leading 1 hidden rehta hai.", fill=(71, 85, 105), font=f_small)
    wb.save(out_path)


# ---------------------------------------------------------------------------
# 3. AIC354 — Machine Learning Fundamentals (Naive Bayes & Laplace Smoothing)
# ---------------------------------------------------------------------------
def _draw_aic354_whiteboard(out_path: Path) -> None:
    W, H = 1280, 760
    wb = Image.new("RGB", (W, H), (242, 244, 240))
    d = ImageDraw.Draw(wb)

    f_title = _load_font(34, bold=True)
    f_h2 = _load_font(25, bold=True)
    f_body = _load_font(22)
    f_small = _load_font(18)

    d.rectangle([14, 14, W - 14, H - 14], outline=(156, 163, 175), width=5)
    d.text((48, 36), "AIC354 - Machine Learning | Naive Bayes & Laplace Smoothing", fill=(20, 44, 92), font=f_title)
    d.line([(48, 84), (1050, 84)], fill=(20, 44, 92), width=3)

    d.text((52, 114), "1. Bayes Decision Rule & Conditional Independence:", fill=(24, 24, 28), font=f_h2)
    d.text((68, 156), "y_hat = argmax_y  [ log P(y) + Sum_{i=1..n} log P(x_i | y) ]   (log-space avoids underflow)", fill=(15, 118, 110), font=f_body)
    d.text((68, 194), "Naive Assumption: features x_i are conditionally independent given class y.", fill=(30, 30, 35), font=f_body)

    # Laplace Smoothing card
    d.text((52, 250), "2. Zero-Frequency Fix (Add-1 / Laplace Smoothing):", fill=(24, 24, 28), font=f_h2)
    d.rounded_rectangle([52, 294, 1222, 434], radius=10, outline=(20, 44, 92), width=2, fill=(248, 250, 247))
    d.text((86, 326), "P(x_i | y)  =  ( N_{y, i} + 1 )  /", fill=(20, 44, 92), font=f_title)
    # Denominator cell smudged!
    denom_box = (552, 316, 828, 376)
    d.text((566, 326), "( N_y + |V| )", fill=(180, 83, 9), font=f_title)
    _apply_smudge(wb, denom_box, radius=9)
    d.text((846, 336), "<-- Smudged denominator!", fill=(180, 83, 9), font=f_small)
    d.text((86, 390), "where N_{y,i} = count of word i in class y,  N_y = total words in class y,  |V| = vocabulary size", fill=(51, 65, 85), font=f_small)

    # Worked Spam vs Ham example
    d.text((52, 466), "3. Worked Spam Filter Example (|V| = 1000, N_Spam = 200):", fill=(24, 24, 28), font=f_h2)
    d.text((68, 512), "- Seen word 'prize' (N = 5):     P('prize' | Spam)  = (5 + 1) / (200 + 1000) = 6 / 1200  = 0.005", fill=(30, 30, 35), font=f_body)
    d.text((68, 554), "- Unseen word 'urgent' (N = 0):  P('urgent' | Spam) = (0 + 1) / (200 + 1000) = 1 / 1200  != 0", fill=(15, 118, 110), font=f_body)
    d.text((68, 600), "- Without +1 smoothing: zero probability product collapses the entire posterior to 0!", fill=(153, 27, 27), font=f_body)
    d.text((52, 666), "Class Note: agar ek naya lafz training mein na aaye toh zero product se bachne ke liye |V| add karte hain.", fill=(71, 85, 105), font=f_small)
    wb.save(out_path)


# ---------------------------------------------------------------------------
# 4. CSC301 — Design & Analysis of Algorithms (Binary Search & Recurrence)
# ---------------------------------------------------------------------------
def _draw_csc301_whiteboard(out_path: Path) -> None:
    W, H = 1280, 760
    wb = Image.new("RGB", (W, H), (242, 244, 240))
    d = ImageDraw.Draw(wb)

    f_title = _load_font(34, bold=True)
    f_h2 = _load_font(25, bold=True)
    f_body = _load_font(22)
    f_mono = _load_font(23, bold=True, mono=True)
    f_small = _load_font(18)

    d.rectangle([14, 14, W - 14, H - 14], outline=(156, 163, 175), width=5)
    d.text((48, 36), "CSC301 - Algorithms | Binary Search & Recurrence Complexity", fill=(20, 44, 92), font=f_title)
    d.line([(48, 84), (1050, 84)], fill=(20, 44, 92), width=3)

    d.text((52, 114), "1. Precondition & Overflow-Safe Midpoint:", fill=(24, 24, 28), font=f_h2)
    d.text((68, 154), "Precondition: input array A[0..n-1] MUST be sorted.", fill=(30, 30, 35), font=f_body)
    d.rounded_rectangle([52, 198, 1222, 314], radius=10, outline=(20, 44, 92), width=2, fill=(248, 250, 247))
    d.text((76, 224), "mid = ", fill=(20, 44, 92), font=f_mono)
    mid_box = (165, 214, 595, 266)
    d.text((178, 224), "low + (high - low) // 2", fill=(180, 83, 9), font=f_mono)
    _apply_smudge(wb, mid_box, radius=9)
    d.text((612, 226), "<-- Smudged! Safe vs (low + high) // 2 overflow", fill=(180, 83, 9), font=f_small)
    d.text((76, 274), "Avoids signed 32-bit integer overflow when low + high > INT_MAX (2^31 - 1).", fill=(51, 65, 85), font=f_small)

    # Recurrence & Master Theorem card
    d.text((52, 344), "2. Recurrence Relation & Asymptotic Complexity:", fill=(24, 24, 28), font=f_h2)
    d.text((68, 386), "T(n) = T(n / 2) + c,   T(1) = c   =>   a = 1,  b = 2,  f(n) = Theta(1)", fill=(15, 118, 110), font=f_body)
    d.text((68, 424), "n^(log_2 1) = n^0 = 1   =>   Master Theorem Case 2 gives   T(n) = Theta(log n)", fill=(30, 30, 35), font=f_body)

    # Array trace
    d.text((52, 478), "3. Worked Trace: A = [3, 8, 14, 21, 29, 37, 45], target = 37", fill=(24, 24, 28), font=f_h2)
    arr = [3, 8, 14, 21, 29, 37, 45]
    ax, ay, cw, ch = 64, 526, 130, 62
    for idx, val in enumerate(arr):
        fill_col = (209, 250, 229) if val == 37 else (232, 239, 248)
        d.rectangle([ax + idx * cw, ay, ax + (idx + 1) * cw, ay + ch], outline=(20, 44, 92), width=2, fill=fill_col)
        d.text((ax + idx * cw + 46, ay + 18), str(val), fill=(20, 44, 92), font=f_body)
        d.text((ax + idx * cw + 52, ay - 22), f"[{idx}]", fill=(100, 116, 139), font=f_small)

    d.text((64, 606), "Step 1: low=0, high=6 => mid=3 (21 < 37) => low=4    |    Step 2: low=4, high=6 => mid=5 (37 == 37, Found!)", fill=(15, 118, 110), font=f_body)
    d.text((64, 658), "Class Note: har step mein search space ko half kar dete hain, is liye complexity O(log n) hoti hai.", fill=(71, 85, 105), font=f_small)
    wb.save(out_path)


# ---------------------------------------------------------------------------
# Specification & Generator Runner
# ---------------------------------------------------------------------------
DEMO_SPECS: list[dict[str, object]] = [
    {
        "code": "CSC322",
        "prefix": "csc322_os",
        "audio_script": (
            "Assalam-o-Alaikum class, welcome to CSC322 Operating Systems. Today we cover First-Come, First-Served, or FCFS CPU Scheduling. "
            "FCFS uses a non-preemptive FIFO ready queue: jo process pehle ready queue mein aaye gi, CPU pehle usay mile ga. "
            "In our board example, P1 arrives at 0 milliseconds with burst 4. P2 arrives at 1 millisecond, and on the Gantt chart P2 runs from 4 to 7, so its smudged burst time is 3 milliseconds. "
            "P3 arrives at 2 with burst 2. Turnaround Time is Completion minus Arrival: 4, 6, and 7. Waiting Time is Turnaround minus Burst: 0, 3, and 5, giving Average Waiting Time 2.67 milliseconds. "
            "Remember the Convoy Effect: when one long CPU burst runs ahead, short jobs pile up behind it."
        ),
        "draw_wb": _draw_csc322_whiteboard,
        "slide_pages": (
            (
                "CSC322 — Operating Systems",
                "First-Come, First-Served (FCFS) CPU Scheduling",
                [
                    (
                        "Core Scheduling Policy",
                        [
                            "Non-preemptive execution backed by a First-In, First-Out (FIFO) ready queue.",
                            "Running process retains the CPU until its CPU burst completes or blocks on I/O.",
                            "Zero starvation, but average waiting time depends heavily on arrival order.",
                        ],
                    ),
                    (
                        "The Convoy Effect",
                        [
                            "One long CPU-bound process arrives first and stalls many short processes behind it.",
                            "Reduces overall CPU and I/O device utilization compared to shorter-first policies.",
                        ],
                    ),
                ],
                "CampusFlow Academic Demo  |  CSC322 Lecture 1",
            ),
            (
                "CSC322 — Operating Systems",
                "Turnaround, Waiting Time Formulas & Worked Example",
                [
                    (
                        "Scheduling Equations",
                        [
                            "Turnaround Time: TAT = Completion Time (CT) - Arrival Time (AT).",
                            "Waiting Time: WT = Turnaround Time (TAT) - Burst Time (BT).",
                        ],
                    ),
                    (
                        "Worked Gantt Chart: [0 | P1 | 4 | P2 | 7 | P3 | 9]",
                        [
                            "P1 (AT=0, BT=4): CT=4 ms, TAT=4 ms, WT=0 ms.",
                            "P2 (AT=1, BT=3 inferred from [4..7]): CT=7 ms, TAT=6 ms, WT=3 ms.",
                            "P3 (AT=2, BT=2): CT=9 ms, TAT=7 ms, WT=5 ms  ->  Avg WT = 2.67 ms, Avg TAT = 5.67 ms.",
                        ],
                    ),
                ],
                "CampusFlow Academic Demo  |  CSC322 Lecture 1",
            ),
        ),
    },
    {
        "code": "CSC325",
        "prefix": "csc325_coal",
        "audio_script": (
            "Assalam-o-Alaikum class, welcome to CSC325 Computer Organization and Assembly Language. Today we encode minus 12.625 in IEEE 754 Single-Precision 32-bit Floating Point. "
            "First, Sign bit S equals 1 because the number is negative. Twelve in binary is 1100, and 0.625 is .101, giving 1100.101. "
            "Normalizing gives 1.100101 times 2 to the power 3. Remember the bias for 32-bit floats is 127. "
            "So for the smudged 8-bit exponent field on the board, we add 127 plus 3 equals 130, which is binary 10000010. "
            "Finally, dropping the hidden leading 1 leaves mantissa 100101 padded with zeros, producing 32-bit hexadecimal 0xC14A0000."
        ),
        "draw_wb": _draw_csc325_whiteboard,
        "slide_pages": (
            (
                "CSC325 — Computer Organization & Assembly",
                "IEEE 754 Single-Precision (32-Bit) Format",
                [
                    (
                        "32-Bit Field Partitioning",
                        [
                            "1-bit Sign (S): 0 for positive values, 1 for negative values.",
                            "8-bit Biased Exponent (E'): stored as E + Bias with Bias = 127 (range 1..254 normalized).",
                            "23-bit Fraction / Mantissa (M): fractional bits after the implicit hidden leading 1.",
                        ],
                    ),
                    (
                        "General Value Equation",
                        [
                            "Normalized floating-point value = (-1)^S * (1.Fraction)_2 * 2^(E' - 127).",
                            "All-zeros exponent encodes 0 or subnormals; all-ones exponent encodes Infinity or NaN.",
                        ],
                    ),
                ],
                "CampusFlow Academic Demo  |  CSC325 Lecture 1",
            ),
            (
                "CSC325 — Computer Organization & Assembly",
                "Worked Encoding Example: -12.625 to 0xC14A0000",
                [
                    (
                        "Binary & Normalization Steps",
                        [
                            "Sign S = 1 (negative); 12 = 1100_2 and 0.625 = 0.101_2 -> 1100.101_2.",
                            "Shift binary point left 3 places: 1.100101 * 2^3 (Unbiased exponent E = 3).",
                        ],
                    ),
                    (
                        "Exponent Packing & Hex Result",
                        [
                            "Biased exponent = 127 + 3 = 130 = 10000010 in 8-bit binary.",
                            "23-bit fraction = 10010100000000000000000 -> Full hex word = 0xC14A0000.",
                        ],
                    ),
                ],
                "CampusFlow Academic Demo  |  CSC325 Lecture 1",
            ),
        ),
    },
    {
        "code": "AIC354",
        "prefix": "aic354_ml",
        "audio_script": (
            "Assalam-o-Alaikum class, welcome to AIC354 Machine Learning Fundamentals. Today we derive Naive Bayes classification and Laplace smoothing. "
            "Starting from Bayes Theorem, Naive Bayes assumes features are conditionally independent given the class label y. "
            "To prevent floating-point underflow on long feature vectors, we classify in log space by summing log P of x_i given y plus log P of y. "
            "For any unseen word with count 0, multiplying by zero collapses the posterior. Using add-one Laplace smoothing, the smudged denominator on the board adds vocabulary size V to total class tokens: N_y plus V, giving P equals N_{y,i} plus 1 divided by N_y plus V."
        ),
        "draw_wb": _draw_aic354_whiteboard,
        "slide_pages": (
            (
                "AIC354 — Machine Learning Fundamentals",
                "Naive Bayes & Conditional Independence",
                [
                    (
                        "Bayes Classifier & Independence Assumption",
                        [
                            "Posterior rule: P(y | X) proportional to P(y) * Product_{i=1..n} P(x_i | y).",
                            "Assumes all features x_i are conditionally independent given target class y.",
                            "Reduces parameter estimation from exponential joint tables to O(n * |C|) likelihoods.",
                        ],
                    ),
                    (
                        "Log-Space MAP Decision Rule",
                        [
                            "Predicting argmax_y [ log P(y) + Sum log P(x_i | y) ] prevents numerical underflow.",
                            "Fast linear-time training and inference, well-suited for high-dimensional text.",
                        ],
                    ),
                ],
                "CampusFlow Academic Demo  |  AIC354 Lecture 1",
            ),
            (
                "AIC354 — Machine Learning Fundamentals",
                "Zero-Frequency Problem & Laplace Smoothing",
                [
                    (
                        "Add-1 (Laplace) Smoothing Equation",
                        [
                            "Unseen words produce P(x_i | y) = 0 and wipe out the entire class posterior.",
                            "Laplace formula: P(x_i | y) = (N_{y,i} + 1) / (N_y + |V|) where |V| is vocabulary size.",
                        ],
                    ),
                    (
                        "Worked Example (|V| = 1000, N_Spam = 200)",
                        [
                            "Seen token 'prize' (N = 5): (5 + 1) / (200 + 1000) = 6 / 1200 = 0.005.",
                            "Unseen token 'urgent' (N = 0): (0 + 1) / (200 + 1000) = 1 / 1200 = 0.000833 (safe).",
                        ],
                    ),
                ],
                "CampusFlow Academic Demo  |  AIC354 Lecture 1",
            ),
        ),
    },
    {
        "code": "CSC301",
        "prefix": "csc301_daa",
        "audio_script": (
            "Assalam-o-Alaikum class, welcome to CSC301 Design and Analysis of Algorithms. Today we analyze Binary Search and its recurrence complexity. "
            "First precondition: the input array A must already be sorted in ascending order. "
            "On the whiteboard, the midpoint equation got slightly smudged: always compute mid equals low plus high minus low integer divided by 2. "
            "Why? Computing low plus high directly can overflow signed 32-bit integers on massive arrays. "
            "At every comparison we discard half the search interval: har step mein search space ko half kar dete hain. "
            "The recurrence T of n equals T of n over 2 plus c solves by Master Theorem Case 2 to Theta of log n."
        ),
        "draw_wb": _draw_csc301_whiteboard,
        "slide_pages": (
            (
                "CSC301 — Design & Analysis of Algorithms",
                "Binary Search & Overflow-Safe Midpoint",
                [
                    (
                        "Algorithm Precondition & Step Rule",
                        [
                            "Input array A[0..n-1] must be sorted; each probe halves the active [low..high] interval.",
                            "If target == A[mid], return mid; if target < A[mid], search left half; else right half.",
                        ],
                    ),
                    (
                        "Avoiding 32-Bit Integer Overflow",
                        [
                            "Naive (low + high) // 2 overflows signed 32-bit integers when low + high > INT_MAX.",
                            "Overflow-safe formula: mid = low + (high - low) // 2 stays strictly within array bounds.",
                        ],
                    ),
                ],
                "CampusFlow Academic Demo  |  CSC301 Lecture 1",
            ),
            (
                "CSC301 — Design & Analysis of Algorithms",
                "Recurrence Relation & Worked Trace",
                [
                    (
                        "Recurrence Complexity Analysis",
                        [
                            "Recurrence: T(n) = T(n/2) + c with constant base case T(1) = c.",
                            "By Master Theorem Case 2 (a=1, b=2, n^(log_2 1)=1): T(n) = Theta(log n) worst/avg case.",
                        ],
                    ),
                    (
                        "Trace on A = [3, 8, 14, 21, 29, 37, 45], target = 37",
                        [
                            "Iteration 1: low=0, high=6 -> mid=3 (A[3]=21 < 37) -> update low=4.",
                            "Iteration 2: low=4, high=6 -> mid=5 (A[5]=37 == 37) -> found at index 5 in 2 steps.",
                        ],
                    ),
                ],
                "CampusFlow Academic Demo  |  CSC301 Lecture 1",
            ),
        ),
    },
]


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    # Remove legacy demo artifacts so demo_files/ holds the clean 12-file set
    for old_file in DEMO_DIR.iterdir():
        if old_file.is_file():
            old_file.unlink()

    for spec in DEMO_SPECS:
        prefix = str(spec["prefix"])
        wav_path = DEMO_DIR / f"{prefix}_audio.wav"
        wb_path = DEMO_DIR / f"{prefix}_whiteboard.png"
        pdf_path = DEMO_DIR / f"{prefix}_slides.pdf"

        _synthesize_wav(wav_path, str(spec["audio_script"]))
        draw_fn: Callable[[Path], None] = spec["draw_wb"]  # type: ignore[assignment]
        draw_fn(wb_path)

        p1_cfg, p2_cfg = spec["slide_pages"]  # type: ignore[misc]
        p1 = _make_slide_page(p1_cfg[0], 1, 2, p1_cfg[1], p1_cfg[2], p1_cfg[3])
        p2 = _make_slide_page(p2_cfg[0], 2, 2, p2_cfg[1], p2_cfg[2], p2_cfg[3])
        _save_two_page_pdf(pdf_path, p1, p2)

        print(f"Generated {spec['code']} kit: {wav_path.name}, {wb_path.name}, {pdf_path.name}")


if __name__ == "__main__":
    main()
