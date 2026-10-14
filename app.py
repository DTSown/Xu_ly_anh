"""Web demo: python app.py  ->  mo http://127.0.0.1:5000
Upload 1 trang truyen, chinh tham so, xem tung buoc xu ly va cac panel da cat.
Van chi dung cv2.imread / cv2.imwrite (luu upload ra file roi doc lai)."""
import os, time, uuid, shutil
import cv2
from flask import Flask, render_template, request, send_from_directory, abort
from main import process

BASE = os.path.dirname(os.path.abspath(__file__))
WEB_OUT = os.path.join(BASE, "web_output")
ALLOWED = {".png", ".jpg", ".jpeg", ".bmp"}
STEPS = [("1_gray.png", "1. Anh xam (thu nho)"),
         ("2_blur.png", "2. Loc Gauss"),
         ("3_binary_otsu_inverted.png", "3. Otsu + dao anh (net ve = trang)"),
         ("4_closing.png", "4. Closing (noi net dut)"),
         ("5_result.png", "5. Ket qua: khung panel")]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
os.makedirs(WEB_OUT, exist_ok=True)


def cleanup(keep=20):
    """Chi giu lai `keep` lan chay gan nhat de khong day o dia."""
    dirs = sorted((os.path.join(WEB_OUT, d) for d in os.listdir(WEB_OUT)), key=os.path.getmtime)
    for d in dirs[:-keep]:
        shutil.rmtree(d, ignore_errors=True)


def read_params(form):
    def num(key, default, lo, hi, cast=float):
        try:
            return min(max(cast(form.get(key, default)), lo), hi)
        except (TypeError, ValueError):
            return default
    return dict(sigma=num("sigma", 1.5, 0.3, 5.0),
                close_px=num("close_px", 20, 1, 80, int),
                use_blur=form.get("use_blur") == "on",
                use_closing=form.get("use_closing") == "on")


@app.route("/", methods=["GET", "POST"])
def index():
    ctx = dict(error=None, result=None,
               params=dict(sigma=1.5, close_px=20, use_blur=True, use_closing=True))
    if request.method == "GET":
        return render_template("index.html", **ctx)

    f = request.files.get("image")
    ctx["params"] = params = read_params(request.form) if request.form.get("use_closing") or request.form.get("use_blur") \
        or request.form.get("sigma") else ctx["params"]
    if not f or not f.filename:
        ctx["error"] = "Chua chon anh."
        return render_template("index.html", **ctx)
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        ctx["error"] = f"Dinh dang {ext or '(khong ro)'} khong ho tro. Dung: png, jpg, jpeg, bmp."
        return render_template("index.html", **ctx)

    token = uuid.uuid4().hex[:12]
    tmp = os.path.join(WEB_OUT, token + ext)
    f.save(tmp)
    try:
        if cv2.imread(tmp) is None:
            ctx["error"] = "Khong doc duoc anh (file hong hoac khong phai anh)."
            return render_template("index.html", **ctx)
        t0 = time.time()
        boxes, dt = process(tmp, WEB_OUT, **params)
        # process() dat ten thu muc theo ten file (token)
        panels = sorted(os.listdir(os.path.join(WEB_OUT, token, "panels")))
        ctx["result"] = dict(token=token, n=len(boxes), time=dt, total=time.time() - t0,
                             boxes=boxes, panels=panels, steps=STEPS)
    except Exception as e:  # khong de server sap
        ctx["error"] = f"Loi xu ly: {type(e).__name__}: {e}"
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
        cleanup()
    return render_template("index.html", **ctx)


@app.route("/files/<token>/<path:name>")
def files(token, name):
    if not token.isalnum():
        abort(404)
    return send_from_directory(os.path.join(WEB_OUT, token), name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
