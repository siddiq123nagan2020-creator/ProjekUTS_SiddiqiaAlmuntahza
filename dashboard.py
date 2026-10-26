import csv
import datetime
import glob
import io
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from plotly import express as px
from plotly import graph_objects as go
from ultralytics import YOLO

# ==========================
# CONFIG
# ==========================
st.set_page_config(page_title="VisionAI - Deteksi Hewan Ternak", page_icon="🐄",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background: linear-gradient(135deg,#b3c7ff,#8e9fff);color:#f2f4ff;font-family:Arial,sans-serif;padding:2rem 3rem;margin:0;}
[data-testid="stHeader"], [data-testid="stToolbar"] {display:none;}
.block-container {padding:0 2rem !important;max-width:100% !important;}
@keyframes fadeSlideIn {0%{opacity:0;transform:translateY(20px);}100%{opacity:1;transform:translateY(0);}}
.main-title,.section-title,.explain-box,.metric-box {transition:all .3s ease-in-out;animation:fadeSlideIn .8s ease forwards;}
.main-title:hover,.section-title:hover,.explain-box:hover,.metric-box:hover {transform:scale(1.02);box-shadow:6px 6px 15px rgba(0,0,0,.12);}
.main-title {background:linear-gradient(145deg,#6b7bd6,#4f63c8);border:2px solid rgba(255,255,255,.08);border-radius:18px;color:#f8fbff;text-align:center;padding:20px;font-size:28px;font-weight:bold;margin:20px auto 25px auto;box-shadow:4px 6px 12px rgba(0,0,0,.08);width:100%;}
.section-title {font-size:20px;font-weight:700;background:rgba(255,255,255,.06);padding:8px 15px;border-radius:12px;color:#f2f6ff;margin:10px 0 15px;text-align:center;border:1px solid rgba(255,255,255,.04);}
.explain-box {background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.03);border-radius:10px;padding:15px;color:#f2f6ff;margin-top:10px;font-weight:500;text-align:justify;}
.metric-box {background:rgba(255,255,255,.08);border-radius:12px;padding:12px;text-align:center;color:#fff;font-weight:600;}
.metric-box span {display:block;font-size:26px;}
.pal {text-align:center;color:#1f2a5a;font-size:13px;font-weight:600;}
.pal div.sw {height:46px;border-radius:8px;margin-bottom:4px;border:1px solid rgba(0,0,0,.15);}
.footer-center {text-align:center;color:#eef2ff;font-weight:500;}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "model/best.pt"
FONT = dict(color="#1f2a5a")
COLORS = ["#4f63c8", "#22b8cf", "#2fb344", "#f5a623", "#a855f7", "#e5484d"]

for k, v in {"page": "home", "user_name": "", "user_campus": "", "used_feature": False,
             "last_mode": None, "last_pred": None, "last_conf": None}.items():
    st.session_state.setdefault(k, v)


def go_to(page: str) -> None:
    st.session_state["page"] = page


# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    yolo_model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    yolo_model = None

# ==========================
# FUNGSI ANALISIS
# ==========================
NAMES = {"Hitam": (20, 20, 20), "Putih": (240, 240, 240), "Abu-abu": (128, 128, 128), "Merah": (200, 40, 40),
         "Hijau": (60, 160, 60), "Biru": (50, 80, 200), "Kuning": (230, 210, 60), "Oranye": (240, 140, 40),
         "Cokelat": (130, 85, 50), "Krem": (225, 210, 180), "Cyan": (50, 200, 210), "Ungu": (140, 70, 170)}


def color_name(rgb):
    return min(NAMES, key=lambda n: sum((a - b) ** 2 for a, b in zip(NAMES[n], rgb)))


def analyze_colors(img: Image.Image, k: int = 5) -> dict:
    """Analisis warna seluruh gambar: palet K-Means, histogram RGB, kecerahan."""
    px_ = np.asarray(img.convert("RGB")).reshape(-1, 3)
    rng = np.random.default_rng(0)
    sample = px_[rng.choice(len(px_), min(20000, len(px_)), replace=False)].astype(np.float32)
    k = max(1, min(k, len(np.unique(sample, axis=0))))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(sample, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k)
    palette = []
    for i in np.argsort(-counts):
        rgb = tuple(int(v) for v in centers[i])
        palette.append({"hex": "#%02X%02X%02X" % rgb, "pct": round(100 * counts[i] / counts.sum(), 1),
                        "name": color_name(rgb)})
    hist = {c: np.histogram(px_[:, i], bins=32, range=(0, 256))[0] for i, c in enumerate("RGB")}
    gray = px_ @ np.array([0.299, 0.587, 0.114])
    return {"palette": palette, "hist": hist, "mean_brightness": float(gray.mean()),
            "bright": np.histogram(gray, bins=32, range=(0, 256))[0]}


def to_png_bytes(arr_rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr_rgb).save(buf, format="PNG")
    return buf.getvalue()


def style_fig(fig, h=300):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(255,255,255,0.35)", font=FONT)
    return fig


# ==========================
# PAGES
# ==========================
def halaman_awal():
    st.markdown('<div class="main-title">🐄 <b>SELAMAT DATANG DI DASHBOARD SIDDIQIA ALMUNTAHZA</b> 🐃</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Aplikasi Deteksi Hewan Ternak (YOLOv8) &amp; Analisis Warna Gambar</div>', unsafe_allow_html=True)
    st.markdown('<div class="explain-box">Halo! Aplikasi ini mendeteksi hewan (kerbau, kuda, nilgai, sapi) pada gambar menggunakan model '
                'YOLOv8 hasil training sendiri, lalu menampilkan confidence, koordinat bounding box, serta analisis pola warna gambar. '
                'Klik tombol di bawah untuk melanjutkan ke registrasi singkat.</div>', unsafe_allow_html=True)
    st.write("")
    st.button("Lanjut ke Registrasi →", on_click=go_to, args=("register",))


def halaman_registrasi():
    st.markdown('<div class="main-title">📝 Registrasi Singkat</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Isi data berikut sebelum masuk ke dashboard</div>', unsafe_allow_html=True)
    with st.form("form_reg"):
        name = st.text_input("Nama Lengkap", value=st.session_state["user_name"])
        campus = st.text_input("Nama Kampus", value=st.session_state["user_campus"])
        if st.form_submit_button("Masuk ke Dashboard Utama"):
            if not name.strip() or not campus.strip():
                st.warning("Mohon isi Nama dan Nama Kampus sebelum melanjutkan.")
            else:
                st.session_state.update(user_name=name.strip(), user_campus=campus.strip(), page="main")
                st.rerun()
    st.button("⬅️ Kembali ke Halaman Awal", on_click=go_to, args=("home",))


def tampil_deteksi(img: Image.Image, conf_thr: float, col1, col2):
    if yolo_model is None:
        st.error("Model YOLO belum tersedia. Pastikan file model/best.pt ada.")
        return
    with st.spinner("🚀 AI sedang mendeteksi objek... harap tunggu sebentar!"):
        bgr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])  # ultralytics membaca array sebagai BGR
        res = yolo_model.predict(bgr, conf=conf_thr, verbose=False)[0]
        result_rgb = res.plot()[:, :, ::-1]
    boxes = res.boxes
    with col2:
        if boxes is None or len(boxes) == 0:
            st.image(result_rgb, caption="📦 Hasil Deteksi Objek", width="stretch")
            st.warning(f"⚠️ Tidak ada objek yang memenuhi ambang confidence {conf_thr:.2f}.")
            return
        st.image(result_rgb, caption="📦 Hasil Deteksi Objek", width="stretch")
        st.download_button("⬇️ Unduh Gambar Hasil Deteksi", to_png_bytes(result_rgb), "hasil_deteksi.png", "image/png")
    rows = []
    for i, b in enumerate(boxes, 1):
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
        rows.append({"No.": i, "Kelas": res.names[int(b.cls[0])], "Confidence (%)": round(float(b.conf[0]) * 100, 2),
                     "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    df = pd.DataFrame(rows)
    confs = df["Confidence (%)"]
    st.success(f"✅ {len(df)} objek berhasil terdeteksi!")
    c = st.columns(4)
    for col, (lab, val) in zip(c, [("Total Objek", len(df)), ("Rata-rata Confidence", f"{confs.mean():.1f}%"),
                                   ("Tertinggi", f"{confs.max():.1f}%"), ("Terendah", f"{confs.min():.1f}%")]):
        col.markdown(f'<div class="metric-box">{lab}<span>{val}</span></div>', unsafe_allow_html=True)
    per = df["Kelas"].value_counts().rename_axis("Kelas").reset_index(name="Jumlah")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="section-title">Jumlah Objek per Kelas</div>', unsafe_allow_html=True)
        st.plotly_chart(style_fig(px.bar(per, x="Kelas", y="Jumlah", color="Kelas", color_discrete_sequence=COLORS)),
                        width="stretch", config={"displayModeBar": False})
    with g2:
        st.markdown('<div class="section-title">Proporsi Kelas</div>', unsafe_allow_html=True)
        st.plotly_chart(style_fig(px.pie(per, values="Jumlah", names="Kelas", hole=0.45, color_discrete_sequence=COLORS)),
                        width="stretch", config={"displayModeBar": False})
    st.subheader("📋 Rincian Hasil Deteksi")
    st.dataframe(df, width="stretch", hide_index=True)
    st.markdown("> 💡 Koordinat dalam piksel (xyxy) pada gambar asli. Confidence menunjukkan keyakinan model, bukan akurasi sebenarnya.")
    st.session_state.update(last_mode="detection", used_feature=True, last_conf=float(confs.mean()),
                            last_pred=", ".join(f"{r.Kelas} ({r.Jumlah})" for r in per.itertuples()))


def tampil_warna(img: Image.Image):
    a = analyze_colors(img)
    st.markdown('<div class="section-title">🎨 Warna Dominan (K-Means, 5 cluster)</div>', unsafe_allow_html=True)
    st.markdown('<div class="explain-box">Analisis dilakukan pada <b>seluruh gambar</b>, bukan hanya objek terdeteksi. '
                'Ini deskripsi visual, bukan ciri khas suatu kelas hewan.</div>', unsafe_allow_html=True)
    cols = st.columns(len(a["palette"]))
    for col, p in zip(cols, a["palette"]):
        col.markdown(f'<div class="pal"><div class="sw" style="background:{p["hex"]}"></div>{p["hex"]}<br>{p["pct"]:.1f}%<br>{p["name"]}</div>',
                     unsafe_allow_html=True)
    x = list(range(0, 256, 8))
    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div class="section-title">Histogram RGB</div>', unsafe_allow_html=True)
        fig = go.Figure([go.Scatter(x=x, y=a["hist"][c], name=c, line=dict(color=col)) for c, col in
                         zip("RGB", ["#e5484d", "#2fb344", "#3b82f6"])])
        fig.update_xaxes(title="Intensitas (0-255)"); fig.update_yaxes(title="Jumlah piksel")
        st.plotly_chart(style_fig(fig), width="stretch", config={"displayModeBar": False})
    with g2:
        st.markdown('<div class="section-title">Distribusi Kecerahan</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=x, y=a["bright"], marker_color="#4f63c8"))
        fig.update_xaxes(title="Kecerahan (0-255)"); fig.update_yaxes(title="Jumlah piksel")
        st.plotly_chart(style_fig(fig), width="stretch", config={"displayModeBar": False})
    st.write(f"Kecerahan rata-rata: **{a['mean_brightness']:.1f}** / 255")
    st.session_state.update(last_mode="color", used_feature=True, last_conf=None,
                            last_pred="Warna dominan " + ", ".join(p["hex"] for p in a["palette"][:3]))


def halaman_main():
    st.sidebar.image("model/LOGO UUSK.jpg", width="stretch")
    menu = st.sidebar.radio("🧭 Pilih Mode Analisis:", ["Deteksi Objek (YOLO)", "Analisis Warna Gambar"])
    conf_thr = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    samples = sorted(glob.glob("sample_images/*.jpg") + glob.glob("sample_images/*.png"))
    sample = st.sidebar.selectbox("Atau pilih gambar contoh:", ["-"] + samples, format_func=lambda p: p if p == "-" else os.path.basename(p))
    st.sidebar.info("📤 Unggah gambar atau pilih gambar contoh untuk mulai analisis AI.")

    st.markdown('<div class="main-title">🔎 Dashboard Utama — Deteksi Hewan &amp; Analisis Warna</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="explain-box">Halo <b>{st.session_state["user_name"]}</b> dari <b>{st.session_state["user_campus"]}</b> '
                '— pilih mode dan unggah gambar untuk memulai.</div>', unsafe_allow_html=True)
    up = st.file_uploader("📂 Unggah Gambar", type=["jpg", "jpeg", "png", "webp"])
    src = up if up is not None else (sample if sample != "-" else None)
    if src is None:
        st.info("👆 Silakan unggah gambar atau pilih gambar contoh terlebih dahulu.")
    else:
        try:
            img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
        except Exception:
            st.error("File bukan gambar yang valid.")
            img = None
        if img is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption=f"🖼️ Gambar Diupload ({img.width}×{img.height} px)", width="stretch")
            if menu == "Deteksi Objek (YOLO)":
                tampil_deteksi(img, conf_thr, col1, col2)
            else:
                tampil_warna(img)
            if st.session_state["used_feature"]:
                st.button("Berikan Feedback", on_click=go_to, args=("feedback",))
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("⬅️ Kembali ke Registrasi", on_click=go_to, args=("register",))


def halaman_feedback():
    st.markdown('<div class="main-title">💬 Feedback Pengguna</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Terima kasih telah menggunakan aplikasi — beri masukanmu di bawah ini</div>', unsafe_allow_html=True)
    with st.form("form_feedback"):
        rating = st.slider("Rating (1 - 5)", 1, 5, 5)
        comment = st.text_area("Komentar / Saran", height=120)
        if st.form_submit_button("Kirim Feedback"):
            row = {"timestamp": datetime.datetime.now().isoformat(), "user_name": st.session_state["user_name"],
                   "user_campus": st.session_state["user_campus"], "mode": st.session_state["last_mode"] or "",
                   "prediction": st.session_state["last_pred"] or "", "confidence": st.session_state["last_conf"] or "",
                   "rating": rating, "comment": comment}
            try:
                new = not os.path.exists("feedbacks.csv")
                with open("feedbacks.csv", "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    if new:
                        w.writeheader()
                    w.writerow(row)
                st.success("✅ Feedback berhasil dikirim — terima kasih!")
                st.session_state["used_feature"] = False
            except Exception as e:
                st.error(f"Gagal menyimpan feedback: {e}")
    st.button("⬅️ Kembali ke Dashboard", on_click=go_to, args=("main",))


# ==========================
# ROUTING
# ==========================
{"home": halaman_awal, "register": halaman_registrasi, "main": halaman_main,
 "feedback": halaman_feedback}.get(st.session_state["page"], halaman_awal)()

st.markdown("---")
st.markdown("<center>created by <b>Siddiqia Almuntahza | 2308108010004</b><br>Proyek UAS Praktikum Pemrograman Big Data | Universitas Syiah Kuala</center>", unsafe_allow_html=True)
