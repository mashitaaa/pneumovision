import json
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_NAME = "best_model.keras"

if os.path.exists(MODEL_NAME):
    model = tf.keras.models.load_model(MODEL_NAME)
else:
    model = tf.keras.models.load_model("pneumonia_model.keras")

with open("label_map.json") as f:
    label_map = json.load(f)

IMG_SIZE = (224, 224)

def predict(image):
    if image is None:
        return None, "<div style='color: #94A3B8; padding-top: 10px; font-size: 0.95rem;'>Unggah foto rontgen dada untuk memuat ringkasan analisis.</div>"

    temp_path = "temp_hf_input.jpg"
    image.convert("RGB").save(temp_path, "JPEG")

    img_keras = load_img(temp_path, target_size=IMG_SIZE)
    img_arr = img_to_array(img_keras)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    img_preprocessed = preprocess_input(img_arr.copy())
    img_batch = np.expand_dims(img_preprocessed, axis=0)

    prob = float(model.predict(img_batch, verbose=0)[0][0])

    results = {
        label_map["0"]: round(1 - prob, 4),
        label_map["1"]: round(prob, 4),
    }

    label = label_map["1"] if prob > 0.5 else label_map["0"]
    conf = prob if prob > 0.5 else 1 - prob

    if label == "PNEUMONIA":
        verdict = f"""
        <div style='padding-top: 15px;'>
            <h4 style='margin-top: 0; margin-bottom: 6px; font-weight: 700; font-size: 1.1rem; letter-spacing: 0.5px; color: #FF6B6B;'>⚠️ PNEUMONIA TERDETEKSI</h4>
            <p style='margin-bottom: 0; font-size: 0.95rem; line-height: 1.5; color: #CBD5E1;'>Indikasi konsolidasi parenkim paru ditemukan dengan tingkat keyakinan <b>{conf*100:.1f}%</b>. Disarankan konfirmasi klinis lanjutan oleh spesialis radiologi.</p>
        </div>
        """
    else:
        verdict = f"""
        <div style='padding-top: 15px;'>
            <h4 style='margin-top: 0; margin-bottom: 6px; font-weight: 700; font-size: 1.1rem; letter-spacing: 0.5px; color: #38BDF8;'>✅ KONDISI PARU NORMAL</h4>
            <p style='margin-bottom: 0; font-size: 0.95rem; line-height: 1.5; color: #CBD5E1;'>Struktur paru tampak bersih dari tanda infiltrat signifikan. Tingkat keyakinan model: <b>{conf*100:.1f}%</b>.</p>
        </div>
        """

    return results, verdict

with gr.Blocks(
    title="Pneumonia Detection System",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css="""
    .gradio-container { background-color: #0F172A !important; color: #F8FAFC !important; }
    .main-container { max-width: 950px; margin: 3rem auto; padding: 0 1.5rem; }
    .header-box { border-bottom: 1px solid #334155; padding-bottom: 1.5rem; margin-bottom: 2.5rem; }
    .header-box h1 { font-size: 1.75rem; font-weight: 700; color: #F8FAFC !important; margin-bottom: 0.25rem; }
    .header-box p { color: #94A3B8 !important; font-size: 0.95rem; }

    .block, .gr-group, .gr-box, .gr-form, .gr-card {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    .section-title { font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #38BDF8 !important; margin-bottom: 1rem; }
    .meta-text, .output-class, .confidence { color: #E2E8F0 !important; font-weight: 500 !important; }

    .bar {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%) !important;
        border-radius: 4px !important;
    }

    button.primary { background-color: #2563EB !important; border: none !important; color: white !important; }
    button.primary:hover { background-color: #1D4ED8 !important; }

    .footer-box { text-align: center; color: #64748B; font-size: 0.8rem; margin-top: 5rem; border-top: 1px solid #334155; padding-top: 1.5rem; }
    footer { display: none !important; }
    """,
) as demo:

    with gr.Column(elem_classes=["main-container"]):

        with gr.Column(elem_classes=["header-box"]):
            gr.Markdown("# PneumoVision+")
            gr.Markdown("Asisten berbasis Deep Learning (EfficientNetB0) untuk skrining indikasi pneumonia klinis.")

        with gr.Row(equal_height=True):

            with gr.Column(scale=5):
                gr.Markdown("### Input Citra Rontgen", elem_classes=["section-title"])

                image_input = gr.Image(
                    type="pil",
                    label="Chest X-Ray (AP/PA)",
                    height=310,
                    show_label=False,
                )

                submit_btn = gr.Button(
                    "Jalankan Analisis",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=5):
                gr.Markdown("### Hasil Analisis", elem_classes=["section-title"])

                label_output = gr.Label(
                    label="Nilai Kepercayaan Kelas",
                    show_label=False,
                )

                verdict_output = gr.HTML(
                    value="<div style='color: #94A3B8; padding-top: 10px; font-size: 0.95rem;'>Unggah foto rontgen dada untuk memuat ringkasan analisis.</div>"
                )

        submit_btn.click(
            fn=predict,
            inputs=image_input,
            outputs=[label_output, verdict_output],
        )

        image_input.change(
            fn=predict,
            inputs=image_input,
            outputs=[label_output, verdict_output],
        )

        with gr.Column(elem_classes=["footer-box"]):
            gr.Markdown(
                "Sistem ini dikembangkan untuk pemenuhan CPMK 6 — Klasifikasi Gambar Medis.<br>"
                "Bukan instrumen diagnosis final. Seluruh interpretasi wajib dikonfirmasi ulang oleh tenaga medis ahli."
            )

if __name__ == "__main__":
    demo.launch()
