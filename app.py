import os
import subprocess
from PIL import Image
import pytesseract
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
import gradio as gr

# Установка Tesseract и языковых пакетов
if not os.path.exists("/usr/bin/tesseract"):
    subprocess.run(["apt-get", "update"])
    subprocess.run([
        "apt-get", "install", "-y",
        "tesseract-ocr", "tesseract-ocr-rus", "tesseract-ocr-spa"
    ])

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Загрузка моделей
model_ru_es_name = "Helsinki-NLP/opus-mt-ru-es"
model_es_ru_name = "Helsinki-NLP/opus-mt-es-ru"

tokenizer_ru_es = MarianTokenizer.from_pretrained(model_ru_es_name)
model_ru_es = MarianMTModel.from_pretrained(model_ru_es_name)

tokenizer_es_ru = MarianTokenizer.from_pretrained(model_es_ru_name)
model_es_ru = MarianMTModel.from_pretrained(model_es_ru_name)

# Разделение текста на части
def split_text(text, max_length=500):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) > max_length:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += (sentence + '. ')
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Предобработка изображения (перевод в ч/б)
def preprocess_image(image):
    gray_image = image.convert('L')
    binarized_image = gray_image.point(lambda p: p > 128 and 255)
    return binarized_image

# Перевод текста
def translate_text(text, direction):
    if not text.strip():
        return "Ошибка: введите текст для перевода."
    if len(text) > 500:
        return "Ошибка: текст слишком длинный (максимум 500 символов)."

    if direction == "Автоопределение":
        lang = detect(text)
        if lang == "ru":
            direction = "Русский → Испанский"
        elif lang == "es":
            direction = "Испанский → Русский"
        else:
            return "Ошибка: поддерживаются только русский и испанский языки."

    if direction == "Русский → Испанский":
        tokenizer = tokenizer_ru_es
        model = model_ru_es
    elif direction == "Испанский → Русский":
        tokenizer = tokenizer_es_ru
        model = model_es_ru
    else:
        return "Ошибка: неверное направление перевода."

    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(**inputs)
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translation

# OCR + перевод
def ocr_and_translate(image, direction):
    try:
        # Предобрабатываем изображение
        preprocessed_image = preprocess_image(image)
        
        # Извлекаем текст
        extracted_text = pytesseract.image_to_string(preprocessed_image, lang="rus+spa")
        
        # Разбиваем текст и переводим
        text_parts = split_text(extracted_text)
        translated_parts = [translate_text(part, direction) for part in text_parts]
        
        # Возвращаем результат
        final_translation = ' '.join(translated_parts)
        return extracted_text.strip(), final_translation
    except Exception as e:
        return f"Ошибка распознавания: {e}", ""

# Интерфейс Gradio
with gr.Blocks(title="Переводчик RU ↔ ES с OCR") as app:
    gr.Markdown("## 🇷🇺 Перевод текста и изображений 🇪🇸")

    with gr.Tab("📝 Текстовый перевод"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(lines=5, label="Введите текст")
                direction_text = gr.Radio(
                    ["Автоопределение", "Русский → Испанский", "Испанский → Русский"],
                    value="Автоопределение",
                    label="Направление перевода"
                )
                btn_text = gr.Button("Перевести")
            with gr.Column():
                output_text = gr.Textbox(lines=5, label="Переведённый текст")

        btn_text.click(fn=translate_text, inputs=[text_input, direction_text], outputs=output_text)

    with gr.Tab("🖼️ Перевод с изображения"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Загрузите изображение")
                direction_image = gr.Radio(
                    ["Автоопределение", "Русский → Испанский", "Испанский → Русский"],
                    value="Автоопределение",
                    label="Направление перевода"
                )
                btn_image = gr.Button("Распознать и перевести")
            with gr.Column():
                extracted_box = gr.Textbox(lines=5, label="Распознанный текст")
                translated_box = gr.Textbox(lines=5, label="Переведённый текст")

        btn_image.click(
            fn=ocr_and_translate,
            inputs=[image_input, direction_image],
            outputs=[extracted_box, translated_box]
        )

app.launch()
