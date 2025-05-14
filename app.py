import os
from PIL import Image
import pytesseract
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
import streamlit as st

# Настройки страницы
st.set_page_config(
    page_title="Переводчик RU ↔ ES с OCR",
    page_icon="🌍",
    layout="wide"
)

# Кэшируем загрузку моделей для ускорения работы
@st.cache_resource
def load_models():
    model_ru_es = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ru-es")
    tokenizer_ru_es = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ru-es")
    
    model_es_ru = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-es-ru")
    tokenizer_es_ru = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-ru")
    
    return {
        "Русский → Испанский": (model_ru_es, tokenizer_ru_es),
        "Испанский → Русский": (model_es_ru, tokenizer_es_ru)
    }

models = load_models()

# Разделение длинного текста на части
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

# Предобработка изображения
def preprocess_image(image):
    gray_image = image.convert('L')
    binarized_image = gray_image.point(lambda p: p > 128 and 255)
    return binarized_image

# Функция перевода
def translate_text(text, direction):
    if not text.strip():
        return "Ошибка: введите текст для перевода."
    
    if len(text) > 5000:
        return "Ошибка: текст слишком длинный (максимум 5000 символов)."

    if direction == "Автоопределение":
        try:
            lang = detect(text)
            if lang == "ru":
                direction = "Русский → Испанский"
            elif lang == "es":
                direction = "Испанский → Русский"
            else:
                return "Ошибка: поддерживаются только русский и испанский языки."
        except:
            return "Ошибка: не удалось определить язык."

    if direction not in models:
        return "Ошибка: неверное направление перевода."

    model, tokenizer = models[direction]
    text_parts = split_text(text)
    translations = []
    
    for part in text_parts:
        inputs = tokenizer(part, return_tensors="pt")
        translated_tokens = model.generate(**inputs)
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        translations.append(translation)
    
    return ' '.join(translations)

# OCR + перевод
def ocr_and_translate(image, direction):
    try:
        preprocessed_image = preprocess_image(image)
        extracted_text = pytesseract.image_to_string(preprocessed_image, lang="rus+spa")
        
        if not extracted_text.strip():
            return "Не удалось распознать текст на изображении", ""
            
        translation = translate_text(extracted_text, direction)
        return extracted_text.strip(), translation
    except Exception as e:
        return f"Ошибка распознавания: {str(e)}", ""

# Интерфейс Streamlit
st.title("🇷🇺 Переводчик с OCR 🇪🇸")
st.write("Перевод текста и изображений между русским и испанским языками")

tab1, tab2 = st.tabs(["📝 Текстовый перевод", "🖼️ Перевод с изображения"])

with tab1:
    with st.form("text_translation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            text_input = st.text_area("Введите текст:", height=200)
            direction_text = st.radio(
                "Направление перевода:",
                ["Автоопределение", "Русский → Испанский", "Испанский → Русский"],
                index=0
            )
            submit_text = st.form_submit_button("Перевести")
        
        with col2:
            if submit_text and text_input:
                with st.spinner("Переводим..."):
                    result = translate_text(text_input, direction_text)
                st.text_area("Переведённый текст:", value=result, height=200)
            else:
                st.text_area("Переведённый текст:", height=200)

with tab2:
    with st.form("image_translation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            image_input = st.file_uploader(
                "Загрузите изображение:",
                type=["jpg", "jpeg", "png"]
            )
            if image_input:
                image = Image.open(image_input)
                st.image(image, caption="Загруженное изображение", width=300)
                
            direction_image = st.radio(
                "Направление перевода:",
                ["Автоопределение", "Русский → Испанский", "Испанский → Русский"],
                index=0,
                key="image_direction"
            )
            submit_image = st.form_submit_button("Распознать и перевести")
        
        with col2:
            if submit_image and image_input:
                with st.spinner("Обрабатываем изображение..."):
                    extracted, translated = ocr_and_translate(image, direction_image)
                
                st.text_area("Распознанный текст:", value=extracted, height=100)
                st.text_area("Переведённый текст:", value=translated, height=100)
            else:
                st.text_area("Распознанный текст:", height=100)
                st.text_area("Переведённый текст:", height=100)

# Подсказка про Tesseract
st.sidebar.info(
    "Для работы с изображениями требуется Tesseract OCR. "
    "На локальной машине установите: \n\n"
    "`sudo apt install tesseract-ocr tesseract-ocr-rus tesseract-ocr-spa`"
)
