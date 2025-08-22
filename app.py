import streamlit as st
import google.generativeai as genai
import cv2
import numpy as np
import PIL.Image
import io
import asyncio
import platform
import time

# Lớp GeminiModel
class GeminiModel:
    def __init__(self, api_key=None):
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def extract_text_from_image(self, image_bytes, prompt="Đọc tất cả biển số xe trong ảnh sau (chỉ trả lời biển số, mỗi biển số trên một dòng):", retries=3):
        for retry in range(retries):
            try:
                img = PIL.Image.open(io.BytesIO(image_bytes))
                response = self.model.generate_content([prompt, img])
                text = response.text.strip()
                if text and any(c.isalnum() for c in text):
                    plates = [plate.strip() for plate in text.split('\n') if plate.strip()]
                    return plates if plates else None
                else:
                    raise ValueError("Không nhận diện được biển số hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi API hoặc kết nối ({retry + 1}/{retries}): {e}")
                if retry < retries - 1:
                    st.info("Đang thử lại sau 30 giây...")
                    time.sleep(30)
                else:
                    st.error("Hết số lần thử. Hãy thử video/ảnh khác hoặc kiểm tra API Key.")
        return None

# Hàm xử lý video
async def process_video(video_file, gemini_model):
    tfile = io.BytesIO(video_file.read())
    cap = cv2.VideoCapture(tfile)
    stframe = st.empty()
    all_plates = set()  # Sử dụng set để tránh trùng lặp
    frame_count = 0
    max_frames = 300  # Giới hạn số frame để tránh quá tải (tùy chỉnh)

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 30 == 0:  # Xử lý 1 frame mỗi giây (30 FPS giả định)
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            plates = gemini_model.extract_text_from_image(image_bytes)
            if plates:
                all_plates.update(plates)  # Cập nhật set với danh sách mới
            stframe.image(frame, caption=f"Frame {frame_count}", channels="BGR", use_column_width=True)
            await asyncio.sleep(1.0 / 30)  # Điều chỉnh tốc độ hiển thị frame
    cap.release()
    return list(all_plates)

# Trang web Streamlit
def main():
    st.title("Nhận Diện Biển Số Xe từ Video/Ảnh")
    st.write("Tải lên video hoặc ảnh chứa biển số xe để nhận diện. Vui lòng nhập API Key Gemini.")

    # Nhập API Key thủ công
    api_key = st.text_input("Nhập API Key Gemini của bạn:", type="password", key="api_key_input")
    if not api_key:
        st.warning("Vui lòng nhập API Key để sử dụng.")
        return

    # Khởi tạo model
    try:
        gemini_model = GeminiModel(api_key=api_key)
    except Exception as e:
        st.error(f"Lỗi cấu hình API: {e}")
        return

    # Upload file (video hoặc ảnh)
    uploaded_file = st.file_uploader("Chọn video hoặc ảnh (MP4, JPG, PNG, JPEG)", type=["mp4", "jpg", "png", "jpeg"], key="file_uploader")

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        if file_type == "video":
            st.video(uploaded_file)
            if st.button("Nhận Diện Biển Số", key="recognize_button"):
                with st.spinner("Đang xử lý video..."):
                    plates = asyncio.run(process_video(uploaded_file, gemini_model))
                    if plates:
                        st.success("Kết quả nhận diện từ video:")
                        for plate in plates:
                            st.write(plate)
                        st.download_button(
                            label="Tải xuống kết quả",
                            data="\n".join(plates),
                            file_name="plate_results.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("Không nhận diện được biển số. Hãy thử video/ảnh khác.")
        else:  # Xử lý ảnh
            image_bytes = uploaded_file.read()
            st.image(image_bytes, caption="Ảnh đầu vào", use_column_width=True)
            if st.button("Nhận Diện Biển Số", key="recognize_button"):
                with st.spinner("Đang xử lý..."):
                    plates = gemini_model.extract_text_from_image(image_bytes)
                    if plates:
                        st.success("Kết quả nhận diện:")
                        for plate in plates:
                            st.write(plate)
                        st.download_button(
                            label="Tải xuống kết quả",
                            data="\n".join(plates),
                            file_name="plate_results.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("Không nhận diện được biển số. Hãy thử ảnh khác.")

    # Thêm dòng chữ ở góc cuối trang
    st.markdown(
        """
        <div style='position: fixed; bottom: 10px; right: 10px; font-size: 12px; color: gray;'>
            Võ Công Nhật-20222627
        </div>
        """,
        unsafe_allow_html=True
    )

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
