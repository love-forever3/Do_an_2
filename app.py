import streamlit as st
import google.generativeai as genai
import PIL.Image
import time
import io
import os

# Lớp GeminiModel
class GeminiModel:
    def __init__(self, model_name="gemini-1.5-flash", api_key=None):
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def extract_text_from_image(self, 
        image_bytes, 
        prompt="Đọc biển số xe trong ảnh sau (câu trả lời chỉ có biển số xe, không thêm bất kỳ ký tự nào khác):",
        retries=3):
        for retry in range(retries):
            try:
                img = PIL.Image.open(io.BytesIO(image_bytes))
                response = self.model.generate_content([prompt, img])
                text = response.text.strip()
                if text and any(c.isalnum() for c in text):
                    if '-' in text and '.' in text:
                        parts = text.split('-')
                        number_part = parts[1].replace('.', '.\n')
                        text = f"{parts[0]}-\n{number_part}"
                    return text
                else:
                    raise ValueError("Không nhận diện được biển số hợp lệ.")
            except Exception as e:
                st.error(f"Lỗi API hoặc kết nối ({retry + 1}/{retries}): {e}")
                if retry < retries - 1:
                    st.info("Đang thử lại sau 30 giây...")
                    time.sleep(30)
                else:
                    st.error("Hết số lần thử. Hãy thử ảnh khác hoặc kiểm tra API Key.")
        return None

# Trang web Streamlit
def main():
    st.title("Nhận Diện Biển Số Xe")
    st.write("Tải lên ảnh chứa biển số xe để nhận diện. Kết quả sẽ hiển thị ngay bên dưới.")

    # Lấy API key từ biến môi trường
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Nhập API Key Gemini của bạn:", type="password", key="api_key_input")
        if not api_key:
            st.warning("Vui lòng nhập API Key để sử dụng hoặc cấu hình GEMINI_API_KEY trong biến môi trường.")
            return
    else:
        st.write("API Key đã được cấu hình qua biến môi trường.")

    # Khởi tạo model
    try:
        gemini_model = GeminiModel(api_key=api_key)
    except Exception as e:
        st.error(f"Lỗi cấu hình API: {e}")
        return

    # Upload ảnh
    uploaded_file = st.file_uploader("Chọn ảnh (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"], key="file_uploader")

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()
        st.image(image_bytes, caption="Ảnh đầu vào", use_column_width=True)

        if st.button("Nhận Diện Biển Số", key="recognize_button"):
            with st.spinner("Đang xử lý..."):
                result = gemini_model.extract_text_from_image(image_bytes)
                if result:
                    st.success("Kết quả nhận diện:")
                    for line in result.split('\n'):
                        st.write(line)
                    st.download_button(
                        label="Tải xuống kết quả",
                        data=result,
                        file_name="plate_result.txt",
                        mime="text/plain"
                    )
                else:
                    st.error("Không nhận diện được biển số. Hãy thử ảnh rõ nét hơn hoặc kiểm tra API Key.")

    # Thêm dòng chữ ở góc cuối trang
    st.markdown(
        """
        <div style='position: fixed; bottom: 10px; right: 10px; font-size: 12px; color: gray;'>
            Đồ án II, GVHD: ThS. Nguyễn Thị Huế
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
