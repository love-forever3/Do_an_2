import streamlit as st
import google.generativeai as genai
import PIL.Image
import time
import io

# Lớp GeminiModel
class GeminiModel:
    def __init__(self, model_name="gemini-1.5-flash", api_key=None):
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def extract_text_from_image(self, image_bytes, prompt="Đọc tất cả biển số xe trong ảnh sau (chỉ trả lời biển số, mỗi biển số trên một dòng):", retries=3):
        for retry in range(retries):
            try:
                img = PIL.Image.open(io.BytesIO(image_bytes))
                response = self.model.generate_content([prompt, img])
                text = response.text.strip()
                if text and any(c.isalnum() for c in text):
                    # Tách các biển số thành danh sách, mỗi dòng là một biển số
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
                    st.error("Hết số lần thử. Hãy thử ảnh khác hoặc kiểm tra API Key.")
        return None

# Trang web Streamlit
def main():
    st.title("Nhận Diện Biển Số Xe")
    st.write("Tải lên ảnh chứa biển số xe để nhận diện. Vui lòng nhập API Key Gemini.")

    # Nhập API Key thủ công (không dùng biến môi trường)
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

    # Upload ảnh
    uploaded_file = st.file_uploader("Chọn ảnh (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"], key="file_uploader")

    if uploaded_file is not None:
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
                    st.error("Không nhận diện được biển số. Hãy thử ảnh rõ nét hơn hoặc kiểm tra API Key.")

    # Thêm dòng chữ ở góc cuối trang
   import streamlit as st

st.markdown(
    """
    <div style='
        position: fixed; 
        bottom: 20px; 
        left: 50%; 
        transform: translateX(-50%); 
        font-size: 14px; 
        color: white; 
        background-color: #4CAF50; 
        padding: 8px 16px; 
        border-radius: 8px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.3);
    '>
        Đồ án II, GVHD: ThS. Nguyễn Thị Huế
    </div>
    """,
    unsafe_allow_html=True
)


if __name__ == "__main__":
    main()
