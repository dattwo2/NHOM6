import streamlit as st
from deepface import DeepFace
import os
from model import Model
from utils.commons import read_image_from_bytes, xywh_to_xyxy
import numpy as np
st.set_page_config(page_title="Nhận dạng khuôn mặt", page_icon="🇻🇳")

@st.cache_data # Disable for Debugging
def find_in_database(image, distance_metric="cosine", max_distance=0.4):
    # Kiểm tra nếu image là đường dẫn hợp lệ
    if isinstance(image, str):
        if not os.path.exists(image):
            raise ValueError(f"Đường dẫn đến ảnh không hợp lệ: {image}")
    elif not isinstance(image, np.ndarray):
        raise TypeError(f"Dữ liệu image không hợp lệ: {type(image)}. Cần là đường dẫn ảnh hoặc mảng NumPy.")

    # Khởi tạo model với các tham số
    model = DeepFace.build_model("VGG-Face")  # Hoặc mô hình khác bạn đang sử dụng
    # Sử dụng DeepFace.find để tìm ảnh tương tự trong cơ sở dữ liệu
    results = DeepFace.find(
        img_path=image,  # Đảm bảo image là đường dẫn hợp lệ đến ảnh hoặc ảnh đã đọc
        db_path="path_to_database",  # Đảm bảo db_path là đúng
        model_name="VGG-Face",  # Sử dụng mô hình "VGG-Face" hoặc mô hình bạn chọn
        distance_metric=distance_metric,  # Sử dụng metric bạn muốn
        enforce_detection=True,  # Nếu muốn phát hiện khuôn mặt trong ảnh
        detector_backend="opencv",  # Có thể thử "mtcnn" nếu "opencv" không hoạt động tốt
    )

    # Lọc kết quả theo khoảng cách max_distance
    filtered_results = []
    for result in results:
        if result["distance"] <= max_distance:
            filtered_results.append(result)

    return filtered_results
if __name__ == "__main__":
    st.sidebar.title("Nhận dạng bằng khuôn mặt")
    st.sidebar.markdown("Đồ án chuyên ngành - Nhóm 16")
    
    st.title("Tìm kiếm thông tin bằng khuôn mặt")
        
    uploaded_file = st.file_uploader("Chọn một ảnh", type=['png', 'jpg'])
    if uploaded_file is None:
        uploaded_file = st.camera_input("Hoặc chụp ảnh")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        input_image = read_image_from_bytes(bytes_data)
        
        st.header("Ảnh đầu vào")
        st.image(bytes_data, use_column_width=True, channels="BGR")

        # Settings
        distance_metric = 'cosine'
        max_distance = 0.4

        if st.checkbox('Hiển thị cài đặt nâng cao'):
            st.header("Cài đặt")
            distance_metric = st.selectbox(
                'Phép đo khoảng cách',
                ('cosine', 'euclidean', 'euclidean_l2'))
            max_distance = st.slider("Khoảng cách tối đa", min_value=0.1, max_value=2.0, value=0.4, step=0.1)

        show_json = st.checkbox('Hiển thị đầu ra dạng json')

        if st.button("Tìm kiếm", type="primary", use_container_width=True):
            results = find_in_database(input_image, distance_metric, max_distance)

            st.header("Kết quả")

            if show_json:
                st.json(results)
            
            for i in range(len(results)):
                result = results[i]
                x1, y1, x2, y2 = xywh_to_xyxy(result["x"], result["y"], result["w"], result["h"])
                
                st.subheader(f"Face {i+1}")

                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(input_image[y1:y2, x1:x2], use_column_width=True, channels="BGR")
                
                with col2:
                    if result["found"]:
                        try:
                            st.info(f'**No.**: {result["No."]}')
                            st.info(f'**Full name**: {result["Full name"]}')
                            st.info(f'**Date of birth**: {result["Date of birth"]}')
                            st.info(f'**Sex**: {result["Sex"]}')
                            st.info(f'**Nationality**: {result["Nationality"]}')
                            st.info(f'**Place of origin**: {result["Place of origin"]}')
                            st.info(f'**Place of residence**: {result["Place of residence"]}')
                        except:
                            st.error("Dữ liệu thông tin của người này bị thiếu")
                            # Another case when there is a lack of the about of this person
                    else:
                        st.error("Không tìm thấy thông tin người này...")
            
            st.header("Kết thúc kết quả nhận dạng...")