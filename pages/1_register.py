import os
import json
import datetime
from pickle import dump, load

import streamlit as st

from model import Model
from utils.commons import read_image_from_bytes, save_image


st.set_page_config(page_title="Đăng ký", page_icon="🇻🇳")

if "images" not in st.session_state:
    st.session_state["images"] = []

if "picture" not in st.session_state:
    st.session_state["picture"] = []

if "submitted" not in st.session_state:
    st.session_state["submitted"] = False

if "folder_name_exist" not in st.session_state:
    st.session_state["folder_name_exist"] = None


def check_info_exist(db_path, no) -> (bool, str):
    """
    Kiểm tra thông tin tồn tại trong thư mục và tệp JSON.

    Args:
        db_path (str): Đường dẫn tới thư mục cơ sở dữ liệu.
        no (str): Số cần kiểm tra.

    Returns:
        tuple: (info_exist (bool), folder_name_exist (str))
    """
    info_exist = False
    folder_name_exist = None

    # Kiểm tra thư mục có tồn tại không
    if not os.path.exists(db_path) or not os.path.isdir(db_path):
        raise FileNotFoundError(f"Thư mục '{db_path}' không tồn tại hoặc không phải là thư mục hợp lệ.")

    for folder_name in os.listdir(db_path):
        folder_path = os.path.join(db_path, folder_name)

        # Bỏ qua nếu không phải là thư mục
        if not os.path.isdir(folder_path):
            continue

        # Kiểm tra tệp 'about.json' trong thư mục
        info_path = os.path.join(folder_path, "about.json")
        if os.path.isfile(info_path):
            try:
                with open(info_path, mode='r', encoding='utf-8') as f:  # Đọc tệp với mã hóa UTF-8
                    info = json.load(f)

                    # So sánh thông tin 'No.'
                    if str(info.get('No.')) == str(no):
                        info_exist = True
                        folder_name_exist = folder_name
                        break
            except json.JSONDecodeError:
                raise ValueError(f"Tệp '{info_path}' không phải là JSON hợp lệ.")
            except Exception as e:
                raise RuntimeError(f"Lỗi khi đọc tệp '{info_path}': {e}")

    return info_exist, folder_name_exist
def save_to_database(folder_path, id_number, images, information):
    # Save images
    for i in range(len(images)):
        image = images[i]
        save_image(f'{folder_path}/ID{id_number}_{i}.jpg', image)
    # Save infomarion
    with open(f'{folder_path}/about.json', mode='w', encoding='utf-8') as f:
        json.dump(information, f, ensure_ascii=False, indent=4)

def represent(images, images_paths):
    model = Model()
    representations = model.represent(images, images_paths)
    return representations

def register_information(images):
    with st.form("registration_form"):
        st.markdown(":red[**Lưu ý:**] Yêu cầu nhập đầy đủ thông tin")
        no = st.text_input("Mã định danh", placeholder="Nhập số định danh trên CCCD")
        full_name = st.text_input("Họ và tên", placeholder="Nhập họ và tên đầy đủ của bạn")
        date_of_birth = st.date_input("Ngày sinh", datetime.date(2002, 1, 1), datetime.date(1900, 1, 1), datetime.date(2023, 1, 1), format="DD/MM/YYYY")
        sex = st.selectbox("Giới tính", options=["Nam", "Nữ", "Khác"])
        nationality = st.text_input("Quốc tịch", placeholder="Nhập quốc tịch của bạn")
        place_of_origin = st.text_input("Quê quán", placeholder="Nhập quê quán")
        place_of_residence = st.text_input("Nơi thường trú", placeholder="Nhập địa chỉ thường trú")

        information = {
            "No.": no,
            "Full name": full_name,
            "Date of birth": str(date_of_birth),
            "Sex": sex,
            "Nationality": nationality,
            "Place of origin": place_of_origin,
            "Place of residence": place_of_residence
        }

        submit = st.form_submit_button("Đăng ký", type="primary", use_container_width=True)

    if submit:
        # Kiểm tra dữ liệu đã điền đầy đủ
        if not (no and full_name and date_of_birth and sex and nationality and place_of_origin and place_of_residence):
            st.error("Vui lòng điền đầy đủ thông tin!")
            st.stop()

        db_path = 'database'
        st.session_state["submitted"] = True
        # Kiểm tra xem mã định danh đã tồn tại trong cơ sở dữ liệu chưa
        info_exist, folder_path_exist = check_info_exist(db_path, no)

        if info_exist:
            st.session_state["folder_name_exist"] = folder_path_exist
            # Cho phép cập nhật thông tin hoặc không
            st.warning("Thông tin đã tồn tại trong cơ sở dữ liệu! Bạn có muốn cập nhật không?")
            # Hiển thị nút "Cập Nhật" và "Không"
            st.button("Cập nhật", key="update", use_container_width=True)
            st.button("Huỷ bỏ", key="cancel", use_container_width=True)
        else:
            # Đăng ký thông tin
            folder_count = 0
            for path in os.listdir(db_path):
                if os.path.isdir(os.path.join(db_path, path)):
                    folder_count += 1
            id_number = folder_count
            folder_path = f'{db_path}/ID{id_number}'
            # Create folder
            os.mkdir(folder_path)
            save_to_database(folder_path, id_number, images, information)
            # Update saved representations
            saved_representations_path = f'{db_path}/representations_arcface.pkl'
            if os.path.exists(saved_representations_path):
                # with open()
                # os.unlink(f'{db_path}/representations_arcface.pkl')
                with open(saved_representations_path, "rb") as f:
                    saved_representations = load(f)
                    images_paths = []
                    for i in range(len(images)):
                        images_paths.append(f'{folder_path}/ID{id_number}_{i}.jpg')
                    representations = represent(images, images_paths)
                    saved_representations.extend(representations)
                    # st.write(saved_representations)
                    with open(saved_representations_path, "wb") as f:
                        dump(saved_representations, f)

            st.success("Thông tin nhân dạng đã được đăng ký thành công!")

    if st.session_state["submitted"] and st.session_state["update"]:
        if st.session_state["folder_name_exist"] is None:
            st.stop()

        db_path = 'database'
        folder_name_exist = st.session_state["folder_name_exist"]
        folder_path = os.path.join(db_path, folder_name_exist)
        id_number = int(folder_name_exist[2:]) #IDxx

        # Xoá thông tin cũ
        for filename in os.listdir(os.path.join(folder_path)):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        # Cập nhật thông tin mới
        save_to_database(folder_path, id_number, images, information)
        # Remove saved representations
        if os.path.exists(f'{db_path}/representations_arcface.pkl'):
            os.unlink(f'{db_path}/representations_arcface.pkl')
        st.success("Cập nhật thông tin thành công!")

    if st.session_state["submitted"] and st.session_state["cancel"]:
        st.warning("Đã huỷ cập nhật thông tin")

@st.cache_data
def filter_uploaded_images(uploaded_files):
    model = Model()
    valid_images = []
    invalid_reports : str = []

    for image_file in uploaded_files:
        bytes_data = image_file.getvalue()
        image = read_image_from_bytes(bytes_data)
        face_objs = model.detect_faces(image)
        if len(face_objs) == 1:
            valid_images.append(image)
        elif len(face_objs) > 1:
            invalid_reports.append(f"**Lỗi** ảnh **{image_file.name}:** chứa nhiều hơn một khuôn mặt trong khung hình")
        else:
            invalid_reports.append(f"**Lỗi** ảnh **{image_file.name}:** không tìm thấy khuôn mặt trong khung hình")

    return valid_images, invalid_reports

if __name__ == '__main__':
    st.sidebar.title("Nhận dạng bằng khuôn mặt")
    st.sidebar.markdown("Đồ án chuyên ngành - Nhóm 16")

    st.title("Đăng ký thông tin nhận dạng")

    st.header("Tải ảnh lên")
    st.markdown(":red[**Lưu ý:**] Bắt buộc mỗi ảnh chỉ có một khuôn mặt")

    uploaded_files = st.file_uploader("Thêm một hoặc nhiều ảnh", type=['png', 'jpg'], accept_multiple_files=True, key='uploaded_images')

    if st.button("Chụp ảnh"):
         # Chụp ảnh từ webcam
         st.camera_input("Take a picture", key="picture")

    if st.session_state["picture"]:
        st.session_state["uploaded_images"].append(st.session_state["picture"])

    if "uploaded_images" not in st.session_state or len(st.session_state["uploaded_images"]) == 0:
        st.stop()

    valid_images, invalid_reports = filter_uploaded_images(uploaded_files)

    if len(invalid_reports):
        for report in invalid_reports:
            st.error(report)

    if len(valid_images) == 0:
        st.stop()

    # st.write(st.session_state)

    st.header("Ảnh đã được tải lên")
    captions = [f'Ảnh {i+1}' for i in range(len(valid_images))]
    image_width = max(200, 600 // len(valid_images))
    st.image(valid_images, caption=captions, width=image_width, channels="BGR")

    st.header("Nhập thông tin")
    register_information(valid_images)
