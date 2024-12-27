import streamlit as st


st.set_page_config(page_title="Giới thiệu", page_icon="🇻🇳")

st.sidebar.title("Nhận dạng bằng khuôn mặt")
st.sidebar.markdown("Đồ án chuyên ngành - Nhóm 16")

st.title('Giới thiệu về hệ thống nhận dạng bằng khuôn mặt')
st.header('Giới thiệu tổng quan về chức năng và nghĩa')
st.subheader('Các chức năng chính gồm:')
st.write('Chức năng tìm kiếm: Tìm kiếm khuôn mặt và thông tin tương ứng với đối tượng người dùng chỉ định, '
         'trả kết quả về là ảnh kèm theo thông tin của đối tượng và file json nếu muốn.\n\n'
         'Chức năng đăng ký nhận dạng: Đăng ký thông tin của một đối tượng mới, chưa có trong cơ sở dữ liệu.')
tab1, tab2, tab3 = st.tabs(["Chức năng tìm kiếm", "Chức năng đăng ký nhận dạng", "Giới thiệu về nhóm"])

with tab1:
    st.header("Chức năng tìm kiếm")
    st.subheader('1. Chụp ảnh từ cam')
    st.image('images/search/1.jpg', caption='1. Chụp ảnh từ cam')
    st.subheader('2. Hoặc tải ảnh lên')
    st.image('images/search/2.jpg', caption='2. Hoặc tải ảnh lên')
    st.subheader('3. Hiển thị ảnh đầu vào')
    st.image('images/search/3.jpg', caption='3. Hiển thị ảnh đầu vào')
    st.subheader('4. Cài đặt mô hình')
    st.image('images/search/4-1.jpg')
    st.image('images/search/4-2.jpg', caption='4. Cài đặt mô hình')
    st.subheader('5. Kết quả nhận dạng khuôn mặt')
    st.image('images/search/5-1.jpg')
    st.image('images/search/5-2.jpg')
    st.image('images/search/5-3.jpg')
    st.image('images/search/5-4.jpg')
    st.image('images/search/5-5.jpg', caption='5. Kết quả nhận dạng khuôn mặt')
    st.subheader('6. Hiển thị đầu ra dạng json file')
    st.image('images/search/6.jpg', caption='6. Hiển thị đầu ra dạng json file')

with tab2:
   st.header('Chức năng đăng ký nhận dạng: ')
   st.subheader('1. Tải ảnh lên')
   st.image('./images/register/1.jpg', caption='1. Tải ảnh lên')
   st.subheader('2.1. Lựa chọn 1 hoặc nhiều ảnh')
   st.image('./images/register/2-2.jpg')
   st.image('./images/register/2-1.jpg', caption='2.1. Lựa chọn 1 hoặc nhiều ảnh')
   st.subheader('2.2. Hệ thống sẽ phân tích để hiển thị ảnh hợp lệ (Nếu không hợp lệ thì sẽ có thông báo như hình bên dưới)')
   st.image('./images/register/4.jpg', caption='2.2. Ảnh không hợp lệ')
   st.subheader('3. Điền thông tin cho ảnh')
   st.image('./images/register/3-1.jpg')
   st.subheader('4.1. Đăng ký thành công')
   st.image('./images/register/3-2.jpg')
   st.subheader('4.2. Nếu thông tin đăng ký đã tồn tại')
   st.image('./images/register/3-3.jpg')
   st.subheader('4.3. Cập nhật thông tin')
   st.image('./images/register/3-4.jpg')
   st.subheader('4.4. Huỷ cập nhật thông tin')
   st.image('./images/register/3-5.jpg')

with tab3:
    st.header('Giới thiệu về nhóm')
    st.subheader('Lê Bá Việt Anh - 2020601175')
    st.subheader('Lý Thành Lâm - 2020600571')
    st.subheader('Nguyễn Trọng Hoàng - 2020600594')

