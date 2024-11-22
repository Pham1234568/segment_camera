import yt_dlp
import cv2
from ultralytics import YOLO

# Khởi tạo model YOLOv8 với nhiệm vụ phân đoạn
model = YOLO("yolo11n-seg.pt")  # Model YOLOv8 dành cho segmentation, bạn có thể thay đổi model tùy thích

# Lấy link phát trực tiếp từ YouTube
url = 'https://www.youtube.com/watch?v=Fu3nDsqC1J0'
ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    video_url = info['url']

# Sử dụng OpenCV để mở luồng trực tiếp
capture = cv2.VideoCapture(video_url)

# Kiểm tra xem luồng có mở thành công không
if not capture.isOpened():
    print("Không thể mở luồng video.")
    exit()

# Xử lý từng khung hình trong luồng video
while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Thực hiện phân đoạn trên khung hình
    results = model.predict(frame, task='segment')  # Dùng task='segment' cho phân đoạn
    print("RESULT")
    print(results)
    # Hiển thị kết quả trên khung hình
    annotated_frame = results[0].plot()  # Kết xuất kết quả phân đoạn lên khung hình

    # Hiển thị khung hình đã được gắn nhãn phân đoạn
    cv2.imshow("YOLO Segmentation on Live Stream", annotated_frame)

    # Nhấn 'q' để thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
capture.release()
cv2.destroyAllWindows()
