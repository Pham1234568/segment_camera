import yt_dlp
import cv2
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt") 
ube
url = 'https://www.youtube.com/watch?v=Fu3nDsqC1J0'
ydl_opts = {'format': 'best'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    video_url = info['url']


capture = cv2.VideoCapture(video_url)

if not capture.isOpened():
    print("Không thể mở luồng video.")
    exit()


while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

 
    results = model.predict(frame, task='segment')  
    print("RESULT")
    print(results)

    annotated_frame = results[0].plot()  
   
    cv2.imshow("YOLO Segmentation on Live Stream", annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()
