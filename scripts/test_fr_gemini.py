import os
import google.generativeai as genai
from moviepy.editor import VideoFileClip
from PIL import Image
import io
import base64 # 인라인 데이터 전송 시 필요

# 1. API 키 설정 (환경 변수 사용 권장)
genai.configure(api_key="AIzaSyCDtY2ywf_0o7VY-_OO4b5Wryi4Xjug8eQ")
video_file_name = "/mnt/ssd_hs/Dataset/lrs3/lrs3_video_seg24s/pretrain/0Amg53UuRqE/00001_01.mp4"
output_image_dir = "extracted_frames" # 추출된 이미지를 저장할 디렉토리

# 2. 추출할 프레임 설정
FPS_TO_EXTRACT = 25 # 초당 25프레임 추출

# 3. 모델 설정
model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20') # 비디오 이해에 더 강력한 Pro 모델 사용 권장

# 출력 이미지 디렉토리 생성
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

try:
    # 4. 비디오 클립 로드
    clip = VideoFileClip(video_file_name)
    duration = clip.duration # 비디오 총 길이 (초)
    print(f"Video duration: {duration:.2f} seconds")

    frames = []
    print(f"Extracting frames at {FPS_TO_EXTRACT} FPS...")

    # 5. 비디오에서 프레임 추출 및 Image 객체로 저장
    # 각 초에 대해 지정된 FPS_TO_EXTRACT 만큼 프레임 추출
    for i in range(int(duration * FPS_TO_EXTRACT)):
        # t = i / FPS_TO_EXTRACT: 프레임이 추출될 정확한 시간 (초)
        current_time = i / FPS_TO_EXTRACT
        if current_time >= duration: # 비디오 길이를 초과하지 않도록 방지
            break

        # moviepy의 get_frame 메서드로 이미지 배열(NumPy array) 추출
        frame_array = clip.get_frame(current_time)

        # NumPy array를 PIL Image 객체로 변환
        image = Image.fromarray(frame_array)
        frames.append(image)

        # (선택 사항) 추출된 프레임을 파일로 저장하여 확인
        # image.save(os.path.join(output_image_dir, f"frame_{i:04d}.jpg"))
        # print(f"Saved frame_{i:04d}.jpg at {current_time:.2f}s")

    clip.close() # 비디오 클립 닫기

    print(f"Total {len(frames)} frames extracted.")

    if not frames:
        print("No frames extracted. Check video file or duration.")
    else:
        # 6. 프롬프트 생성 및 이미지 추가
        # Gemini 모델은 PIL Image 객체를 직접 contents 리스트에 받을 수 있습니다.
        contents_parts = [
            "Recognize what a person says by inferring from their lip movements."
        ]
        contents_parts.extend(frames) # 추출된 모든 Image 객체 추가

        print("\nSending frames to Gemini model...")
        response = model.generate_content(
            contents=contents_parts
        )

        print("\n--- Gemini Model Response ---")
        print(response.text)

except FileNotFoundError:
    print(f"오류: 비디오 파일 '{video_file_name}'을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"\nAPI 요청 중 오류 발생: {e}")
    print("API 키가 올바른지, 네트워크 연결 상태를 확인해주세요.")
    print("비디오 파일의 크기, 프레임 수, 또는 내용이 모델의 제한을 초과하지 않는지 확인해주세요.")
    print(f"자세한 오류: {e}")