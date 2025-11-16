"""
1. gemini api 사용
2. /mnt/ssd_hs/Dataset/lrs3/lrs3_video_seg24s/test/0Fi83BHQsMA/00002.mp4 불러오기
3. Instruction 제공해서 추론.


"""


import os
import google.generativeai as genai
# types를 직접 import할 필요가 없습니다.

# 1. API 키 설정
# 환경 변수 사용을 강력히 권장합니다.
# 예: export GOOGLE_API_KEY="YOUR_API_KEY" (Linux/macOS)
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# 테스트를 위해 직접 입력하는 경우 (권장하지 않음, 실제 키로 대체)
genai.configure(api_key="AIzaSyCDtY2ywf_0o7VY-_OO4b5Wryi4Xjug8eQ")

# `Client` 객체는 더 이상 직접 생성하지 않습니다.
# client = genai.Client(api_key=genai.get_default_api_key()) # 이 줄은 삭제 또는 주석 처리

video_file_name ="/mnt/ssd_hs/Dataset/lrs3/lrs3_video_seg24s/pretrain/0Amg53UuRqE/00001_01.mp4"
# "/mnt/ssd_hs/Dataset/lrs3/lrs3_video_seg24s/test/0Fi83BHQsMA/00002.mp4"

try:
    # 2. 비디오 파일 업로드
    # genai.upload_file() 함수를 직접 호출합니다.
    print(f"Uploading video file: {video_file_name}...")
    uploaded_file = genai.upload_file(
        path=video_file_name,
        mime_type='video/mp4'
    )
    print(f"File uploaded: {uploaded_file.uri}")

    # 3. 모델 초기화 및 콘텐츠 전달
    # genai.GenerativeModel()을 직접 사용합니다.
    model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20') # 'gemini-2.5-flash-preview-05-20' 대신 최신 버전 사용 권장

    response = model.generate_content(
        contents=[
            uploaded_file, # 업로드된 File 객체
            "Recognize what a person says by inferring from their lip movements."
        ]
    )

    print("\n--- Gemini Model Response ---")
    print(response.text)

    # 4. 사용 후 업로드된 파일 삭제 (권장)
    # 비용 절감을 위해 불필요한 파일을 삭제합니다.
    genai.delete_file(uploaded_file.name) # genai.delete_file() 함수를 직접 호출
    print(f"\nUploaded file {uploaded_file.name} deleted.")

except FileNotFoundError:
    print(f"오류: 비디오 파일 '{video_file_name}'을 찾을 수 없습니다. 경로를 확인해주세요.")
except Exception as e:
    print(f"\nAPI 요청 중 오류 발생: {e}")
    print("API 키가 올바른지, 네트워크 연결 상태를 확인해주세요.")
    print("비디오 파일의 크기나 형식, 내용이 모델의 제한을 초과하지 않는지 확인해주세요.")