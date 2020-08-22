# SportsFriends
SportsFriends is a system that can grade sports activities using Pose Estimation AI technology.

# 1. Input
Dataset 
1) Training dataset 
-	https://archive.org/details/YogaVidCollected 
-	88 videos, a total of about 111,750 frames, 1h 6min 5s, at 30 frames/s, more than 45 s per video in an indoor environment 
2) AI Hub
- http://aihub.or.kr/aidata/138 의 VideoAction20.tar(86.7GB)
- Yoga : 5~10초 길이
3) LHJ's yoga video 
- Yoga : 2분 길이

# 2. Output

# 3. Model 생성

## OpenPose 코드 적용 
- [Openpose Video Test --hand, Data/Bhumi_Trik.mp4, OpenPoseVideo.ipynb](OpenPoseVideo.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github//AI-FLEX-9/SportsFriends/blob/master/Code/OpenPoseVideo.ipynb)


# 4. Model 최적화 / 모바일 기기 배포
- 정확도에 영향을 주지 않으면;서 모델크기를 줄이고 속도를 높이는 방법
- 모델 계산 성능 상세분석
- 휴대전화(iOS/Androoid), 브라우저에서 모델 실행

# 5. WEB test
 - 결과물을 WEB 페이지에 test 배포
 - https://sportsfriends.netlify.com
