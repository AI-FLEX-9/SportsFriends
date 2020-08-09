## ForSportsFriends
----------------------------------------------------------------
This is the study space for the SportsFriends project.

## 1. Study Keyword
1.  OpenPose

     OpenposeVideo.ipynb 샘플 코드 참고

      ```python
          !cd openpose && ./build/examples/openpose/openpose.bin \
          --keypoint_scale 3 --frame_step 5 --video ../video.mp4 --write_json\
           ./output/ --display 0 --write_video ../openpose.avi
      ```

     이처럼 openpose.bin에 각종 옵션(flag)을 줄수 있음

    - 동영상 대신 카메라 영상을 입력으로 넣을 경우   
      --`video ../video.mp4 `대신에 `--camera -1` 로 설정   
    - keypoint 의 (x, y) 좌표값의 range를 정하려면   
      `--keypoint_scale 0` : original source resolution   
      `--keypoint_scale 3` : scale it in the range [0,1], where (0,0) would be the top-left corner of the image, and (1,1) the bottom-right one   
    - keypoint 추출하려는 동영상의 프레임 수를 줄이고 싶을 경우
      `--frame_step 5` (0, 5, 10, 15, ... 이렇게 frame을 선별하여 추출.)


2. Object Detection Evaluation  
    - Percentage of Detected Joints (PDJ) :sparkling_heart:

      <p align="center"><img width="30%" src="images/math_pdj.gif" /></p>

       where    
         di :Euclidean distance between  keypoints    
         s : scale factor   
         B : base element   

    - Object Keypoint Similarity (OKS) :seedling:
       <p align="center"><img width="30%" src="images/math_oks.gif" /></p>

       where  
        di: Euclidean distance between  keypoints     
        s : scale factor    
        ki: base element of each keypoint   

## 2. Related Articles & Repos


## 3. Dataset
