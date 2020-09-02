from os import listdir
from os.path import isfile, join
import json
from scipy.spatial import distance
import math

'''
Pose Estimation. Metrics. by Alexander Stasiuk
https://medium.com/@masherov14/pose-estimation-metrics-844c07ba0a78
'''

# json파일 폴더에서 180개의 json파일을 읽어 (x, y) tuple이 들어있는 4500 (25 keypoints * 180 frame) 벡터 추출
def read_vec(json_path):
    json_path = json_path
    keypoint_data_jsons = [f for f in listdir(json_path) if isfile(join(json_path, f))]
    keypoint_data_jsons.sort()

    keypoint_data = []
    for keypoint_data_json in keypoint_data_jsons:
        #print(keypoint_data_json)
        with open(json_path+keypoint_data_json) as f:
            data = json.load(f)
        ary = data["people"][0]["pose_keypoints_2d"]
        del ary[2::3]

        # 코의 좌표 추출
        nose_x = ary[0]
        nose_y = ary[1]

        odd = ary[0::2]
        even = ary[1::2]

        # 코의 좌표가 (0, 0)이 되게끔 모든 keypoint 위치를 이동 (shift)
        origin_x = [x - nose_x for x in odd]
        origin_y = [y - nose_y for y in even]

        # 각 keypoint의 위치 정보를 (x, y)로 묶어서 list 안에 넣음
        keypoint_data += tuple(zip(origin_x, origin_y))

    return keypoint_data

# 사람 기준 길이의 diagonal에 해당하는 값을 base (length between nose and middle hip)의 형태로 구함
def get_base(v1, v2):
    # 스승과 학생의 base 길이를 각각 구함
    base1 = distance.euclidean(v1[0], v1[8]) # 스승
    base2 = distance.euclidean(v2[0], v2[8]) # 학생

    scaling_factor = base1 / base2
    
    # 두 base의 평균을 기준 길이(PDJ의 diagonal)로 정함
    base = (base1 + base2) / 2
    return scaling_factor, base


# OKS (Object Keypoint Similarity)를 계산
# OKS = exp(-1.0 * (di ** 2) / (2 * alpha * base ** 2))
def compute_pdj(v1, v2, alpha):
    scaling_factor, base = get_base(v1, v2)
    nose_to_hip_len = alpha * base

    oks = 0
    for i in range(len(v1)):
        # 학생(v2)의 좌표값에 대해 scaling_factor를 곱해 스승에 맞춤 
        v2_scaled = tuple(scaling_factor * val for val in v2[i])

        pointwise_dist = distance.euclidean(v1[i], v2_scaled)
        oks += math.exp(-1.0 * pointwise_dist ** 2 / (2 * nose_to_hip_len ** 2))

    result = oks / len(v1)
    return result

path_tt = "keypoint_data_json/teacher_tadasan_json/content/openpose/output/"
path_st = "keypoint_data_json/student_tadasan_json/content/openpose/output/"

path_tv = "keypoint_data_json/teacher_vriksh_json/content/openpose/output/"
path_sv = "keypoint_data_json/student_vriksh_json/content/openpose/output/"

tt = read_vec(path_tt)
print(len(tt))

st = read_vec(path_st)
print(len(st))

tv = read_vec(path_tv)
print(len(tv))

sv = read_vec(path_sv)
print(len(sv))

alpha = 0.10

print("teacher_tadasan vs. student_tadasan:", compute_pdj(tt, st, alpha))
print("teacher_vriksh vs. student_vriksh:", compute_pdj(tv, sv, alpha))
print("")
print("teacher_tadasan vs. student_vriksh:", compute_pdj(tt, sv, alpha))
print("teacher_vriksh vs. student_tadasan:", compute_pdj(tv, st, alpha))

