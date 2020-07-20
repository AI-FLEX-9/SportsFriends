from os import listdir
from os.path import isfile, join
import json
from scipy.spatial import distance
import math

# json파일 폴더에서 180개의 json파일을 읽어 (x, y) tuple이 들어있는 4500 (25 keypoints * 180 frame) 벡터 추출
def read_vec(json_path):
    json_path = json_path
    json_files = [f for f in listdir(json_path) if isfile(join(json_path, f))]
    json_files.sort()

    keypoint_data = []
    for json_file in json_files:
        #print(json_file)
        with open(json_path+json_file) as f:
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

# PDJ (Percentage of Detected Joints) 계산식 중 diagonal에 해당하는 값을 base (length between nose and middle hip)의 형태로 구함
def get_base(v1, v2):
    # 스승과 학생의 base 길이를 각각 구함
    base1 = distance.euclidean(v1[0], v1[8]) # 스승
    base2 = distance.euclidean(v2[0], v2[8]) # 학생
    
    # 두 base의 평균을 기준 길이(PDJ의 diagonal)로 정함
    result = (base1 + base2) / 2
    return result


# OKS (Object Keypoint Similarity)를 계산
# OKS = exp(-1.0 * (di ** 2) / (2 * alpha * base ** 2))
def compute_pdj(v1, v2, alpha):
    nose_to_hip_len = alpha * get_base(v1, v2)

    oks = 0
    for i in range(len(v1)):
        pointwise_dist = distance.euclidean(v1[i], v2[i])
        oks += math.exp(-1.0 * pointwise_dist ** 2 / (2 * nose_to_hip_len ** 2))

    result = oks / len(v1)
    return result


path_at = "json_file/ameya_tadasan/ameya_tadasan_json/content/openpose/output/"
path_kt = "json_file/kaustuk_tadasan/kaustuk_tadasan_json/content/openpose/output/"

path_av = "json_file/ameya_vriksh/ameya_vriksh_json/content/openpose/output/"
path_kv = "json_file/kaustuk_vriksh/kaustuk_vriksh_json/content/openpose/output/"

at = read_vec(path_at)
print(len(at))
#print(at)

kt = read_vec(path_kt)
print(len(kt))

av = read_vec(path_av)
print(len(av))

kv = read_vec(path_kv)
print(len(kv))

alpha = 0.05
alpha = 0.08
alpha = 0.1
alpha = 0.02

print("ameya_tadasan vs. kaustuk_tadasan:", compute_pdj(at, kt, alpha))
print("ameya_vriksh vs. kaustuk_vriksh:", compute_pdj(av, kv, alpha))

print("ameya_tadasan vs. kaustuk_vriksh:", compute_pdj(at, kv, alpha))
print("ameya_vriksh vs. kaustuk_tadasan:", compute_pdj(av, kt, alpha))


'''

# base = 0.05

ameya_tadasan vs. kaustuk_tadasan: 0.3883285341813686
ameya_vriksh vs. kaustuk_vriksh: 0.6260041766873111
ameya_tadasan vs. kaustuk_vriksh: 0.24443818287106514
ameya_vriksh vs. kaustuk_tadasan: 0.2201549375829494

# base = 0.08

ameya_tadasan vs. kaustuk_tadasan: 0.4720479550029235
ameya_vriksh vs. kaustuk_vriksh: 0.7489350960360074
ameya_tadasan vs. kaustuk_vriksh: 0.3661227570383297
ameya_vriksh vs. kaustuk_tadasan: 0.2841280968781947

# base = 0.1

ameya_tadasan vs. kaustuk_tadasan: 0.5123015136631814
ameya_vriksh vs. kaustuk_vriksh: 0.7958715022298191
ameya_tadasan vs. kaustuk_vriksh: 0.429864415504008
ameya_vriksh vs. kaustuk_tadasan: 0.33548209155983333

# base = 0.02

ameya_tadasan vs. kaustuk_tadasan: 0.19927770104438375
ameya_vriksh vs. kaustuk_vriksh: 0.33182506708160125
ameya_tadasan vs. kaustuk_vriksh: 0.11242642938951239
ameya_vriksh vs. kaustuk_tadasan: 0.16051129829340371

'''
