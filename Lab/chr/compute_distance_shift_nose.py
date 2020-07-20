from os import listdir
from os.path import isfile, join
import json
from scipy.spatial import distance
from statistics import mean

# json파일 폴더에서 180개의 json파일을 읽어 들여 각 파일에서 25개 keypoints의 (x, y) 위치정보를 추출
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

def euclid(v1, v2):
    zipped_lists = zip(v1, v2)
    
    result = [distance.euclidean(v1, v2) for (v1, v2) in zipped_lists]
    result = mean(result)
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


print("ameya_tadasan vs. kaustuk_tadasan:", euclid(at, kt))
print("ameya_vriksh vs. kaustuk_vriksh:", euclid(av, kv))

print("ameya_tadasan vs. kaustuk_vriksh:", euclid(at, kv))
print("ameya_vriksh vs. kaustuk_tadasan:", euclid(av, kt))

'''

ameya_tadasan vs. kaustuk_tadasan: 0.04167024025416529
ameya_vriksh vs. kaustuk_vriksh: 0.021478992006291466
ameya_tadasan vs. kaustuk_vriksh: 0.09570864672309816
ameya_vriksh vs. kaustuk_tadasan: 0.08175281486444648


'''
