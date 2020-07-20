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
        # (x, y) tuple로 만들어 리스트에 삽입
        it = iter(ary)
        ary = zip(it, it)
        keypoint_data += ary

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

ameya_tadasan vs. kaustuk_tadasan: 0.05270315704996472
ameya_vriksh vs. kaustuk_vriksh: 0.06992130281296693
ameya_tadasan vs. kaustuk_vriksh: 0.09609215334769194
ameya_vriksh vs. kaustuk_tadasan: 0.10482387564914232


'''
