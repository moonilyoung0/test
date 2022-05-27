import urllib.request

def save_video(video_url) :
    savename = 'test.mp4'

    urllib.request.urlretrieve(video_url,savename)
    print("저장완료")

save_video("https://firebasestorage.googleapis.com/v0/b/speakingsignlang-lab-44591.appspot.com/o/motion_detection%2F157?alt=media&token=0696f4e9-e3c2-49d0-b974-012f4d89d16c")