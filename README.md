# Color Detection
이미지 연산 방법을 통해 영역을 찾는 방법 중 두번째로 **객체의 색상을 이용하여 검출**하는 방법에 대해 알아 보겠습니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/nSDVn/btrRXeUi0uW/9xJTgPWcesfT0TmerS3kNk/img.gif" width="50%">
</div>

------

#### **Import packages**

```python
import cv2
import numpy as np
from scipy.spatial import distance as dist
import imutils
import matplotlib.pyplot as plt
```

#### **Function declaration**

Jupyter Notebook 및 Google Colab에서 이미지를 표시할 수 있도록 Function으로 정의

```python
def img_show(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()
```

Color 을 판단하는 Function (이 부분을 수정하여 특정 Color만 추출을 하는 기능을 구현할 수도 있습니다.)

```python
def color_label(image, c):
    # 윤곽선에 대한 마스크를 구성한 다음 영역에 대한 평균 L*a*b* 값을 계산
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)
    mask = cv2.erode(mask, None, iterations=2)
    mean = cv2.mean(image, mask=mask)[:3]
    
    # 정의한 색상 L*a*b* Array를 loop로 수행하여 L*a*b* 색상 값과 이미지 평균 사이의 거리를 계산하며 
    # 최소 거리의 L*a*b* 색상 값을 찾음
    min_dist = (np.inf, None)
 
    for (i, row) in enumerate(lab_array):
        d = dist.euclidean(row[0], mean)
        if d < min_dist[0]:
            min_dist = (d, i)
    
    return color_names[min_dist[1]]
```

#### **Load Image**

```python
cv2_image = cv2.imread('asset/images/color.jpg', cv2.IMREAD_COLOR)
img_show('original image', cv2_image)
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/bpeku1/btrR1gpxM8t/nmo075QhY1e3RMUM26Juy0/img.png" width="50%">
</div>

#### **Color Detection**

Key는 색상명, Value는 RGB값을 tuple로 색상 dictionary를 정의합니다.

```python
colors_dict = {"Red": (255, 0, 0),
               "Green": (0, 255, 0),
               "Blue": (0, 0, 255)}
```

RGB color space를 [L*a*b* color space](https://ko.wikipedia.org/wiki/CIELAB_색_공간)로 변환 (RGB나 HSV가 아닌 L*a*b* 를 이용하는 이유는 선택된 객체의 색상의 어떤 색상과 가까운지 측정하기 위해 유클리디안 거리(Euclidean Distance) 측정 방식을 사용하기 때문입니다.)

유클리디안 거리(Euclidean Distance) 혹은 유클리드 거리 측정 방식이라 불리는 이 알고리즘은 단순히 값들 간의 거리를 구하는 방법입니다. 간단히 파타고라스 정의와 같이 삼각형을 만들어서 계산하는 방법인데 다른 점은 여러차원의 거리를 계산 할 수 있다는 것입니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/bz97eZ/btrR2HUxuYZ/qt0LXkprE0r1j5evq4RNOk/img.png" width="30%">
</div>

```python
lab_array = np.zeros((len(colors_dict), 1, 3), dtype="uint8")
color_names = []
 
for (i, (name, rgb)) in enumerate(colors_dict.items()):
    lab_array[i] = rgb
    color_names.append(name)
    
lab_array = cv2.cvtColor(lab_array, cv2.COLOR_RGB2LAB)
```

아래 과정을 통해 이미지를 그레이스케일로 변환하고 노이즈를 줄이기 위한 이미지 블러링 후 이진화 합니다.

```python
resized = imutils.resize(cv2_image, width=640)
ratio = cv2_image.shape[0] / float(resized.shape[0])
blurred = cv2.GaussianBlur(resized, (5, 5), 0)
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
lab_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
 
img_show(['GaussianBlur', 'L*a*b* color', 'Threshold'], [blurred, lab_img, thresh])
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/d1BHZO/btrRXt47mOg/nPdRg0cd5NoxKkLkNsk1bK/img.png" width="50%">
</div>

이진화 이미지에서 윤곽선을 검출합니다.

```python
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
```

추출된 윤곽선을 기준으로 Color을 판단하여 이미지에 표시합니다.

```python
vis = cv2_image.copy()
 
# loop over the contours
for c in cnts:
    # cv2.moments를 이용하여 객체의 중심을 계산
    M = cv2.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    
    # 이미지에서 객체의 윤곽선과 Color를 표시
    color = color_label(lab_img, c)
 
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv2.drawContours(vis, [c], -1, (0, 255, 0), 10)
    cv2.circle(vis, (cX, cY), 20, (0, 255, 0), -1); 
    cv2.putText(vis, color, (cX-80, cY-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
```

Color를 표현한 이미지를 확인합니다.

```python
img_show('Color Detection', vis, figsize=(16,10))
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/ccW10W/btrR2GnRCu1/QvjZjQtPCqUykbfCpwBmi1/img.png" width="50%">
</div>

------

색상의 유사도를 이용한 방식 역시 단점은 있습니다. Color Detection을 위해 사용한 L*a*b* 에서 L*는 값의 밝기, a*는 빨강과 초록 중 어느쪽으로 치우쳤는지, *b 는 노랑과 파랑 중 어디로 치우쳤는지를 나타냅니다. L*a*b* 색 공간은 RGB나 CMYK가 표현할 수 있는 모든 색상을 포함하여 나타낼 수 있고 인간이 지각할 수 없는 색상도 나타낼 수 있습니다. 하지만 위에서 우리가 정의한 color_dict의 경계에 모호한 경우는 사람이 인지하는 색상의 분류와 다르게 판단 될 수도 있습니다.

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/PwOpl/btrR2H8cJWP/WZRBKaOuqytN8vCrMUyhDk/img.png" width="50%">
</div>

경계가 모호한 경우 잘못된 색으로 판단 될 수 있습니다. 색상을 좀 더 세분화 한다면 더 나은 결과를 얻을 수 있습니다.

```python
colors_dict = {"Red": (255, 0, 0),
               "Green": (0, 255, 0),
               "Blue": (0, 0, 255),
               "Lite blue": (118, 214, 255),
               "Moderate green": (146, 208, 80),
               "Dark green": (0, 176, 80),
               "Light red": (255, 126, 121),}
```

<div align="center">
  <img src="https://blog.kakaocdn.net/dn/VlAT1/btrR2FihNzQ/yVXH4ebktikQmkT8k4Da2K/img.png" width="50%">
</div>
