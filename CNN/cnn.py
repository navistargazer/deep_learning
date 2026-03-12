# 패딩을 도입해서 원본과 결과물의 좌표를 일치시킴
import numpy as np

# 1. 가상의 이미지
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# 2. 필터(kernel) : 3x3, 세로 윤곽선을 찾는 필터
kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

# 제로 패딩 적용(np.pad)
padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)

# 3. 출력 행렬의 크기 계산 : 출력크기 = 입력크기 - 필터크기 + 1 = 5 - 3 + 1 = 3
img_h, img_w = image.shape
kernel_h, kernel_w = kernel.shape
pad_h, pad_w = padded_image.shape
out_h = pad_h - kernel_h + 1
out_w = pad_w - kernel_w + 1
output = np.zeros((out_h, out_w))

# 4. 합성곱 연산(sliding window)
for h in range(out_h):
    for w in range(out_w):
        # 이미지에서 필터크기만큼 관심영역(ROI)를 잘라냄
        roi = padded_image[h:h+kernel_h, w:w+kernel_w]
        # 합성곱 : elementwise로 곱한뒤 합계를 구함
        output[h, w] = np.sum(roi * kernel)

# 결과는 3*3 행렬이므로 왜곡이 발생(원본의 왼쪽 윤곽선은 두번째 컬럼인데 결과에서는 첫번째 컬럼이 윤곽선처럼 인식됨)
print('original image', image, sep='\n')
print('padded image', padded_image, sep='\n')
print('\n패딩 적용 합성곱 결과', output, sep='\n')
