import numpy as np

# XOR을 사용한 2층 신경망(순전파, 오류계산, 역전파)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])
# 랜덤 고정
np.random.seed(42)
# 히든 뉴런의 갯수
hidden_size = 4

# 은닉층 함수 : ReLU
def fn_relu(X):
    return np.maximum(0, X)

# ReLU 미분
def fn_relu_deravative(X):
    return np.where(X > 0, 1, 0)

# 출력층 함수 : Sigmoid
def fn_sigmoid(X):
    return 1 / (1 + np.exp(-X))

# 시그모이드 미분
def fn_sigmoid_deravative(X):
    Y = fn_sigmoid(X)
    return Y * (1 - Y)

# 사용할 함수
act_fn = fn_relu
act_fn_deravative = fn_sigmoid_deravative


# 입력층2 -> 은닉층(hidden_size) -> 출력층(1)
W1 = np.random.randn(2, hidden_size)   # 2 -> 2
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, 1)   # 2 -> 1 
b2 = np.zeros((1, 1))
lr = 0.3

# 학습
for epoch in range(1001):
    # 순전파
    Z1 = X @ W1 + b1        # (4, 2)
    A1 = act_fn(Z1)   # 은닉층 출력 (4, 2)
    Z2 = A1 @ W2 + b2     # (4, 1)
    A2 = fn_sigmoid(Z2)   # 최종 출력 (4, 1)
    # 오차 계산(MSE)
    loss = np.mean(0.5 * (Y - A2) ** 2)     # 미분 편의를 위해 1/2을 미리 곱해둠, (4, 1) -> 1(mean)
    
    # 역전파 : 오차를 줄이기 위한 가중치 변화 = 오차를 가중치로 편미분 (dL/dW)
    # 1. 은닉층으로 역전파 : 출력층으로 도달하기 위해 사용된 가중치가 오차에 얼마나 영향을 주었는가? (dL/dW2)
    # chain rule : dL/dW2 = dL/A2 * dA2/dZ2 * dZ2/dW2
    # dL/A2 : 오차에 활성화함수값이 끼친 영향
    dL_dA2 = A2 - Y     # (4, 1)
    # dA2/dZ2 : 활성화함수값에 은닉층 함수값이 끼친 영향
    dA2_dZ2 = fn_sigmoid_deravative(Z2) # (4, 1)
    # dL/dZ2 : 출력층 내부에서 일어난 변화가 오차에 끼친 영향 : elementwise
    delta2 = dL_dA2 * dA2_dZ2  # (4, 1)
    # 출력값을 가중치로 편미분 dZ2/dW2 = A1  # (4, 2)
    # 오차값에 가중치가 얼마나 기여하였는가?를 알기 위해 입력값 @ 최종오차신호 (역전파로 가중치를 수정하려면 가중치의 차원에 맞춰줌)
    dL_dW2 = A1.T @ delta2    # (2, 4) @ (4, 1) = (2, 1)
    # 편향의 경우 dL/db2 = dL/dA2 * dA2/dZ2 * dZ2/db2 = delta2 * 1 => 모든 요소에 영향을 끼쳤으므로 모든 행의 합계를 구함
    dL_db2 = np.sum(delta2, axis=0, keepdims=True)
   
    # 2. 입력층으로 역전파
    # chain rule : dL/dW1 = dL/dZ2 * dZ2/dA1 * dA1/dZ1 * dZ1/dW1
    # dL/dZ2는 위에 구했음 (4, 1)
    # dZ2/dA1 = W2  # (2, 1)
    # dL/dA1 = dL/dZ2 @ dZ2/dA1
    dL_dA1 = delta2 @ W2.T  # (4, 1) @ (1, 2) = (4, 2)
    # dA1/dZ1
    dA1_dZ1 = act_fn_deravative(Z1)  # (4, 2)
    # dL/dZ1 = dL/dA1 * dA1/dZ1 : 은닉층 내부에서 일어난 변화가 오차에 끼친 영향
    delta1 = dL_dA1 * dA1_dZ1  # (4, 2)
    # dL/dW1 = dL/dZ1 * dZ1/dW1
    # dZ1/dW1 = X  # (4, 2) => X.T # (2, 4)
    dL_dW1 = X.T @ delta1  # (2, 4) @ (4, 2) = (2, 2)
    # 편향 dL/db1 = dL/dA1 * dA1/dZ1 * dZ1/db1 = delta1 * 1
    dL_db1 = np.sum(delta1, axis=0, keepdims=True)
    
     # 경사하강법 : 가중치에서 dL/dw2를 빼줌(학습률 곱해서)
    W2 -= dL_dW2 * lr   # (2, 1)
    b2 -= dL_db2 * lr
    W1 -= dL_dW1 * lr  # (2, 2)
    b1 -= dL_db1 * lr

    pred_Y = (A2 > 0.5).astype(int)
    if (pred_Y == Y).all():
        print(f'{epoch}번 학습 후 정답 예측')
        print("\n[학습 결과]")
        print(A2)
        print("예측값 (0.5 기준): \n", pred_Y)
        break

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss}")




