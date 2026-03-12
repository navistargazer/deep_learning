import numpy as np

# XOR을 사용한 2층 신경망
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])
np.random.seed(42)

def fn_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fn_sigmoid_deravative(x):
    return x * (1 - x)

# 은닉층 뉴런의 갯수
hidden_neurons = 4

W1 = np.random.randn(2, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.randn(hidden_neurons, 1)
b2 = np.zeros((1, 1))
lr = 0.5  # 학습이 잘 안되면 0.1 ~ 1.0 사이로 조절 가능

for epoch in range(1000):
    # 1. 순전파
    y1 = x @ W1 + b1
    out1 = fn_sigmoid(y1)
    y2 = out1 @ W2 + b2
    out2 = fn_sigmoid(y2)
    
    loss = np.mean(0.5 * (y_xor - out2) ** 2)
    
    # 2. 역전파 (기울기 계산)
    
    # --- 출력층 (W2) 기울기 계산 ---
    dL_dout2 = out2 - y_xor
    dout2_dy2 = fn_sigmoid_deravative(out2)
    delta2 = dL_dout2 * dout2_dy2
    
    dL_dw2 = out1.T @ delta2
    dL_db2 = np.sum(delta2, axis=0, keepdims=True)
    
    # [수정] 여기서 W2를 바로 업데이트하면 안 됩니다! 
    # 아직 W1의 오차를 구할 때 '현재의 W2'가 필요하기 때문입니다.

    # --- 은닉층 (W1) 기울기 계산 ---
    # W2가 수정되기 전(Pre-update) 상태여야 정확한 오차 전파가 가능합니다.
    dL_dout1 = delta2 @ W2.T 
    dout1_dy1 = fn_sigmoid_deravative(out1)
    delta1 = dL_dout1 * dout1_dy1
    
    dL_dW1 = x.T @ delta1
    dL_db1 = np.sum(delta1, axis=0, keepdims=True)

    # 3. 경사하강법 (가중치 업데이트)
    # [수정] 모든 기울기 계산이 끝난 후, 맨 마지막에 업데이트합니다.
    W2 -= dL_dw2 * lr
    b2 -= dL_db2 * lr
    W1 -= dL_dW1 * lr
    b1 -= dL_db1 * lr

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

print("\n[학습 결과]")
print(out2)
print("예측값 (0.5 기준): \n", (out2 > 0.5).astype(int))