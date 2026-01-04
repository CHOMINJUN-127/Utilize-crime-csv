from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- [1] 시각화 결과 (학습 과정) ---
plt.figure(figsize=(12, 5))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='학습 정확도 (Train Acc)')
plt.plot(history.history['val_accuracy'], label='검증 정확도 (Val Acc)')
plt.title('모델 정확도 변화')
plt.xlabel('Epoch')
plt.legend()

# 손실(Loss) 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='학습 손실 (Train Loss)')
plt.plot(history.history['val_loss'], label='검증 손실 (Val Loss)')
plt.title('모델 손실 변화')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# --- [2] 점수 결과 ---
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n[최종 테스트 점수]\n정확도: {acc*100:.2f}%")
print(f"손실값: {loss:.4f}")

# --- [3] 예측 테스트 및 혼동 행렬 시각화 ---
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# 유니크한 라벨 및 클래스 이름 준비
unique_labels_in_test = np.unique(y_test)
target_names_for_report = encoder.inverse_transform(unique_labels_in_test)

# 분류 리포트 출력
print("\n[분류 예측 결과 보고서]")
print(classification_report(y_test, y_pred, labels=unique_labels_in_test, target_names=target_names_for_report, zero_division=0))

# 혼동 행렬 시각화
plt.figure(figsize=(18, 12)) # 가독성 확보를 위해 이미지 크기 유지
cm = confusion_matrix(y_test, y_pred, labels=unique_labels_in_test)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names_for_report, yticklabels=target_names_for_report)

plt.title('예측 결과 혼동 행렬 (Confusion Matrix)')
plt.xlabel('예측한 범죄')
plt.ylabel('실제 범죄')

#  X축 레이블을 수평(0도)으로 설정하여 기울임을 제거
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()
