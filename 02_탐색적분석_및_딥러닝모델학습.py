import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import platform
from matplotlib import font_manager, rc
import os
import numpy as np

# =========================================================
# **[오류 수정] 강력한 한글 폰트 설정 (Colab/Windows/Mac 호환)**
# =========================================================
plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지

if platform.system() == 'Darwin':  # Mac OS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
elif 'Linux' in platform.system():  # Google Colab, Linux
    try:
        # Colab에서 Nanum Gothic 폰트 설치 및 설정
        !sudo apt-get install -y fonts-nanum > /dev/null 2>&1
        !fc-cache -fv > /dev/null 2>&1
        font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
    except:
        print("경고: Linux 환경 폰트 설정 실패. 한글 깨짐 가능성 있음.")
# =========================================================

# --- [1] 필요한 데이터 시각화 ---
# 범죄 대분류별 총 발생 건수 시각화
df['총발생건수'] = df.iloc[:, 2:].sum(axis=1) # 2번째 컬럼부터 끝까지 합산
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='총발생건수', y='범죄대분류', ci=None)
plt.title('범죄 대분류별 총 발생 건수')
plt.xlabel('총 발생 건수')
plt.ylabel('범죄 대분류')
plt.show()

# --- [2] 레이블 처리 & 정규화(Scale) 처리 ---

# 1. 단일 샘플을 가진 클래스 식별 및 제거
# y_data_raw를 사용하여 전체 클래스에 대한 인코딩을 수행합니다.
le_full = LabelEncoder()
y_encoded_full = le_full.fit_transform(y_data_raw)

unique_classes, counts = np.unique(y_encoded_full, return_counts=True)
single_sample_encoded_labels = unique_classes[counts < 2]
single_sample_crime_categories = le_full.inverse_transform(single_sample_encoded_labels)

# 원본 DataFrame (df)에서 단일 샘플 클래스에 해당하는 행 제거
df_filtered = df[~df['범죄대분류'].isin(single_sample_crime_categories)].copy()

# 필터링된 데이터로 X_data_raw_filtered와 y_data_raw_filtered를 다시 정의
X_data_raw_filtered = df_filtered.iloc[:, 2:].values
y_data_raw_filtered = df_filtered['범죄대분류'].values

# y(텍스트)를 숫자로 변환 (Label Encoding) - 필터링된 데이터에 대해
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_data_raw_filtered)
class_names = encoder.classes_ # 나중에 Confusion Matrix에 사용할 클래스 이름

# X(수치)를 정규화 (StandardScaler) - 필터링된 데이터에 대해
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data_raw_filtered)

# --- [3] 데이터 분리 (테스트 데이터, 학습 데이터) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n학습 데이터 크기: {X_train.shape}, 테스트 데이터 크기: {X_test.shape}")

# --- [4] 분석 모델 적용 (fit) ---
# 딥러닝 모델 설계 (DNN)
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.3)) # 과적합 방지
model.add(Dense(32, activation='relu'))
model.add(Dense(len(class_names), activation='softmax')) # 출력층 노드 수 = 클래스 개수

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', # y가 정수 인코딩 상태이므로 사용
              metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=8,
                    validation_split=0.2, # 학습 중 검증에 사용할 데이터 비율
                    verbose=0) # 학습 과정 상세 로그는 생략
print("\n딥러닝 모델 학습 완료.")