import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import platform
from matplotlib import font_manager, rc

# =========================================================
# 한글 폰트 설정 (시각화 시 한글 깨짐 방지)
# =========================================================
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':  # Mac OS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font_name)
elif 'Linux' in platform.system():  # Google Colab, Linux
    try:
        !sudo apt-get install -y fonts-nanum > /dev/null 2>&1
        !fc-cache -fv > /dev/null 2>&1
        font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
    except:
        pass
# =========================================================

# --- [1] 레이블 처리 & 정규화(Scale) 처리 ---
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_data_raw_filtered)
class_names = encoder.classes_ # 클래스 이름 저장

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data_raw_filtered)

# --- [2] 데이터 분리 (테스트 데이터, 학습 데이터) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- [3] 분석 모델 적용 (fit) ---
# 딥러닝 모델 설계 (DNN)
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 (history 변수 생성)
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=8,
                    validation_split=0.2,
                    verbose=0)

print("\n2단계 딥러닝 모델 학습 완료. (history, model 변수 생성됨)")