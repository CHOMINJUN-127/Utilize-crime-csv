import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- [1] 데이터 불러오기 ---
filename = '경찰청_범죄 발생 장소별 통계_20241231.csv'

# 한글 CSV 파일의 인코딩 문제 해결을 위해 두 가지 방식으로 시도
try:
    df = pd.read_csv(filename, encoding='cp949')
except:
    df = pd.read_csv(filename, encoding='euc-kr')

print("데이터 로드 완료. 크기:", df.shape)

# --- [2] 결측치 처리 ---
# 범죄 통계 데이터에서 빈 칸(NaN)은 발생 건수가 0인 경우가 많으므로 0으로 채움
df = df.fillna(0)

# X, y 설정
# X_data: 장소별 통계 수치 (3번째 컬럼부터 끝까지)
# y_data: 예측할 대상 (범죄대분류)
X_data_raw = df.iloc[:, 2:].values
y_data_raw = df['범죄대분류'].values

# --- [3] 단일 샘플 클래스 처리 (오류 방지) ---
# 데이터 분리(train_test_split) 시 하나의 샘플만 있는 클래스는 오류를 일으킬 수 있어 제거
le_full = LabelEncoder()
y_encoded_full = le_full.fit_transform(y_data_raw)
unique_classes, counts = np.unique(y_encoded_full, return_counts=True)
single_sample_crime_categories = le_full.inverse_transform(unique_classes[counts < 2])

# 단일 샘플 클래스에 해당하는 행 제거
df_filtered = df[~df['범죄대분류'].isin(single_sample_crime_categories)].copy()

# 필터링된 데이터로 X, y 최종 정의
X_data_raw_filtered = df_filtered.iloc[:, 2:].values
y_data_raw_filtered = df_filtered['범죄대분류'].values

print("\n1단계 데이터 전처리 완료.")






