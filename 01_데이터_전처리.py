import pandas as pd
import numpy as np

# --- [1] 데이터 불러오기 ---
filename = '경찰청_범죄 발생 장소별 통계_20241231.csv'

# 한글 CSV 파일의 인코딩 문제 해결을 위해 두 가지 방식으로 시도
try:
    df = pd.read_csv(filename, encoding='cp949')
except:
    df = pd.read_csv(filename, encoding='euc-kr')

print("데이터 로드 완료. 크기:", df.shape)

# --- [2] 데이터 정보 확인 및 결측치 처리 ---
# 전체 결측치 개수 확인
total_nan = df.isnull().sum().sum()
print(f"\n[초기 데이터 결측치 개수]: {total_nan}개")

# 범죄 통계 데이터에서 빈 칸(NaN)은 발생 건수가 0인 경우가 많으므로 0으로 채움 (결측치 처리)
df = df.fillna(0)

# X, y 설정
# X_data: 장소별 통계 수치 (3번째 컬럼부터 끝까지)
# y_data: 예측할 대상 (범죄대분류)
X_data_raw = df.iloc[:, 2:].values
y_data_raw = df['범죄대분류'].values

print("\n[전처리 완료된 데이터 X의 첫 5개 행]")
print(pd.DataFrame(X_data_raw).head())
print("\n[전처리 완료된 레이블 y의 종류]")
print(np.unique(y_data_raw))