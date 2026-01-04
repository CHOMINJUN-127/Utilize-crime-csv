import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from matplotlib import font_manager, rc

# --- [1] 데이터 불러오기 (nxwcxdjq-sxc 셀에서 복사) ---
filename = '경찰청_범죄 발생 장소별 통계_20241231.csv'

try:
    df = pd.read_csv(filename, encoding='cp949')
except:
    df = pd.read_csv(filename, encoding='euc-kr')

df = df.fillna(0) # 결측치 0으로 채우기

# --- [2] 폰트 설정 (TWAFDK2oJbeK 셀에서 복사) ---
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

# --- [3] '총발생건수' 컬럼 추가 (TWAFDK2oJbeK 셀에서 복사) ---
df['총발생건수'] = df.iloc[:, 2:].sum(axis=1)

# 2단계 코드의 Bar Plot 다음에 추가하여 실행해 보세요.

# 모든 장소 컬럼을 가져옵니다.
location_data = df.iloc[:, 2:-1] # '총발생건수' 컬럼 제외

# 범죄대분류별로 장소별 발생 건수의 평균을 집계합니다.
heatmap_data = df.groupby('범죄대분류')[location_data.columns].sum()

# 히트맵 시각화
plt.figure(figsize=(15, 8))
# MinMax Scaling을 적용하여 색상 대비를 명확하게 합니다.
sns.heatmap(heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1),
            cmap="YlGnBu",
            annot=False, # 숫자는 너무 많으므로 생략
            fmt=".2f",
            linewidths=.5,
            cbar_kws={'label': '정규화된 발생 건수 비율'})
plt.title('범죄 대분류와 발생 장소 간의 상관관계 (히트맵)')
plt.show()


