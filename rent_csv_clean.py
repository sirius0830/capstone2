import pandas as pd

# 원본 
file_path = "gwangju_rent.csv"

# 설명 줄 건너뛰기 (전용면적이 포함된 줄부터 시작)
with open(file_path, encoding='cp949') as f:
    for i, line in enumerate(f):
        if '전용면적' in line:
            header_row = i
            break

# CSV 로드 (헤더 시작 줄부터)
df = pd.read_csv(file_path, encoding='cp949', skiprows=header_row)

# 열 이름 영어로 변경
df = df.rename(columns={
    '전용면적(㎡)': 'area',
    '보증금(만원)': 'deposit',
    '월세금(만원)': 'monthly_rent',
    '시군구': 'dong',
    '전월세구분': 'rent_type'
})

# 필요한 열만 선택
df = df[['dong', 'area', 'rent_type', 'deposit', 'monthly_rent']].dropna()

# price 쉼표 제거하고 숫자형으로 변환
df['deposit'] = df['deposit'].replace(',', '', regex=True).astype(int)
df['monthly_rent'] = df['monthly_rent'].replace(',', '', regex=True).astype(int)

# 새 CSV로 저장
df.to_csv("cleaned_gwangju_rent.csv", index=False, encoding='utf-8-sig')