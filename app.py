from datetime import datetime
import yfinance as yf

def get_current_time():
    current_time = datetime.now()
    return current_time.strftime('%Y-%m-%d %H:%M:%S')

def get_nasdaq_price():
    nasdaq = yf.Ticker("^IXIC")
    return nasdaq.info['regularMarketPrice']

def get_even_numbers():
    number = [1,2,3,4,5,6,7,8,9,10]
    return [num for num in number if num % 2 == 0]

def get_user_info():
    name = input("이름을 입력하세요: ")
    age = input("나이를 입력하세요: ")
    return name, age

def main():
    print(f"현재 시각: {get_current_time()}")
    print(f"현재 나스닥 지수: {get_nasdaq_price():,.2f}")
    print(f"1부터 10까지의 짝수: {get_even_numbers()}")
    
    name, age = get_user_info()
    print(f"입력하신 정보 - 이름: {name}, 나이: {age}")

if __name__ == "__main__":
    main()

#1부터 10까지 숫자 중 짝수만 출력하는 프로그램
number = [1,2,3,4,5,6,7,8,9,10]
even_number=[]
for run in number:
    if run % 2 == 0:
        even_number.append(run)
print(even_number)

# 사용자로부터 이름과 나이를 입력받아 출력하는 프로그램
name = input("이름을 입력하세요: ") 

