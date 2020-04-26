import sys

def sqrtnum(num):
    ans = num/2.0
    while ans*ans - num >= 0.000001:
        ans = (ans + num/ans)/2.0
    return ans

def cal(a, b, c):
    indelta = 4*a*a-8*a*b*c
    if indelta <= 0:
        return 0
    else:
        y1 = (a + sqrtnum(indelta)) / b
        y2 = (a - sqrtnum(indelta)) / b
        area = y1*y1/2/b - y1*c/b - y1*y1*y1/6/a - y2*y2/2/b + y2*c/b + y2*y2*y2/6/a
        return area


numline = sys.stdin.readline()
for i in range(int(numline)):
    data = list(map(float, sys.stdin.readline().split()))
    A, B, C = data[0], data[1], data[2]
    print(cal(A, B, C))