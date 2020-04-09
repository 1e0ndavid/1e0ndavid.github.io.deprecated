
class Solution:
    def fib1(self, N: int) -> int:
        res = [0, 1]
        if N <= 1:
            return res[N]
        else:
            return self.fib1(N-1) + self.fib1(N-2)

    def fib2(self, N: int) -> int:
        res = [0, 1]
        if N <= 1:
            return res[N]
        else:
            fibNminusOne = 0
            fibNminusTwo = 1
            for _ in range(N):
                fibN = fibNminusOne + fibNminusTwo
                fibNminusTwo, fibNminusOne = fibNminusOne, fibN
            return fibN

    def fib3(self, N: int) -> int:
        fibNminusOne, fibNminusTwo = 0, 1
        for _ in range(N):
            fibNminusTwo, fibNminusOne = fibNminusOne, fibNminusOne + fibNminusTwo
        return fibNminusOne

    def fib4(self, N: int) -> int:
        return 0


if __name__ == '__main__':
    s = Solution()
    for i in range(11):
        print(s.fib1(i), s.fib2(i), s.fib3(i))
