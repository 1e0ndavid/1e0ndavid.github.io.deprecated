class Solution:
    def twoSum1(self, numbers, target):
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                if numbers[i] + numbers[j] == target:
                    return [i + 1, j + 1]

    def twoSum2(self, numbers, target):
        i, j = 0, len(numbers)-1
        while i < j:
            if numbers[i] + numbers[j] > target:
                j -= 1
            elif numbers[i] + numbers[j] < target:
                i += 1
            else:
                return [i + 1, j + 1]

    def twoSum3(self, numbers, target):
        i, j = 0, len(numbers) - 1
        for i in range(len(numbers)):
            while j - 1 > i and numbers[j - 1] + numbers[i] >= target:
                j -= 1
            if numbers[i] + numbers[j] == target:
                return [i + 1, j + 1]


if __name__ == "__main__":
    s = Solution()
    testcases = [[2, 7, 11, 15, 9], [2, 3, 4, 6]]
    for ele in testcases:
        print(s.twoSum1(ele[:-1], ele[-1]))
        print(s.twoSum2(ele[:-1], ele[-1]))
        print(s.twoSum3(ele[:-1], ele[-1]))
