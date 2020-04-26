class Solution:
    def eliminateChars(self, string, flag=0):
        print(string)
        stringLen = len(string)
        if stringLen <= 2:
            return string
        else:
            l, r = stringLen - 2, stringLen - 1
            # print(l, r)
            while l >= 0:
                # print(string[l], string[r])
                if string[l] == string[r]:
                    l -= 1
                else:
                    if r - l >= 3:
                        flag = 1
                        string = string.replace(string[l+1:r+1], "")
                        print(string)
                        r = l
                        l -= 1
                    else:
                        r = l
                        l -= 1
            if r - l >= 3:
                flag = 1
                string = string.replace(string[l+1:r+1], "")
                print(string)
        if flag == 0:
            return string
        else:
            return self.eliminateChars(string, flag=0)


if __name__ == "__main__":
    s = Solution()
    # testlist = ['aaababccc', 'aa', 'c', 'aaaabbbciiia', 'ababbbacccaauuyiiiyyih', 'uuuuu', 'acccabbbabbbaba']
    testlist = ['acccabbbabbbaba', 'accca', 'abbbdddcccabbbabbbaba']
    for ele in testlist:
        print('ans:', s.eliminateChars(ele, flag=0))


