#Optimized version of longest palindromic substring using list comprehension and slicing
def expand(string, left, right):
    while left >= 0 and right < len(string) and string[left] == string[right]:
        left -= 1
        right += 1
    return string[left + 1 : right]


def stingsetter(string):
    longest = ""
    if len(string)==0:
        return "Provide a valid string"
    else:
        for i in range(len(string)):
            resultOdd=expand(string,i,i)
            resulteven=expand(string,i,i+1)
            if(len(resultOdd)>=len(longest)):
                longest=resultOdd
            if(len(resulteven)>=len(longest)):
                longest=resulteven


    return longest
def main():
    #string = input("Enter the string:" )

    # test_data = [
    #     "a",
    #     "aba",
    #     "racecar",
    #     "madam",
    #     "level",
    #     "rotator",
    #     "malayalam",
    #     "abcbaefg",
    #     "xyzracecaruvw",
    #     "noonmadamcivic",
    #     "aa",
    #     "abba",
    #     "noon",
    #     "redder",
    #     "123321",
    #     "!!@@!!@@",
    #     "aabbccbbaa",
    #     "12 21",
    #     "AaAa",
    #     "  ",
    #     "ab",
    #     "abcd",
    #     "xyyxzz",
    #     "abcbaefgxyz",
    #     "123456",
    #     ""
    # ]



    # for str in test_data:
        str = "babbbabracecar"
        print(f"Result for String:{str}")
        print(stingsetter(str))






























if __name__ == "__main__":
    print("Victory is Certain \\o/")
    print("**********************************")
    main()