#Optimized version of longest palindromic substring using list comprehension and slicing

def oddPalindromeChecker(string,center):
    left=center-1
    right=center+1
    while left>=0 and right<len(string) and string[left]==string[right]:
        left-=1
        #print(f"left = {left}")
        right+=1
        #print(f"right = {right}")

    return string[left+1:right]

def evenPalindromeChecker(string,center):
    center1= center-1
    left=center1-1
    right=center+1
    #print(f"center1 {string[center1]}")
    #print(f"center {string[center]}")
    if string[center] != string[center1]:
        #print("Given String is not a palindrome")
        #print(f"Centers Unequal : {string[center1:center+1]}")
        return ""
    while left>=0 and right<len(string) and string[left]==string[right]:
        #print(string[left])
        #print(string[right])
        left-=1
        right+=1

    return string[left+1:right]



def stingsetter(string):
    longest_list = []
    max_len = 0
    if len(string)==0:
        return ["Provide a valid string"]
    else:
        for i in range(len(string)):
            center=i
            resultOdd=oddPalindromeChecker(string,center)
            if len(resultOdd) > max_len:
                longest_list = [resultOdd]
                max_len = len(resultOdd)
            elif len(resultOdd) == max_len:
                if resultOdd not in longest_list:
                    longest_list.append(resultOdd)

            resultEven=evenPalindromeChecker(string,center)
            if len(resultEven) > max_len:
                longest_list = [resultEven]
                max_len = len(resultEven)
            elif len(resultEven) == max_len:
                if resultEven not in longest_list:
                    longest_list.append(resultEven)

    return longest_list
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