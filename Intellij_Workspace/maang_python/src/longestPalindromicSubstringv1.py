#Optimized version of longest palindromic substring using list comprehension and slicing

def longestPalindrome(s):
    if len(s) == 0:
        return "Provide a valid string"
    else:
        center = len(s) // 2
        print(center)
        i = 1
        if len(s) % 2 != 0:
            print(len(s) % 2)
            print("entered odd")
            while center - i >= 0 and center + i < len(s) and s[center - i] == s[center + i]:
                i += 1
            return s[center - i + 1 : center + i]
        else:
            centerleft = center
            print(f"centerleft {centerleft}")
            centerright = center + 1
            print(f"centerright {centerright}")
            while centerleft - i >= 0 and centerright + i < len(s) and s[centerleft - i] == s[centerright + i]:
                i += 1
            return s[centerleft - i + 1 : centerright + i]


def main():
    string = input("Enter the string:" )
    print(string)
    print(longestPalindrome(string))


























if __name__ == "__main__":
    print("Victory is Certain \\o/")
    print("**********************************")
    main()