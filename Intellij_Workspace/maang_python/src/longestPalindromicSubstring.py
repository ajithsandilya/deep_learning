#Brute Force Solution to find the longest palindromic substring in a given string

def palindromeList(a):
    palinlist=[]
    for i in range(len(a)):
        for j in range(i+1, len(a)+1):
            if a[i:j] == a[i:j][::-1]:
                palinlist.append(a[i:j])
    return palinlist

def longestPalindrome(palinlist):
    longest = ""
    longestlist = []
    for i in palinlist:
        if len(i) > len(longest):
            longest = i
    for i in palinlist:
        if len(i) == len(longest):
            longestlist.append(i)
    return longestlist

def main():
    string = input("Enter the string:" )
    print(string)
    palinlist = palindromeList(string)
    print(palinlist)
    print("Longest Substring is:")
    print(longestPalindrome(palinlist))






















if __name__ == "__main__":
    print("Victory is Certain \\o/")
    print("**********************************")
    main()