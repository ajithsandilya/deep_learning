def palindromeList(string):
    palindromeLists = []
    for i in range(len(string)):
        for j in range(i, len(string)):
            substring = string[i:j+1]
            if substring == substring[::-1]:
                palindromeLists.append(substring)
    print(palindromeLists)
    return palindromeLists


def main():
    string = input("Enter the string:" )
    print(string)
    print(palindromeList(string))





















if __name__ == "__main__":
    print("Victory is Certain \\o/")
    print("**********************************")
    main()