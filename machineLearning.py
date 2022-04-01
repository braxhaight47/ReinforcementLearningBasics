import random
import string
import time
# Monkey!!! This is a program to simulate a monkey infinitely typing random chars till it hits a key word


class Monkey:

    def get_rand_char(self):
        # Returns the random value into the word/stream of words
        rn = random.randint(1, 30)
        if rn == 27:
            print(" ", end="")
            return self + " "
        elif rn == 28:
            print(".", end="")
            return self + "."
        elif rn == 29:
            print(",", end="")
            return self + ","
        elif rn == 30:
            print("?", end="")
            return self + "?"
        word2 = random.choice(string.ascii_letters)
        print(word2, end="")
        return self + word2

    # Monkey beginning his search for key
    key = "fuck"
    word = ""
    n = 1
    m = 0
    print("Start : %s" % time.ctime())
    while n > 0:
        if m % 100 == 0:
            print(" ", end="")
            print(m / 100)
        if key in word:
            print(" ", end="")
            print(m / 100)
            print("")
            print("found!")
            n = 0
            break
        word = get_rand_char(word)
        m = m + 1
        #if m % 500000 == 0:
            #time.sleep(900)
    print("End : %s" % time.ctime())
# monkey be done now

