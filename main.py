import sys
from test import dataset

if __name__ == "__main__":
    if sys.argv[1] == "--test":
        if sys.argv[2] == 'wheel':
            pass

    elif sys.argv[1] == '--dataset':
        if sys.argv[2] == 'mining':
            dataset.mining()

    else:
        print("Error: Invalid option!")
