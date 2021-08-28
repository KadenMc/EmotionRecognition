import numpy as np
import pandas as pd
import cv2
import imutils


def main():
    # Get and load the data
    try:
        df = pd.read_csv("../data/fer2013race.csv")
    except:
        df = pd.read_csv("../data/fer2013.csv")
        df["race"] = -1

    exit = False
    cv2.namedWindow('img')
    cv2.moveWindow('img', 500, 500)

    for index, row in df.iterrows():
        # Only edit if race is -1 (Undefined)
        if row['race'] == -1:
            if exit:
                break
            
            img = np.reshape(np.array(row['pixels'].split(' '), dtype=np.uint8),  (48, 48))
            img = imutils.resize(img, width=200)

            while True:
                # Show image
                cv2.imshow('img', img)
                cv2.waitKey(500)
                cv2.destroyAllWindows()
                
                # Get response
                race = input("Race: ")
            
                # Exit the loop and save progress
                if race == "exit":
                    exit = True
                    break
            
                # Try to convert to int, otherwise try again
                try:
                    race = int(race)
                    break
                except:
                    print("Invalid response. Try again.")
            
            if not exit:
                # Set race in dataframe
                df.set_value(index, 'race', race)

    df.to_csv("../data/fer2013race.csv", index=False)

if __name__ == '__main__':
    main()
