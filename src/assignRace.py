import numpy as np
import pandas as pd
import cv2
import imutils


def parse_response(race, df, index):
    # Exit the loop and save progress
    if race == "exit":
        return True, "exit"
    
    # If the previous image was mislabelled, go back
    if race == "back":
        # Ignore duplicates when finding previous image
        sub = 1
        while df.iloc[index - sub]['duplicate'] == 1:
            sub += 1
        
        handle_response(df, index - sub)
        return False, "Now classify current image."
    
    # Try to convert to int, otherwise try again
    try:
        race = int(race)
        return True, race
    except:
        return False, "Invalid response. Try again."

def handle_response(df, index):
    # Prepare image to show
    row = df.iloc[index]
    img = np.reshape(np.array(row['pixels'].split(' '), dtype=np.uint8),  (48, 48))
    img = imutils.resize(img, width=200)
    
    complete = False
    while not complete:
        # Show image
        cv2.imshow('img', img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        
        # Get response
        race = input("Race: ")
        
        # Parse response
        complete, response = parse_response(race, df, index)
        print(response)
    
    if response == "exit":
        return True
    
    else:
        # Set the value
        df.loc[index, "race"] = race
        return False

def main():
    # Get and load the data
    df = pd.read_csv("../data/fer2013race_sim.csv")

    exit = False
    cv2.namedWindow('img')
    cv2.moveWindow('img', 500, 500)

    for index, row in df.iterrows():
        # Only edit if race is -1 (Undefined) and not a duplicate (0)
        if row['race'] == -1 and row['similar'] == 0:
            
            if index % 10 == 0:
                print(index)

            # Get and parse the response
            exit = handle_response(df, index)
            if exit:
                break

    df.to_csv("../data/fer2013race.csv", index=False)

if __name__ == '__main__':
    main()
