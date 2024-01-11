import pandas as pd
import openai
import time

from fuzzywuzzy import fuzz
from tqdm import tqdm  # Import tqdm for the progress bar

openai.organization = "org-pzwdKxabuPnrc0kMWRWci2qK"
openai.api_key_path = "OPENAI_API_KEY.txt"

df = pd.read_csv("data/base_categories_and_streets.csv")
all_categories = df.TITLE.unique()

# Testing loop: Retrying records that were not successfully processed.
while df["RESPONSE"].isna().any():
    indexes = df[df["RESPONSE"].isna()].index
    for i in tqdm(indexes, desc="Testing phase", unit="trial"):
        row = df.iloc[i].copy()

        premise = (
            "You have this list of categories "
            + str(all_categories)
            + " and the following message, representing a ticket from milwaukee citizens reporting one of the problems specified in the previous category list:\n\n"
            + str(df.iloc[i].CASECLOSUREREASONDESCRIPTION)
            + "\n\ncan you identify the category of the ticket and the street in which the problem occurred? \n Please note that Missed Collection: Garbage is used to categorize a ticket where the citizen is complaining about garbage carts not being collected as scheduled; Sanitation Inspector Notification is used to complain about unexpected garbage dispersion on the streets of the city.\n\n"
        )

        answer_specifics = (
            "Please answer like this: <name of the category>;<name of street>\n"
            + "If there is more than one street in the sentence write: <name of the category>;<name of first street>,<name of second street> etc.\n"
        )

        street_notes = "Do not take into consideration blocks or adresses.\nIf the street is an Avenue use Ave\nIf the street is a Road use Road\nIf it is a Boulevard use Blvd\nWhen is a generic Street use Street.\nIf there is no street name replace <name of street> with only this one character: 0. Do not give any explanation just use the 0 symble.\n\n"

        example_1 = "These are some examples.\n\n Neighbor at 3044 N. 79th has a queen size matress and box frame laying on walk near garbage carts has been there for months never been picked up.\nHere you should write: Sanitation Inspector Notification; 79th Street.\n\n"

        example_2 = "Caller states the owners of the yellow store on 11th and Burleigh (1032 W Burleigh) are taking their G-carts down the alley dumping the trash on city vacant lot at 3127 N 10TH ST and 3121 N 10TH ST. Thank you\n Here you should write: Sanitation Inspector Notification; 10th Street.\n\n"

        example_3 = "If there are adresses you must ignore them i.e.\nThe garbage carts at my property are extremely filthy, and I would like them to be replaced.  The containers are for 5016 & 5018 N 106th Street. The system isn't allowing me to enter both addresses.\nWrite: Garbage Cart: Damaged;106th Street\n\n"

        final_notes = "Please do not use any other syntactic sugar. Your answer must have the following structure: <name of the category>;<name of street>."

        content = (
            premise
            + answer_specifics
            + street_notes
            + example_1
            + example_2
            + example_3
            + final_notes
        )

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": content}]
        )

        response = completion.choices[0].message.content.split(";")
        if len(response) == 2:
            if response[0] in all_categories:
                if df.iloc[i].TITLE in response[0]:
                    row.at["IS_CATEGORY_CORRECT"] = True

                street_response = response[1].strip()
                row.at["STREET_CORRECTNESS"] = (
                    100
                    if street_response[0] == "-"
                    else fuzz.ratio(
                        street_response.lower(), df.iloc[i].STREET_LABEL.lower()
                    )
                )
                row.at["CATEGORY_ANSWER"] = response[0]
                row.at["STREET_ANSWER"] = street_response
                row.at["RESPONSE"] = response

                df.iloc[i] = row
            else:
                tqdm.write(f"Assigned category is not accepted (Record {i})")
        else:
            tqdm.write(f"Response is too short (Record {i})")

        df.to_csv("data/results_categories_and_streets.csv", index=False)
        time.sleep(10)
