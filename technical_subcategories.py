import pandas as pd
import openai
import time

from fuzzywuzzy import fuzz
from tqdm import tqdm  # Import tqdm for the progress bar

openai.organization = "org-pzwdKxabuPnrc0kMWRWci2qK"
openai.api_key_path = "OPENAI_API_KEY.txt"

df = pd.read_csv("data/base_subcategories.csv")
all_subcategories = list(df.TITLE.unique())

# Testing loop: Retrying records that were not successfully processed.
while df["RESPONSE"].isna().any():
    indexes = df[df["RESPONSE"].isna()].index
    for i in tqdm(indexes, desc="Testing phase", unit="trial"):
        row = df.iloc[i].copy()

        premise = (
            "You have a list of subcategories\n\n"
            + str(all_subcategories)
            + "\n\nand the following message, representing a ticket from milwaukee citizens reporting one of the problems specified in the previous category list\n\n"
            + str(df.iloc[i].CASECLOSUREREASONDESCRIPTION)
            + "\n\nThe problems are related to Garbage Carts. Can you identify the ticket subcategory? Please note that Garbage Cart: Missing refers to citizen-owned carts that are missing, while Garbage Cart: Additional refers to problems regarding additional carts requested by the citizen. Garbage Cart: Delete refers to the removal of a garbage cart.\n\n"
        )

        answer_specifics = "Please answer like this: <name of the subcategory>\n\n"

        examples = "Each list represents an example. The first element is the ticket and the second is the category you need to associate to it: \n\n['Owner, single family home, 1G cart on site, would like an additional G cart as it is already being harged on water bill', 'Garbage Cart: Additional'],['Garbage can lid is broken and there is a hole near the top of the can.  Requesting a replacement.' 'Garbage Cart: Damaged'],['Address is a duplex with 3 carts on site. Only need 2. Please delete/ collect 1 Garbage Cart.','Garbage Cart: Delete'],['Caller reports cart was stolen and he needs another one, he states he previously requested this two weeks ago.','Garbage Cart: Missing']\n\n"

        final_notes = "Please do not use any other syntactic sugar in your answer."

        content = premise + answer_specifics + examples + final_notes

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": content}]
        )

        response = completion.choices[0].message.content

        if response in all_subcategories:
            if "Garbage Cart" in response:
                row["IS_CATEGORY_CORRECT"] = True
                row["CATEGORY_ANSWER"] = "Garbage Cart"
                if df.iloc[i].TITLE == response:
                    row["IS_SUBCATEGORY_CORRECT"] = True
                row["SUBCATEGORY_ANSWER"] = response

                row["RESPONSE"] = completion.choices[0].message.content

                df.iloc[i] = row
            else:
                row["CATEGORY_ANSWER"] = "Wrong"
                row["SUBCATEGORY_ANSWER"] = "None"
        else:
            tqdm.write(f"This response isn't in the correct format (Ticket {i})")

        df.to_csv("data/results_subcategories.csv", index=False)
        time.sleep(10)
