import openai
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

openai.organization = "org-pzwdKxabuPnrc0kMWRWci2qK"
openai.api_key_path = "OPENAI_API_KEY.txt"

df = pd.DataFrame(
    columns=[
        "TRANSCRIPTION_NUMBER",
        "KINDNESS",
        "ABILITY_TO_EXPLAIN",
        "PITCH_SALE",
        "CUSTOMER_SATISTFACTION",
        "TARGET_KINDNESS",
        "TARGET_EXPLAIN",
        "TARGET_PITCH",
        "TARGET_SATISFACTION",
        "OVERALL",
        "TARGET_OVERALL",
    ]
)

expected_values = [
    [1, 1, 1, 1],
    [1, 0.75, 1, 1],
    [1, 0.75, 1, 1],
    [1, 0.75, 1, 1],
    [1, 1, 1, 1],
    [1, 0.75, 0, 0.75],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
    [0, 0.5, 0, 0],
    [0, 0.5, 0, 0],
    [0, 0, 0, 0],
    [0.75, 1, 1, 0.5],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]

for i in tqdm(range(1, 19), desc="Testing phase"):
    ctm_file_path = "data/transcriptions/Script0" + str(i) + "_clean_en.ctm"
    ctm_text = []
    with open(ctm_file_path, "r", encoding="utf-8") as ctm_file:
        for line in ctm_file:
            line = line.strip()
            if not line:
                continue

            ctm_text.append(line.strip())

    premise = (
        "\n\nYou need to evaluate the following conversation, contained in a list: \n\n"
        + str(ctm_text)
        + "\n\nEach sentence is separated by a comma and starts with an identifier that reveals who is speaking (Either 'Customer' or 'Tech'). You are a Human Resources employee you have to answer the following question: "
    )
    additional_notes = " Mentioning a new product or service is considered GOOD BEHAVIOR by the company. PLEASE DO NOT REMOVE ANY POINT TO ANYONE IF THEY ARE TRYING TO SELL WHATEVER. Sometimes some information may be repeated THIS IS EXPECTED because when processing the customer request the tech is INSTRUCTED TO ALWAYS DOUBLE CHECK EVERY INFORMATION, do not remove any point for this behavior. Keep in mind that using offensive and inappropriate language is prohibited by the company policy, so the scoring must be strongly penalized in any situation in which this occurs."
    scoring = (
        " Give me 4 if he was, 0 otherwise. \n\nYou have to answer only with the score for each question. For example: 8;"
        + additional_notes
    )
    scoring_binary = (
        " Give me 1 if he was, 0 otherwise. \n\nYou have to answer only with the score for each question. For example: 8;"
        + additional_notes
    )

    first_question = "Was the tech kind?"
    second_question = "Was the tech able to explain himself?"
    third_question = "Was the tech able to mention the existence of additional products while conversating with the customer?"
    fourth_question = "Was the customer satisfied with the tech service in the end?"
    questions = [first_question, second_question, third_question, fourth_question]

    examples = """
    This is an example of a conversation between the tech support operator and the customer. 

    Tech: Hello, good morning. This is Gigi speaking from Milano. How can I assist you today?

    Customer: Hello, I'm having a problem with my internet connection. Today it's very slow.

    Tech: Alright. I understand your frustration and I want to assure you your issue is important for us. I'm sorry our service has created you any inconvenience. Can you please provide me your account number so that I can check your connection?

    Customer: Sure, my account number is 82323489203.

    Tech: Alright this seems to be due to a scheduled maintenance task that will be completed in a few minutes.

    Customer: Thanks.
    
    In this case the operator was very kind because they made the user feel important and apologized for the inconvenience. The operator was also able to explain himself and provide all the necessary information.

    
    This is another example of a conversation between the tech support operator and the customer. 

    Tech: Hello, good morning. This is Gigi speaking from Milano. How can I assist you today?

    Customer: Hello, I'm having a problem with my internet connection. Today it's very slow.

    Tech: Alright. I understand your frustration and I want to assure you your issue is important for us. I'm sorry our service has created you any inconvenience. Can you please provide me your number?
    
    Customer: Which number?

    Tech: The account number.

    Customer: Oh ok, my account number is 82323489203.

    Tech: Ah I see there is currently a situation. Please wait a bit and it will resolve.

    Customer: Ok, but may I ask which situation?

    Tech: The one with the thing, the maintenance.

    Customer: Ok, goodbye.
    
    In this case the operator was very kind because they made the user feel important and apologized for the inconvenience. They were also very kind throughout the conversation even when they were not able to answer the customer questions. The operator was not able to explain themselves very well.

    
    This is another example of a conversation between the tech support operator and the customer. 

    Tech: Hello, good morning. This is Gigi speaking from Milano. How can I assist you today?

    Customer: Hello, I'm having a problem with my internet connection. Today it's very slow.

    Tech: It's the fourth today someone is asking me why their connection is slow. It happens. I can give you a specific reason if you want but you can just wait.

    Customer: Well I would like to know why my internet is slowing down.

    Tech: Alright. Give me your account number.

    Customer: My account number is 94302392380

    Tech: As I predicted, the reason is a scheduled maintenance task that will be completed in a few minutes. Just wait.

    Customer: Ok.
    
    In this case the operator was rude with a customer that was experiencing a problem, even though they were actually able to quickly explain themselves to the customer.

    
    This is another example of a conversation between the tech support operator and the customer. 

    Tech: Hello, good morning. This is Gigi speaking from Milano. How can I assist you today?

    Customer: Hello, I'm having a problem with my internet connection. Today it's very slow.

    Tech: It's the fourth today someone is asking me why their connection is slow. It happens. I do not give a fuck if your connection is slow, I have more important stuff to do.

    Customer: Well I'm a customer and I deserve to know why I'm not receieving the service I'm paying for.

    Tech: No you don't.
    
    In this case the operator was rude with a customer that was experiencing a problem. He refused to explain themselves and hung up which is not what should have happened.
    """

    scoring_system_list = [
        scoring,
        scoring,
        scoring_binary,
        scoring,
    ]

    count = 0

    values = []

    for q in tqdm(questions, desc="Asking questions.."):
        prompt = examples + premise + q + scoring_system_list[count]

        time.sleep(3)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        response = completion.choices[0].message.content
        value = response.split(";")[0]

        if count != 2:
            min_value = 0
            max_value = 4
            normalized_value = (int(value) - min_value) / (max_value - min_value)
            values.append(int(normalized_value))
        else:
            values.append(int(value))

        count += 1

    overall = sum(np.array(values))
    target_overall = sum(np.array(expected_values[i - 1]))

    row = pd.DataFrame(
        {
            "TRANSCRIPTION_NUMBER": [i],
            "KINDNESS": [values[0]],
            "ABILITY_TO_EXPLAIN": [values[1]],
            "PITCH_SALE": [values[2]],
            "CUSTOMER_SATISTFACTION": [values[3]],
            "TARGET_KINDNESS": [expected_values[i - 1][0]],
            "TARGET_EXPLAIN": [expected_values[i - 1][1]],
            "TARGET_PITCH": [expected_values[i - 1][2]],
            "TARGET_SATISFACTION": [expected_values[i - 1][3]],
            "OVERALL": [overall],
            "TARGET_OVERALL": [target_overall],
        }
    )
    df = pd.concat([df, row], ignore_index=True)

df.to_csv("data/results_communication_competences_v2.1.csv", index=False)
