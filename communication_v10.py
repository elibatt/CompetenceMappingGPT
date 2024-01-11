import openai
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

openai.organization = "org-pzwdKxabuPnrc0kMWRWci2qK"
openai.api_key_path = "OPENAI_API_KEY.txt"

df = pd.DataFrame(columns=["TRANSCRIPTION_NUMBER", "OVERALL", "TARGET_OVERALL"])

expected_values = [
    4,
    4,
    4,
    4,
    4,
    3,
    4,
    4,
    4,
    3,
    2,
    0,
    1,
    1,
    0,
    2,
    0,
    0,
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
    additional_notes = " Mentioning a new product or service is considered good behavior by the company. Please do not remove any score point to any employee if they are trying to sell a product to the customer. Sometimes some information may be repeated, this is expected because when processing the customer request the tech operator is instructed to double check every information: do not remove any point for this behavior. Keep in mind that using offensive and inappropriate language is prohibited by the company policy, so the scoring must be strongly penalized in any situation in which this occurs."
    scoring = (
        " \n\nYou have to answer only with the score number from 0 to 4. Do not add any explanation or syntactic sugar. For example: 3"
        + additional_notes
    )

    question = "How good was the comunication skill of the tech operator?"
    questions = [question]

    examples = """
    This is an example of a conversation between the tech support operator and the customer. 

    Tech: Hello, good morning. This is Gigi speaking from Milano. How can I assist you today?

    Customer: Hello, I'm having a problem with my internet connection. Today it's very slow.

    Tech: Alright. I understand your frustration and I want to assure you your issue is important for us. I'm sorry our service has created you any inconvenience. Can you please provide me your account number so that I can check your connection?

    Customer: Sure, my account number is 82323489203.

    Tech: Alright this seems to be due to a scheduled maintenance task that will be completed in a few minutes.

    Customer: Thanks.
    
    In this case the operator was very kind because they made the user feel important and apologized for the inconvenience. The operator was also able to explain himself and provide all the necessary information. The overall vote is 5.

    
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
    
    In this case the operator was very kind because they made the user feel important and apologized for the inconvenience. They were also very kind throughout the conversation even when they were not able to answer the customer questions. The operator was not able to explain themselves very well. The overall vote is 3.

    
    This is another example of a conversation between the tech support operator and the customer. 

    Tech: Hello, good morning. This is Gigi speaking from Milano. How can I assist you today?

    Customer: Hello, I'm having a problem with my internet connection. Today it's very slow.

    Tech: It's the fourth time today someone is asking me why their connection is slow. It happens. I can give you a specific reason if you want but you can just wait.

    Customer: Well I would like to know why my internet is slowing down.

    Tech: Alright. Give me your account number.

    Customer: My account number is 94302392380

    Tech: As I predicted, the reason is a scheduled maintenance task that will be completed in a few minutes. Just wait.

    Customer: Ok.
    
    In this case the operator was rude with a customer that was experiencing a problem, even though they were actually able to quickly explain themselves to the customer. The overall vote is 2.

    
    This is another example of a conversation between the tech support operator and the customer. 

    Tech: Hello, good morning. This is Gigi speaking from Milano. How can I assist you today?

    Customer: Hello, I'm having a problem with my internet connection. Today it's very slow.

    Tech: It's the fourth time today someone is asking me why their connection is slow. It happens. I do not give a damn if your connection is slow, I have more important problems to solve.

    Customer: Well I'm a customer and I deserve to know why I'm not receieving the service I'm paying for.

    Tech: No you don't.
    
    In this case the operator was rude with a customer that was experiencing a problem. He refused to explain themselves and hung up which is not what should have happened. The overall vote is 1.
    """

    prompt = examples + premise + question + scoring

    time.sleep(3)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    response = completion.choices[0].message.content
    score = response.split(";")[0]

    target = expected_values[i - 1]

    row = pd.DataFrame(
        {"TRANSCRIPTION_NUMBER": [i], "OVERALL": [score], "TARGET_OVERALL": [target]}
    )
    df = pd.concat([df, row], ignore_index=True)

df.to_csv("data/results_communication_competences_v1.0.csv", index=False)
