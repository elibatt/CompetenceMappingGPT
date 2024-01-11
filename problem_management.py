import openai
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

openai.organization = "org-pzwdKxabuPnrc0kMWRWci2qK"
openai.api_key_path = "OPENAI_API_KEY.txt"


def mapToInteger(data):
    mapping_Fourth = {"linea esterna": 0, "impianto interno": 1}
    mapping_Sixth = {"No": 0, "Si": 1}

    mapped_data = data.copy()

    mapped_data["Fourth"] = [mapping_Fourth[value] for value in expected_data["Fourth"]]
    mapped_data["Sixth"] = [mapping_Sixth[value] for value in expected_data["Sixth"]]
    mapped_data["Seventh"] = [
        int(val) if val != "" else 0 for val in expected_data["Seventh"]
    ]

    i = 0
    for val in expected_data["Third"]:
        if str(val) == "Si" or str(val) == "Sì":
            mapped_data["Third"][i] = 1
        elif str(val) == "No":
            mapped_data["Third"][i] = -1
        else:
            mapped_data["Third"][i] = int(val)
        i = i + 1

    i = 0
    for val in expected_data["Fifth"]:
        if "Si" in str(val) or "Sì" in str(val):
            mapped_data["Fifth"][i] = -2 if str(val) == "-2" else 2
        elif "No" in str(val):
            mapped_data["Fifth"][i] = -2 if str(val) == "-2" else 2
        else:
            mapped_data["Fifth"][i] = int(val)
        i = i + 1

    return mapped_data


A_answers = [[], [], [], [], [], [], [], [], []]

expected_data = {
    "First": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "Second": [-1, -1, 1, 1, 1, 1, -1, 1, 1, -1],
    "Third": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "Fourth": [
        "linea esterna",
        "impianto interno",
        "impianto interno",
        "impianto interno",
        "linea esterna",
        "linea esterna",
        "impianto interno",
        "impianto interno",
        "linea esterna",
        "impianto interno",
    ],
    "Fifth": [+2, -2, +2, +2, +2, +2, -2, +2, +2, -2],
    "Sixth": ["No", "No", "Si", "No", "No", "No", "Si", "No", "No", "Si"],
    "Seventh": ["", "", +1, "", "", "", +1, "", "", +1],
}

for i in tqdm(range(1, 11), desc="Testing phase"):
    ctm_file_path = "data/ticket_notes/Note0" + str(i) + ".ctm"
    ctm_text = []
    with open(ctm_file_path, "r", encoding="utf-8") as ctm_file:
        for line in ctm_file:
            line = line.strip()
            if not line:
                continue

            ctm_text.append(line.strip())

    premise = (
        "Date le seguenti note di lavorazione di un operatore in ambito provider Internet: \n\n"
        + str(ctm_text)
        + "\n\nRispondi alla seguente domanda: \n"
    )

    first_question = (
        premise
        + "L'operatore ti è sembrato professionale? Sì: +1, No: -1\n\nDevi rispondere solo con il 'punteggio numerico della risposta' senza spiegazione e senza riscrivere le domande."
    )
    second_question = (
        premise
        + "Il numero di giorni che sono serviti per la risoluzione del guasto è maggiore uguale di 2? Sì: -1; No: +1\n\nDevi rispondere così 'punteggio della risposta;' con una spiegazione e senza riscrivere le domande."
    )
    third_question = (
        premise
        + "Il cliente è stato aggiornato sulla situazione del ticket? Si: +1, No: -1\n\nDevi rispondere solo con il 'punteggio numerico della risposta;' con una spiegazione e senza riscrivere le domande."
    )
    fourth_question = (
        premise
        + "Il tecnico ha riscontrato che il problema è riconducibile all'impianto interno (es. problemi sul router o di configurazioni) o alla linea esterna ?"
        + "\n\nDevi rispondere solo con la risposta alla domanda senza nessuna spiegazione e senza riscrivere la domanda."
        + " La risposta alla domanda può essere solo 'impianto interno' oppure 'linea esterna'."
    )
    sixth_question = (
        premise
        + "Durante la risoluzione del guasto è stato necessario sostituire/cambiare il router? Si: +1, No: -1 "
        + "\n\nDevi rispondere solo con la risposta alla domanda senza nessuna spiegazione e senza riscrivere la domanda."
        + " Rispondi con 'Si' oppure 'No'."
    )
    questions = [
        first_question,
        second_question,
        third_question,
        fourth_question,
        sixth_question,
    ]

    count = 0
    for q in questions:
        time.sleep(5)
        count += 1
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.1,
            messages=[{"role": "user", "content": q}],
        )

        response = completion.choices[0].message.content

        if q == second_question:
            parts_resp_2 = response.split(";")
            response = parts_resp_2[0]
        if q == third_question:
            response = response[0:2]

        A_answers[count].append(response)

        if q == fourth_question:
            punteggio_si = "+2"
            punteggio_no = "-2"
            if str(response).find("impianto interno") > -1:
                punteggio_si = "-2"
                punteggio_no = "+2"

            fifth_question = (
                premise
                + "C'è stato bisogno di far intevenire esternamente il tecnico territoriale? Sì: "
                + str(punteggio_si)
                + ", No: "
                + str(punteggio_no)
                + "\n\nDevi rispondere solo con il punteggio della risposta senza nessuna spiegazione e senza riscrivere le domande."
            )

            time.sleep(5)
            completion_5 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=0.1,
                messages=[{"role": "user", "content": fifth_question}],
            )

            response_5 = completion_5.choices[0].message.content

            count += 1
            A_answers[count].append(response_5)

        elif q == sixth_question:
            response_7 = ""
            if str(response).find("Si") > -1:
                seventh_question = (
                    premise
                    + "Il router che è stato cambiato era effettivamente rotto/guasto? Si: +1, No: -1"
                    + "\n\nDevi rispondere solo con il punteggio della risposta senza nessuna spiegazione e senza riscrivere le domande."
                )

                time.sleep(5)
                completion_7 = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    messages=[{"role": "user", "content": seventh_question}],
                )

                response_7 = completion_7.choices[0].message.content

            count += 1
            A_answers[count].append(response_7)


data = {
    "First": A_answers[1],
    "Second": A_answers[2],
    "Third": A_answers[3],
    "Fourth": A_answers[4],
    "Fifth": A_answers[5],
    "Sixth": A_answers[6],
    "Seventh": A_answers[7],
}
df = pd.DataFrame(data)

mapped_data = mapToInteger(data)
mapped_expected_data = mapToInteger(expected_data)

num_item = len(mapped_data["First"])

df = pd.DataFrame(
    columns=[
        "NOTE_NUMBER",
        "FIRST",
        "TARGET_FIRST",
        "SECOND",
        "TARGET_SECOND",
        "THIRD",
        "TARGET_THIRD",
        "FOURTH",
        "TARGET_FOURTH",
        "FIFTH",
        "TARGET_FIFTH",
        "SIXTH",
        "TARGET_SIXTH",
        "SEVENTH",
        "TARGET_SEVENTH",
    ]
)

for i in range(0, num_item):
    row = pd.DataFrame(
        {
            "NOTE_NUMBER": [i + 1],
            "FIRST": [mapped_data["First"][i]],
            "TARGET_FIRST": [mapped_expected_data["First"][i]],
            "SECOND": [mapped_data["Second"][i]],
            "TARGET_SECOND": [mapped_expected_data["Second"][i]],
            "THIRD": [mapped_data["Third"][i]],
            "TARGET_THIRD": [mapped_expected_data["Third"][i]],
            "FOURTH": [mapped_data["Fourth"][i]],
            "TARGET_FOURTH": [mapped_expected_data["Fourth"][i]],
            "FIFTH": [mapped_data["Fifth"][i]],
            "TARGET_FIFTH": [mapped_expected_data["Fifth"][i]],
            "SIXTH": [mapped_data["Sixth"][i]],
            "TARGET_SIXTH": [mapped_expected_data["Sixth"][i]],
            "SEVENTH": [mapped_data["Seventh"][i]],
            "TARGET_SEVENTH": [mapped_expected_data["Seventh"][i]],
        }
    )
    df = pd.concat([df, row], ignore_index=True)

df.to_csv("data/results_problem_solving_competences.csv", index=False)
