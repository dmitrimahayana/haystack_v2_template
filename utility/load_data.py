from dotenv import load_dotenv
from haystack import Document
from datetime import datetime
import pandas as pd
import os

# Load env
load_dotenv()


def load_data():
    digimon_dataset = os.getenv("digimon_dataset_path")
    df = pd.read_csv(digimon_dataset)
    list_doc = []
    for index, row in df.iterrows():
        content = f"digimon {row['Digimon']} is on stage {row['Stage']}. It has {row['Type']} type and {row['Attribute']} attribute."
        doc_data = Document(
            content=content,
            id=row['Number'],
            meta={
                "digimon": row['Digimon'],
                "stage": row['Stage'],
                "type": row['Type'],
                "attribute": row['Attribute'],
                "memory": row['Memory'],
                "equip_slots": row['Equip Slots'],
                "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            }
        )
        list_doc.append(doc_data)
    return list_doc
