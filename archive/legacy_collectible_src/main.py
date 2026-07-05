import pandas as pd

# CSV FILE PATH
DAILY_SNAPSHOTS_CSV_FILE_PATH = "./data_souece/daily_snapshots.csv"
EVENTS_CSV_FILE_PATH = "./data_souece/events.csv"
ITEMS_CSV_FILE_PATH = "./data_souece/items.csv"


# load csv and convert to dataframe
def load_csv_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


# extract target item data (include index)
def extract_item_data(item_name: str, items_data: pd.DataFrame) -> pd.DataFrame:
    return items_data[items_data["item_name"] == item_name]


item_name = "ONE PIECE プロモカードA"
df = load_csv_data(DAILY_SNAPSHOTS_CSV_FILE_PATH)
item = extract_item_data(item_name, df)

print(item)
