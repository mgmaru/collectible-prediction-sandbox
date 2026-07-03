from typing import cast

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.elements.plotly_chart import PlotlyState

# CSV FILE PATH
DAILY_SNAPSHOTS_CSV_FILE_PATH = "./data_source/daily_snapshots.csv"
EVENTS_CSV_FILE_PATH = "./data_source/events.csv"
ITEMS_CSV_FILE_PATH = "./data_source/items.csv"


# load csv and convert to dataframe
def load_csv_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


# extract target item data (include index)
def extract_selected_item_data(
    selected_item: str, items_data: pd.DataFrame
) -> pd.DataFrame:
    return items_data[items_data["item_name"] == selected_item]


def extract_events_for_selected_item(
    selected_item_data: pd.DataFrame, events_data: pd.DataFrame
) -> pd.DataFrame:
    item_id = cast(str, selected_item_data["item_id"].iloc[0])
    selected_item_events = events_data[events_data["item_id"] == item_id]
    return selected_item_events


# list items
# use seidevar select box
def make_item_name_list(item_data: pd.DataFrame) -> list[str]:
    items = item_data["item_name"]
    return list(set(items))


def make_selected_item_line_chart(
    horintal_axis_data: str,
    vertical_axis_data: str,
    selected_item_data: pd.DataFrame,
    selected_item_events_data: pd.DataFrame,
) -> None:
    fig = px.line(selected_item_data, x=horintal_axis_data, y=vertical_axis_data)
    fig.update_xaxes(tickformat="%Y/%m/%d")
    for _, event in selected_item_events_data.iterrows():
        event_date = cast(pd.Timestamp, event["date"])
        event_type = cast(str, event["event_type"])
        fig.add_vline(
            x=event_date,
            line_dash="dash",
            line_color="red",
            annotation_text=event_type,  # 注釈ラベルも付けられる
        )
    st.plotly_chart(fig, use_container_width=True)


daily_snapshots_data = load_csv_data(DAILY_SNAPSHOTS_CSV_FILE_PATH)
events_data = load_csv_data(EVENTS_CSV_FILE_PATH)

# frontend
st.set_page_config(layout="wide")

st.title("collectible-prediction-demo")

selected_item = st.sidebar.selectbox(
    "ITEMS",
    make_item_name_list(daily_snapshots_data),
)

# data
selected_item_data = extract_selected_item_data(selected_item, daily_snapshots_data)
selected_item_events_data = extract_events_for_selected_item(
    selected_item_data, events_data
)

col1, col2 = st.columns(2, gap="xlarge")

# line chart
with col1:
    st.header("現在価格（円）")
    make_selected_item_line_chart(
        "date", "price", selected_item_data, selected_item_events_data
    )

    st.header("SNS言及数")
    make_selected_item_line_chart(
        "date", "sns_mentions", selected_item_data, selected_item_events_data
    )

with col2:
    st.header("出品数")
    make_selected_item_line_chart(
        "date", "listing_count", selected_item_data, selected_item_events_data
    )

    st.header("検索関心")
    make_selected_item_line_chart(
        "date", "trends_score", selected_item_data, selected_item_events_data
    )
