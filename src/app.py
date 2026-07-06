from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st
from typing_extensions import Any

# CSV FILE PATH
DATA_SOURCE_DIR = Path(__file__).parent.parent / "data/raw/bike_sharing"
HOUR_CSV_FILE_PATH = DATA_SOURCE_DIR / "hour.csv"
DAY_CSV_FILE_PATH = DATA_SOURCE_DIR / "day.csv"


# load data
def load_csv_data(
    file_path: Path, parse_dates: list[str] | None = None
) -> pd.DataFrame:
    return pd.read_csv(file_path, parse_dates=parse_dates)


# display data
def display_dataframe_row_and_columns(df: pd.DataFrame) -> None:
    print(f"row:{df.shape[0]}, column:{df.shape[1]}")
    return


def display_dataframe_info(df: pd.DataFrame) -> None:
    df.info()
    return


def display_dataframe_startdate_and_enddate(df: pd.DataFrame) -> None:
    print(f"start_date:{df['dteday'].min()}, end_date:{df['dteday'].max()}")
    return


# Average for each time period within the specified period
def calc_cnt_average_each_month_for_the_period(
    df: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:

    if "hr" not in df.columns:  # error
        print("The dataframe does not contain a column named hr.")
        return pd.DataFrame()

    else:
        df_for_the_period = df[
            (df["dteday"] >= pd.to_datetime(start_date))
            & (df["dteday"] <= pd.to_datetime(end_date))
        ]

        df_for_the_period["datetime"] = pd.to_datetime(  # add "datetime" column
            df_for_the_period["dteday"]
        ) + pd.to_timedelta(df_for_the_period["hr"], unit="h")

        monthly_avg_cnt = (
            df_for_the_period.resample(rule="ME", on="datetime")[["cnt"]]
            .mean()
            .reset_index()
        )  # 平均を算出
        return monthly_avg_cnt


def calc_basic_stat(df: pd.DataFrame) -> dict[str, float]:
    basic_statistics: dict[str, float] = {}
    avg_cnt = cast(float, df.mean()["cnt"])
    max_cnt = cast(float, df.max()["cnt"])
    min_cnt = cast(float, df.min()["cnt"])
    med_cnt = cast(float, df.median()["cnt"])
    basic_statistics["avg"] = avg_cnt
    basic_statistics["max"] = max_cnt
    basic_statistics["min"] = min_cnt
    basic_statistics["med"] = med_cnt
    return basic_statistics


def display_basic_stat(basic_stat: dict[str, float]) -> str:
    basic_stat_text = f"cntの平均：{basic_stat['avg']}<br>cntの最大値：{basic_stat['max']}<br>cntの最小値：{basic_stat['min']}<br>cntの中央値：{basic_stat['med']}"
    return basic_stat_text


# create chart
def create_contents(text: str):
    st.markdown(
        text,
        unsafe_allow_html=True,
    )


def create_line_chart(
    df: pd.DataFrame, horintal_axis_data: str, vertical_axis_data: str
) -> None:
    fig = px.line(df, x=horintal_axis_data, y=vertical_axis_data)
    fig.update_xaxes(tickformat="%Y/%m/%d")
    st.plotly_chart(fig, width="stretch")


def create_bar_chart(
    df: pd.DataFrame, horintal_axis_data: str, vertical_axis_data: str
) -> None:
    fig = px.bar(df, x=horintal_axis_data, y=vertical_axis_data)
    fig.update_xaxes(tickformat="%Y/%m")
    st.plotly_chart(fig, width="stretch")


def create_histgram_chart(
    df: pd.DataFrame, horintal_axis_data: str, vertical_axis_data: str
) -> None:
    fig = px.histogram(df, x=horintal_axis_data, y=vertical_axis_data)
    fig.update_xaxes(tickformat="%Y/%m")
    st.plotly_chart(fig, width="stretch")


# 実行
# データ読み込み
hour_data = load_csv_data(HOUR_CSV_FILE_PATH, ["dteday"])
day_data = load_csv_data(DAY_CSV_FILE_PATH, ["dteday"])

dict_basic_stat_hour_data = calc_basic_stat(hour_data)
dict_basic_stat_day_data = calc_basic_stat(day_data)
text_basic_stat_hour_data = display_basic_stat(dict_basic_stat_hour_data)
text_basic_stat_day_data = display_basic_stat(dict_basic_stat_day_data)

monthly_cnt_avg = calc_cnt_average_each_month_for_the_period(
    hour_data, "2011-01-01", "2012-12-31"
)


# frontend
st.set_page_config(layout="wide")

st.header("1. 生データ")

st.subheader("1.1 hour.csv（時間ごとのデータ）")
st.dataframe(hour_data, hide_index=True)

st.subheader("1.2 day.csv（日にちごとのデータ）")
st.dataframe(day_data, hide_index=True)

st.subheader("1.3 cnt推移")
create_line_chart(day_data, "dteday", "cnt")

st.header("2. データ理解")
st.subheader("2.1 基本統計量（hour.csv）")
create_contents(text_basic_stat_hour_data)

st.subheader("2.2 基本統計量（day.csv）")
create_contents(text_basic_stat_day_data)

st.subheader("2.2 cnt分布")
st.subheader("2.3 月毎の平均cnt")
create_bar_chart(monthly_cnt_avg, "datetime", "cnt")
