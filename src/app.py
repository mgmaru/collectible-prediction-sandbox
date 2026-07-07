import statistics
from pathlib import Path
from typing import cast

import pandas as pd
import plotly.express as px
import streamlit as st
from numpy._core.umath import fabs

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


def calc_cnt_average_each_time_for_the_period(
    df: pd.DataFrame, start_date: str, end_date: str
):
    if "hr" not in df.columns:  # error
        print("The dataframe does not contain a column named hr.")
        return pd.DataFrame()
    else:
        df_for_the_period = df[  # 期間内の行を抽出
            (df["dteday"] >= pd.to_datetime(start_date))
            & (df["dteday"] <= pd.to_datetime(end_date))
        ]
        # 時間帯ごとのcntを抽出
        time_and_cnt_dict: dict[int, list[int]] = {}  # キー：hr 値：cntのリスト
        for _, row in df_for_the_period.iterrows():
            key = cast(int, row["hr"])
            value = cast(int, row["cnt"])
            if key not in time_and_cnt_dict:
                time_and_cnt_dict[
                    key
                ] = []  # キーがない場合、空のリストを作ってからappend
            time_and_cnt_dict[key].append(value)  # Pythonのmapのキーの順序は保証される
        # 時間帯ごとの平均cntを計算
        time_and_cnt_avg_dict: dict[int, list[float]] = {}
        for key, value in time_and_cnt_dict.items():
            if key not in time_and_cnt_avg_dict:
                time_and_cnt_avg_dict[key] = []
            time_and_cnt_avg_dict[key].append(statistics.mean(value))
        # convert dict to DataFrame (key to column, value to row)
        df_time_and_cnt_avg = pd.DataFrame(time_and_cnt_avg_dict).T.reset_index()
        df_time_and_cnt_avg.columns = ["time_zone", "cnt_avg"]
    return df_time_and_cnt_avg


def calc_basic_stat(df: pd.DataFrame) -> dict[str, float]:
    basic_statistics: dict[str, float] = {}
    basic_statistics["avg"] = df["cnt"].mean()
    basic_statistics["max"] = cast(float, df["cnt"].max())
    basic_statistics["min"] = cast(float, df["cnt"].min())
    basic_statistics["med"] = df["cnt"].median()
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
time_and_cnt_avg = calc_cnt_average_each_time_for_the_period(
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

st.subheader("2.4 時間帯ごとの平均cnt")
create_bar_chart(time_and_cnt_avg, "time_zone", "cnt_avg")
