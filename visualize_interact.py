import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to load the Excel file
def load_data():
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df.fillna(0)  # Replace NaN with 0
    else:
        st.warning("Please upload an Excel file.")
        return None

# Load the data
df = load_data()

if df is not None:
    # Define the default color map for different statuses with hex codes
    default_color_map = {
        1: "#FFA500",  # orange
        2: "#FFFF00",  # yellow
        3: "#FF0000",  # red
        4: "#008000",  # green
        5: "#0000FF"   # blue
    }

    # Bot selection
    st.header("Bot Lifecycle Visualization")
    selected_bots = st.multiselect("Select Bots:", options=df['bot/time'].unique(), default=df['bot/time'].unique())

    # Color selection for statuses
    st.subheader("Color Selection for Statuses")
    color_map = {}
    for status in default_color_map.keys():
        color = st.color_picker(f'Select color for status {status}', default_color_map[status])
        color_map[status] = color

    # Adjustable line height
    line_height = st.slider("Select line height", 0.01, 0.1, 0.03)

    # Filter the DataFrame based on selected bots
    filtered_df = df[df['bot/time'].isin(selected_bots)]

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each selected bot's lifecycle using thinner horizontal bars
    for i, bot in enumerate(filtered_df['bot/time']):
        for time in filtered_df.columns[1:]:  # time columns only
            status = filtered_df.loc[i, time]
            if status in color_map:
                start_time = list(filtered_df.columns[1:]).index(time)  # Get the numerical time for start
                ax.broken_barh([(start_time, 1)], (i - line_height / 2, line_height), facecolors=color_map[status])
                # Tooltip with status value
                ax.text(start_time + 0.5, i, str(status), ha='center', va='center', color='white', fontsize=8)

    # Set X-axis labels as the actual time intervals
    ax.set_xticks(range(len(filtered_df.columns[1:])))
    ax.set_xticklabels(filtered_df.columns[1:])

    # Set Y-axis labels to the bot names
    ax.set_yticks(range(len(filtered_df)))
    ax.set_yticklabels(filtered_df['bot/time'])

    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_title('Bot Lifecycle Visualization')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Summary of selected bots
    st.subheader("Summary of Selected Bots:")
    for bot in selected_bots:
        st.write(bot)
