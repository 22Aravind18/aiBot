import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Excel file
df = pd.read_excel("Book1.xlsx")

# Replace NaN with 0 (or another placeholder) to indicate no activity
df = df.fillna(0)

# Define the color map for different statuses
color_map = {1: "orange", 2: "yellow", 3: "red", 4: "green", 5: "blue"}

# Convert time labels (e.g., '9AM', '9:15AM', etc.) into numerical values for plotting
time_labels = df.columns[1:]  # Extract the time columns (excluding the bot/time column)
time_mapping = {label: idx for idx, label in enumerate(time_labels)}

# Sidebar for user input
st.sidebar.header("Bot Selection")
# Create a multi-select box for bot selection
selected_bots = st.sidebar.multiselect("Select Bots:", options=df['bot/time'].unique(), default=df['bot/time'].unique())

# Filter the DataFrame based on selected bots
filtered_df = df[df['bot/time'].isin(selected_bots)]

# Create a plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each selected bot's lifecycle using thinner horizontal bars (lines)
for i, bot in enumerate(filtered_df['bot/time']):
    for time in time_labels:
        status = filtered_df.loc[i, time]
        if status in color_map:
            start_time = time_mapping[time]  # Get the numerical time for start
            # Thinner bars by reducing the height from 0.8 to 0.03 (adjustable)
            ax.broken_barh([(start_time, 1)], (i - 0.05, 0.03), facecolors=color_map[status])

# Set X-axis labels as the actual time intervals
ax.set_xticks(list(time_mapping.values()))
ax.set_xticklabels(list(time_mapping.keys()))

# Set Y-axis labels to the bot names
ax.set_yticks(range(len(filtered_df)))
ax.set_yticklabels(filtered_df['bot/time'])

# Set labels and title
ax.set_xlabel('Time')
ax.set_title('Bot Lifecycle Visualization (Thinner Lines)')

# Display the plot in Streamlit
st.pyplot(fig)

# Optionally, add a description or summary of the selected bots
st.sidebar.subheader("Summary")
st.sidebar.write("Selected Bots:")
for bot in selected_bots:
    st.sidebar.write(bot)
