import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# Data for bot statuses over time
times = ['9:00', '9:15', '9:30', '9:45', '10:00', '10:15', '10:30', '10:45', '11:00']
status_mapping = {
    1: 'Started',
    2: 'Running',
    3: 'Error',
    4: 'Completed',
    5: 'Not started'
}
bots_status = {
    'bot1': ['Started', 'Running', 'Running', 'Running', 'Completed'],
    'bot2': ['Started', 'Running', 'Running', 'Running', 'Running'],
    'bot3': ['Not started'],
    'bot4': ['Started', 'Error'],
    'bot5': ['Started', 'Running', 'Completed'],
    'bot6': ['Not started', 'Not started', 'Started', 'Running', 'Completed'],
    'bot7': ['Started', 'Error'],
    'bot8': ['Not started', 'Started', 'Running', 'Running', 'Running', 'Running', 'Error']
}

# Status to color mapping
status_colors = {
    'Started': 'yellow',
    'Running': 'blue',
    'Error': 'red',
    'Completed': 'green',
    'Not started': 'gray'
}

# Convert time to x-axis values (numeric)
time_values = np.linspace(0, len(times) - 1, 100)  # More points for smoother animation

# Set up the figure and axis
fig, ax = plt.subplots()

ax.set_xticks(np.arange(len(times)))
ax.set_xticklabels(times)
ax.set_yticks(np.arange(len(bots_status)))
ax.set_yticklabels(bots_status.keys())
ax.set_xlim(0, len(times) - 1)
ax.set_ylim(-0.5, len(bots_status) - 0.5)
ax.set_xlabel('Time')
ax.set_ylabel('Bots')
fig.patch.set_facecolor('#f5f5f5')

# Initialize an empty line object for each bot
lines = []
for i in range(len(bots_status)):
    line, = ax.plot([], [], lw=2)  # Initial line without markers
    lines.append(line)

# Function to find the start time of each bot
def find_start_time(statuses):
    for i, status in enumerate(statuses):
        if status != 'Not started':
            return i
    return len(statuses)  # If all statuses are "Not started"

# Function to update each bot's line for the current frame
def update(frame):
    for i, (bot, statuses) in enumerate(bots_status.items()):
        x_data = []
        y_data = []
        colors = []

        # Calculate the percentage of completion based on the frame
        total_time_points = len(time_values)
        time_progress = frame / total_time_points * (len(times) - 1)  # Progress in time units

        # Find the bot's actual start time
        start_time_index = find_start_time(statuses)

        # Determine the current status for each bot based on time progress
        current_index = int(time_progress)

        # Only start drawing the line after the bot has started
        if current_index >= start_time_index:
            # Loop through each time period up to the current frame
            for j in range(start_time_index, current_index + 1):
                # Check if the current index exceeds the length of the bot's statuses
                if j >= len(statuses):
                    break  # Stop the line here if no more status values exist

                x_data.append(j)  # Time values on the x-axis
                y_data.append(i)  # Bot index on the y-axis
                colors.append(status_colors[statuses[j]])  # Get color for the current status

            # Now add the segment currently in progress, if within bounds
            if current_index < len(times) - 1 and current_index < len(statuses):
                x_data.append(time_progress)
                y_data.append(i)
                colors.append(status_colors[statuses[current_index]])  # Use current status color

            # Create a multi-colored line by breaking the segments based on status changes
            for k in range(len(x_data) - 1):
                ax.plot([x_data[k], x_data[k + 1]], [y_data[k], y_data[k + 1]], color=colors[k], lw=2)

            # Set the data for the line object
            lines[i].set_data(x_data, y_data)

    return lines

# Create the animation
ani = FuncAnimation(fig, update, frames=len(time_values), blit=False, interval=50, repeat=False)

# Display the animation
plt.show()
