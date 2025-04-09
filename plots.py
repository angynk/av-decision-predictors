'''import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data from CSV
df = pd.read_csv("ttc_data.csv")

# Optional: Convert European decimal commas to dots (if needed)
df[['YOLOV', 'Ours']] = df[['YOLOV', 'Ours']].replace(',', '.', regex=True).astype(float)


# Set Seaborn style
sns.set(style="whitegrid", palette="pastel", font_scale=1.2)

# -------- BOXPLOT: TTC Distributions for A, B, C --------
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[['YOLOV', 'Ours']])
plt.title("TTC Distribution Comparison betwween YOLOV and Ours Agents")
plt.xlabel("Agents")
plt.ylabel("TTC (Time to Collision)")
plt.ylim(-1, 6)
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- LINE PLOT: TTC Trends Across Scenarios --------
plt.figure(figsize=(14, 6))
sns.lineplot(x="SCENARIO ID", y="value", hue="variable",
             data=pd.melt(df, id_vars=["SCENARIO ID"], value_vars=['YOLOV', 'Ours']),
             marker="o")

plt.title("TTC Trends Across Scenarios")
plt.xlabel("Scenario")
plt.ylabel("TTC (Time to Collision)")
plt.xticks(rotation=90)
plt.ylim(-1, 6)
plt.legend(title="Agents")
plt.grid(True)
plt.tight_layout()
plt.show()'''

'''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV data
data = pd.read_csv("anticipation.csv")  # Replace with your actual CSV file path

data[['YOLOV', 'Ours']] = data[['YOLOV', 'Ours']].replace(',', '.', regex=True).astype(float)

# The data is in wide format, so we need to melt it to long format for easier plotting
data_melted = data.melt(id_vars=["SCENARIO ID"], value_vars=['YOLOV', 'Ours'], 
                        var_name="Category", value_name="Anticipation Time")

# Boxplot comparison between categories A, B, and C
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Anticipation Time', data=data_melted, palette="Set2")
plt.title('Comparison of Anticipation Time Distributions Across t')
plt.xlabel('Category (A, B, C)')
plt.ylabel('Anticipation Time (seconds)')
plt.show()

# Histogram showing the distribution of anticipation values for each category
plt.figure(figsize=(10, 6))
sns.histplot(data=data_melted, x='Anticipation Time', hue='Category', multiple="stack", kde=True, palette="Set2", bins=20)
plt.title('Distribution of Anticipation Pedestrian Detection Times for Each Agent')
plt.xlabel('Anticipation Time (seconds)')
plt.ylabel('Frequency')
plt.show()'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV
df = pd.read_csv('collision_intensity.csv')  # Adjust filename as needed

df[['Cautious', 'Agressive', 'Normal', 'Basic', 'YOLOV', 'Ours']] = df[['Cautious', 'Agressive', 'Normal', 'Basic', 'YOLOV', 'Ours']].replace(',', '.', regex=True).astype(float)


df = df.melt(id_vars='Scenario', var_name='Agent', value_name='CollisionIntensity')
# Set a nice style
sns.set(style="whitegrid", font_scale=1.2)

# --- Boxplot ---
plt.figure(figsize=(10, 6))
sns.boxplot(x="Agent", y="CollisionIntensity", data=df, palette="Set2")
plt.title("Collision Intensity Distribution per Agent (Boxplot)")
plt.ylabel("Collision Intensity")
plt.xlabel("Agent")
plt.tight_layout()
plt.show()

# --- Violin Plot ---
plt.figure(figsize=(10, 6))
sns.violinplot(x="Agent", y="CollisionIntensity", data=df, palette="Set3", inner="box", scale="width")
plt.title("Collision Intensity Distribution per Agent")
plt.ylabel("Collision Intensity")
plt.xlabel("Agent")
plt.tight_layout()
plt.show()


'''df_melted = df.melt(id_vars='Scenario', var_name='Agent', value_name='CollisionIntensity')


# Identify top N scenarios by max collision intensity across agents
top_n = 5
top_scenarios = (
    df_melted.groupby("Scenario")["CollisionIntensity"]
    .max()
    .nlargest(top_n)
    .index.tolist()
)

# ---- PLOT ----

# Set plot aesthetics
sns.set(style="whitegrid", palette="Set2", font_scale=1.1)
plt.figure(figsize=(15, 6))

# Lineplot of collision intensity
sns.lineplot(
    data=df_melted,
    x='Scenario',
    y='CollisionIntensity',
    hue='Agent',
    marker='o'
)

# Highlight top scenarios with vertical lines
for scenario in top_scenarios:
    plt.axvline(x=scenario, color='red', linestyle='--', alpha=0.3)

# Annotate top scenarios
for scenario in top_scenarios:
    plt.text(
        x=scenario,
        y=df_melted[df_melted['Scenario'] == scenario]['CollisionIntensity'].max() + 0.3,
        s='🔥',
        ha='center',
        fontsize=12,
        color='red'
    )

# ---- FORMATTING ----

plt.title("Collision Intensity Trends Across Scenarios for slow ego-vehicle profile", fontsize=16)
plt.xlabel("Scenario")
plt.ylabel("Collision Intensity")
plt.xticks(rotation=45, ha='right')
plt.ylim(0, df_melted["CollisionIntensity"].max() + 1)
plt.legend(title="Agent", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()'''