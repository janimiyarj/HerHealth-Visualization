import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import re
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns


# Load your dataset
df = pd.read_excel("Final_HerHealthAI_InterviewSheet.xlsx")  # Update path if needed

# Define keywords
symptoms_keywords = [
    "acne", "hair loss", "facial hair", "irregular periods", "cramps", "mood swings",
    "sticky blood", "weight gain", "irregular cycles", "bleeding", "fatigue"
]

treatment_keywords = [
    "pills", "birth control", "hormonal", "exercise", "diet", "lifestyle", "homeopathy", "gym",
    "no treatment", "meditation", "sleep", "healthy food"
]

# --- Keyword extract helper ---
def extract_keywords(text, keywords):
    if pd.isnull(text):
        return []
    text = str(text).lower()
    found = [word for word in keywords if re.search(r'\b' + re.escape(word) + r'\b', text)]
    return list(set(found))  # remove duplicates

# --- Doctor visit detection ---
def classify_doctor(text):
    if pd.isnull(text):
        return "Unknown"
    text = str(text).lower()
    return "Yes" if "doctor" in text or "gynac" in text else "No"

# --- Impact classification ---
def classify_impact(text):
    if pd.isnull(text): return "Unknown"
    text = text.lower()
    if "no impact" in text or "didn’t feel" in text or "not impacted" in text:
        return "No Impact"
    elif "moderate" in text:
        return "Moderate"
    elif "very significant" in text or "very impactful" in text:
        return "Very Significant"
    elif "significant" in text or "tough time" in text:
        return "Significant"
    elif "stress" in text or "can't control" in text:
        return "High"
    else:
        return "Slight or Unclear"

# --- Learned Category classification ---
def classify_learning(text):
    if pd.isnull(text): return "Unknown"
    t = str(text).lower()
    if "doctor" in t and "generalize" in t: return "Doctors Generalize"
    if "ignored" in t or "nothing" in t: return "Ignored or Nothing New"
    if "pills" in t and "consequence" in t: return "Pills Temporary"
    if "awareness" in t or "gap" in t: return "Awareness Gap"
    return "Other"

# --- Apply everything ---
def structure_survey_data(df):
    df["SYMPTOM_TAGS"] = df["SUPPORTED HP (What health problems/experiences support your view?)"].apply(
        lambda x: extract_keywords(x, symptoms_keywords)
    )
    df["DISPUTED_TAGS"] = df["Disputed HP (What health problems/experiences do you question?)"].apply(
        lambda x: extract_keywords(x, symptoms_keywords)
    )
    df["TREATMENT_TAGS"] = df["Other Info/What Currently Doing (How are you currently managing your health?)"].apply(
        lambda x: extract_keywords(x, treatment_keywords)
    )
    df["DOCTOR_VISIT"] = df["Disputed HP (What health problems/experiences do you question?)"].apply(classify_doctor)
    df["IMPACT_LEVEL"] = df["Learned Something New (Did the interview change your perspective?)"].apply(classify_impact)
    df["LEARNED_CATEGORY"] = df["Learned Something New (Did the interview change your perspective?)"].apply(classify_learning)
    return df

# Run structuring
df = structure_survey_data(df)

# Optional: Save it to file
# df.to_excel("HerHealth_Structured_Output.xlsx", index=False)

# Preview
print(df[[
    "NAME", "SYMPTOM_TAGS", "DISPUTED_TAGS", "TREATMENT_TAGS", 
    "DOCTOR_VISIT", "IMPACT_LEVEL", "LEARNED_CATEGORY"
]].head(10))
print(df.columns)


# 1. Top 10 Reported Symptoms
def plot_symptoms(df):
    all_symptoms = [s for row in df["SYMPTOM_TAGS"].dropna() for s in row]
    symptom_counts = Counter(all_symptoms)
    top_symptoms = dict(symptom_counts.most_common(10))

    plt.figure(figsize=(10, 5))
    plt.bar(top_symptoms.keys(), top_symptoms.values(), alpha=0.8)
    plt.title("Top 10 Reported PCOS-Related Symptoms")
    plt.xlabel("Symptom")
    plt.ylabel("Number of Respondents")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Top_10_Symptoms.png")
    plt.show(block=False)
    plt.pause(3)  # show for 3 seconds
    plt.close()

# 2. Doctor Visit Pie Chart
def plot_doctor_visit(df):
    visit_counts = df["DOCTOR_VISIT"].value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(visit_counts, labels=visit_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title("Doctor Visit Mentioned?")
    plt.tight_layout()
    plt.savefig("Doctor_Visit_Pie.png")
    plt.show(block=False)
    plt.pause(3)  # show for 3 seconds
    plt.close()

# 3. Learning Categories
def plot_learnings(df):
    learning_counts = df["LEARNED_CATEGORY"].value_counts()
    plt.figure(figsize=(10, 4))
    plt.bar(learning_counts.index, learning_counts.values, color="green", alpha=0.75)
    plt.title("Key Learnings Shared by Respondents")
    plt.xlabel("Learning Type")
    plt.ylabel("Frequency")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig("Learning_Categories.png")
    plt.show(block=False)
    plt.pause(3)  # show for 3 seconds
    plt.close()
    

# 4. Treatment Tags Bar Chart
def plot_treatments(df):
    all_treatments = [t for row in df["TREATMENT_TAGS"].dropna() for t in row]
    treatment_counts = Counter(all_treatments)
    top_treatments = dict(treatment_counts.most_common(10))

    plt.figure(figsize=(10, 5))
    plt.bar(top_treatments.keys(), top_treatments.values(), color="purple", alpha=0.75)
    plt.title("Top 10 Mentioned Treatment/Lifestyle Choices")
    plt.xlabel("Treatment Type")
    plt.ylabel("Mentions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Top_10_Treatments.png")
    plt.show(block=False)
    plt.pause(3)  # show for 3 seconds
    plt.close()

# --- Visualization 1: Symptom frequency ---
all_symptoms = [symptom for sublist in df["SYMPTOM_TAGS"].dropna() for symptom in sublist]
symptom_counts = Counter(all_symptoms)
labels, values = zip(*symptom_counts.most_common())

plt.figure(figsize=(10, 5))
plt.bar(labels, values)
plt.title("Top Reported PCOS/Menstrual Symptoms")
plt.ylabel("Number of Mentions")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("Top_Symptoms.png")
plt.show(block=False)
plt.pause(3)  # show for 3 seconds
plt.close()


# --- Visualization 2: Treatment methods ---
all_treatments = [treat for sublist in df["TREATMENT_TAGS"].dropna() for treat in sublist]
treatment_counts = Counter(all_treatments)
t_labels, t_values = zip(*treatment_counts.most_common())

plt.figure(figsize=(10, 5))
plt.bar(t_labels, t_values)
plt.title("Treatment and Management Strategies")
plt.ylabel("Number of Mentions")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("Top_Treatments_methods.png")
plt.show(block=False)
plt.pause(3)  # show for 3 seconds
plt.close()

# --- Visualization 3: Doctor visit pie chart ---
doctor_counts = df["DOCTOR_VISIT"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(doctor_counts, labels=doctor_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Doctor Consultation Mentioned")
plt.tight_layout()
plt.savefig("Doctor_Visit_Pie_chart.png")
plt.show(block=False)
plt.pause(3)  # show for 3 seconds
plt.close()

# --- Visualization 4: Impact Level ---
impact_counts = df["IMPACT_LEVEL"].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(impact_counts.index, impact_counts.values)
plt.title("Impact on Quality of Life")
plt.ylabel("Number of Respondents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Impact_on_Quality_of_Life.png")
plt.show(block=False)
plt.pause(3)  # show for 3 seconds
plt.close()

# --- Visualization 5: Learned Category ---
learned_counts = df["LEARNED_CATEGORY"].value_counts()
plt.figure(figsize=(8, 5))
plt.bar(learned_counts.index, learned_counts.values)
plt.title("Learnings or Awareness Gained")
plt.ylabel("Number of Respondents")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Learnings_Category_Awareness.png")
plt.show(block=False)
plt.pause(3)  # show for 3 seconds
plt.close()

df["Doctor_Group"] = df["DOCTOR_VISIT"].fillna("Unknown")

treatment_counts_by_doctor = df.groupby("Doctor_Group")["TREATMENT_TAGS"].apply(lambda tags: sum([len(t) for t in tags if isinstance(t, list)]))

plt.figure(figsize=(7, 5))
treatment_counts_by_doctor.plot(kind='bar', color="teal")
plt.title("Number of Management Strategies by Doctor Visit")
plt.xlabel("Doctor Visit Status")
plt.ylabel("Total Strategies Mentioned")
plt.tight_layout()
plt.savefig("Doctor_vs_Management.png")
plt.show(block=False)
plt.pause(3)
plt.close()


magic_text = df["MAGIC WAND (If you had a magic wand, how would you solve your PCOS challenges?)"].dropna().astype(str).str.lower().str.cat(sep=" ")
words = re.findall(r'\b[a-z]{4,}\b', magic_text)  # 4+ letter words to remove filler
common = Counter(words).most_common(10)

labels, values = zip(*common)
plt.figure(figsize=(10, 5))
plt.bar(labels, values, color="crimson")
plt.title("Most Common Words in Magic Wand Wishes")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("MagicWand_TopWords.png")
plt.show(block=False)
plt.pause(3)
plt.close()


# --- Call each plot function ---
plot_symptoms(df)
plot_doctor_visit(df)
plot_learnings(df)
plot_treatments(df)

# Word cloud generator
def generate_wordcloud(df, column_name, title):
    text = df[column_name].dropna().astype(str).str.lower().str.cat(sep=" ")
    wordcloud = WordCloud(width=1000, height=400, background_color='white', colormap='viridis').generate(text)
    
    plt.figure(figsize=(12, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

# Call the function for each column
generate_wordcloud(df, "SUPPORTED HP (What health problems/experiences support your view?)", 
                   "Word Cloud: Supported Health Problems (HP)")

generate_wordcloud(df, "Disputed HP (What health problems/experiences do you question?)", 
                   "Word Cloud: Disputed Health Problems (HP)")

generate_wordcloud(df, "Learned Something New (Did the interview change your perspective?)", 
                   "Word Cloud: Learnings from Interview")

generate_wordcloud(df, "Other Info/What Currently Doing (How are you currently managing your health?)", 
                   "Word Cloud: Current Health Management")

generate_wordcloud(df, "MAGIC WAND (If you had a magic wand, how would you solve your PCOS challenges?)", 
                   "Word Cloud: Magic Wand Wishes")


mlb = MultiLabelBinarizer()
symptom_matrix = pd.DataFrame(mlb.fit_transform(df["SYMPTOM_TAGS"]), columns=mlb.classes_, index=df.index)

# Correlation matrix for symptoms
corr = symptom_matrix.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", square=True, linewidths=0.5)
plt.title("Symptom Co-occurrence Heatmap")
plt.tight_layout()
plt.savefig("Symptom_CoOccurrence_Heatmap.png")
plt.show(block=False)
plt.pause(3)
plt.close()


df["Doctor_Group"] = df["DOCTOR_VISIT"].fillna("Unknown")

treatment_counts_by_doctor = df.groupby("Doctor_Group")["TREATMENT_TAGS"].apply(lambda tags: sum([len(t) for t in tags if isinstance(t, list)]))

plt.figure(figsize=(7, 5))
treatment_counts_by_doctor.plot(kind='bar', color="teal")
plt.title("Number of Management Strategies by Doctor Visit")
plt.xlabel("Doctor Visit Status")
plt.ylabel("Total Strategies Mentioned")
plt.tight_layout()
plt.savefig("Doctor_vs_Management.png")
plt.show(block=False)
plt.pause(3)
plt.close()


magic_text = df["MAGIC WAND (If you had a magic wand, how would you solve your PCOS challenges?)"].dropna().astype(str).str.lower().str.cat(sep=" ")
words = re.findall(r'\b[a-z]{4,}\b', magic_text)  # 4+ letter words to remove filler
common = Counter(words).most_common(10)

labels, values = zip(*common)
plt.figure(figsize=(10, 5))
plt.bar(labels, values, color="crimson")
plt.title("Most Common Words in Magic Wand Wishes")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("MagicWand_TopWords.png")
plt.show(block=False)
plt.pause(3)
plt.close()



df["DOCTOR_GROUP"] = df["DOCTOR_VISIT"].fillna("Unknown")
df["TREATMENT_COUNT"] = df["TREATMENT_TAGS"].apply(lambda x: len(x) if isinstance(x, list) else 0)

grouped = df.groupby("DOCTOR_GROUP")["TREATMENT_COUNT"].mean()

plt.figure(figsize=(8, 5))
grouped.plot(kind='bar', color="salmon")
plt.title("Avg. Self-Management Actions Taken (With vs Without Doctors)")
plt.ylabel("Avg. Number of Strategies")
plt.xlabel("Doctor Consultation")
plt.tight_layout()
plt.savefig("Avg_SelfManagement_Actions.png")
plt.show(block=False)
plt.pause(3)
plt.close()

def categorize_magic_wand(text):
    if pd.isnull(text): return "Unspecified"
    t = str(text).lower()
    if "vanish" in t or "remove" in t or "eliminate" in t: return "Eliminate PCOS"
    if "doctor" in t or "healthcare" in t: return "Better Medical Support"
    if "track" in t or "app" in t or "tool" in t: return "Tech/Tracking Support"
    if "pain" in t or "cramps" in t: return "Pain-Free Periods"
    if "confidence" in t or "feel normal" in t: return "Self-Confidence"
    return "Other"

df["MAGIC_WAND_CATEGORY"] = df["MAGIC WAND (If you had a magic wand, how would you solve your PCOS challenges?)"].apply(categorize_magic_wand)
cat_counts = df["MAGIC_WAND_CATEGORY"].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("If Women Had a Magic Wand...")
plt.tight_layout()
plt.savefig("MagicWand_Categories.png")
plt.show(block=False)
plt.pause(3)
plt.close()


from sklearn.preprocessing import MultiLabelBinarizer

mlb_sym = MultiLabelBinarizer()
mlb_trt = MultiLabelBinarizer()

symptom_df = pd.DataFrame(mlb_sym.fit_transform(df["SYMPTOM_TAGS"]), columns=mlb_sym.classes_)
treatment_df = pd.DataFrame(mlb_trt.fit_transform(df["TREATMENT_TAGS"]), columns=mlb_trt.classes_)

correlation_matrix = symptom_df.T.dot(treatment_df)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap="YlGnBu", linewidths=.5, annot=True)
plt.title("Symptom-to-Treatment Action Map")
plt.xlabel("Treatment Actions")
plt.ylabel("Symptoms Experienced")
plt.tight_layout()
plt.savefig("Symptom_Treatment_Heatmap.png")
plt.show(block=False)
plt.pause(3)
plt.close()


from textblob import TextBlob

# Replace the column name with the correct one from your dataset
df["MAGIC_WAND_POLARITY"] = df["MAGIC WAND (If you had a magic wand, how would you solve your PCOS challenges?)"].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0
)


plt.figure(figsize=(10, 5))
sns.histplot(df["MAGIC_WAND_POLARITY"], bins=30, kde=True, color="purple")
plt.title("Emotional Polarity in Magic Wand Responses")
plt.xlabel("Polarity Score (-1 to +1)")
plt.ylabel("Number of Respondents")
plt.axvline(0, color='red', linestyle='--')
plt.tight_layout()
plt.savefig("MagicWand_EmotionalPolarity.png")
plt.show(block=False)
plt.pause(3)
plt.close()

df["DOCTOR_GROUP"] = df["DOCTOR_VISIT"].fillna("Unknown")
df["TREATMENT_COUNT"] = df["TREATMENT_TAGS"].apply(lambda x: len(x) if isinstance(x, list) else 0)

grouped = df.groupby("DOCTOR_GROUP")["TREATMENT_COUNT"].mean()

plt.figure(figsize=(8, 5))
grouped.plot(kind='bar', color="salmon")
plt.title("Avg. Self-Management Actions Taken (With vs Without Doctors)")
plt.ylabel("Avg. Number of Strategies")
plt.xlabel("Doctor Consultation")
plt.tight_layout()
plt.savefig("Avg_SelfManagement_Actions.png")
plt.show(block=False)
plt.pause(3)
plt.close()

# Column name shortcuts for readability
LEARNED_COL = "Learned Something New (Did the interview change your perspective?)"
MANAGEMENT_COL = "Other Info/What Currently Doing (How are you currently managing your health?)"
MAGIC_WAND_COL = "MAGIC WAND (If you had a magic wand, how would you solve your PCOS challenges?)"

import seaborn as sns

sentiment_df = pd.DataFrame({
    "Learned": df["Learned Something New (Did the interview change your perspective?)"].apply(lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0),
    "Magic Wand": df["MAGIC WAND (If you had a magic wand, how would you solve your PCOS challenges?)"].apply(lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0),
    "Management": df["Other Info/What Currently Doing (How are you currently managing your health?)"].apply(lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0),
})

sentiment_df_melted = sentiment_df.melt(var_name="Category", value_name="Polarity")

plt.figure(figsize=(10, 6))
sns.violinplot(x="Category", y="Polarity", data=sentiment_df_melted, palette="coolwarm")
plt.title("Sentiment Distribution: Lived vs. Desired Experience")
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.savefig("Sentiment_Distribution.png")
plt.show(block=False)
plt.pause(3)
plt.close()


stages = {
    "Experienced Symptoms": df["SYMPTOM_TAGS"].apply(lambda x: len(x) > 0).sum(),
    "Consulted Doctor": df["DOCTOR_VISIT"].apply(lambda x: x == "Yes").sum(),
    "Mentioned PCOS/Diagnosis": df["SUPPORTED HP (What health problems/experiences support your view?)"].str.contains("pcos", case=False, na=False).sum(),
    "Tried Treatment": df["TREATMENT_TAGS"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum(),
    "Shared Positive Learning": df[LEARNED_COL].apply(lambda x: TextBlob(str(x)).sentiment.polarity > 0.2 if pd.notnull(x) else False).sum()
}

plt.figure(figsize=(10, 5))
sns.barplot(x=list(stages.keys()), y=list(stages.values()), palette="magma")
plt.title("Pain to Power: Journey Through the PCOS Experience")
plt.ylabel("Number of Respondents")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("Pain_to_Power_Journey.png")
plt.show(block=False)
plt.pause(3)
plt.close()

# Add this once you've done theme tagging (manually or through keywords)
DISPUTED_COL = "Disputed HP (What health problems/experiences do you question?)"

themes = {
    "Doctor Dismissal": df["LEARNED_CATEGORY"].str.contains("doctor", case=False, na=False).sum(),
    "Cultural Shame": df[MAGIC_WAND_COL].str.contains("normal|confidence|society|shame", case=False, na=False).sum(),
    "Self-doubt": df[LEARNED_COL].str.contains("identity|feel like woman|feminine", case=False, na=False).sum(),
    "Diet Pressure": df[MANAGEMENT_COL].str.contains("diet|food|control|weight", case=False, na=False).sum(),
    "Pain Ignored": df[DISPUTED_COL].str.contains("pain|ignored|overlooked", case=False, na=False).sum()
}


# Radar Chart
labels = list(themes.keys())
values = list(themes.values())

# Close the loop
labels += [labels[0]]
values += [values[0]]

angles = [n / float(len(labels)) * 2 * 3.14159 for n in range(len(labels))]

fig = plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)
ax.plot(angles, values, linewidth=2, linestyle='solid')
ax.fill(angles, values, 'teal', alpha=0.3)
ax.set_xticks(angles)
ax.set_xticklabels(labels)
plt.title("Radar of Emotional Stigma & Struggles")
plt.tight_layout()
plt.savefig("Radar_chart.png")
plt.show(block=False)
plt.pause(3)
plt.close()

# Breakdown of doctor visit vs no doctor by impact level
doctor_impact = df.groupby(['DOCTOR_VISIT', 'IMPACT_LEVEL']).size().unstack().fillna(0)

doctor_impact.T.plot(kind='bar', stacked=True, figsize=(10, 5), colormap='coolwarm')
plt.title("Life Impact by Access to Doctors")
plt.ylabel("Number of Respondents")
plt.xlabel("Impact on Life")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Impact_vs_Doctor_Access.png")
plt.show(block=False)
plt.pause(3)
plt.close()


theme_groups = {
    "🩸 Period Irregularity": df["SYMPTOM_TAGS"].apply(lambda x: "irregular periods" in x or "irregular cycles" in x).sum(),
    "⚖️ Weight/Metabolism": df["SYMPTOM_TAGS"].apply(lambda x: "weight gain" in x or "fatigue" in x).sum(),
    "🧠 Mood/Emotions": df["SYMPTOM_TAGS"].apply(lambda x: "mood swings" in x).sum(),
    "💇 Hair Issues": df["SYMPTOM_TAGS"].apply(lambda x: "hair loss" in x or "facial hair" in x).sum(),
    "😶 Skin & Acne": df["SYMPTOM_TAGS"].apply(lambda x: "acne" in x).sum()
}

plt.figure(figsize=(7, 7))
plt.pie(theme_groups.values(), labels=theme_groups.keys(), autopct='%1.1f%%', startangle=140)
plt.title("What PCOS Feels Like: Themes from Women")
plt.tight_layout()
plt.savefig("PCOS_Themes_Pie.png")
plt.show(block=False)
plt.pause(3)
plt.close()


emoji_map = {
    "pills": "💊", "exercise": "🏃", "diet": "🥗", "lifestyle": "🌿",
    "gym": "💪", "meditation": "🧘", "birth control": "⚖️", "sleep": "🛌"
}

flat_treatments = [t for tags in df["TREATMENT_TAGS"] if isinstance(tags, list) for t in tags]
treatment_counts = Counter(flat_treatments)

# Replace keys with emojis
labels = [f"{emoji_map.get(k, '')} {k}" for k, _ in treatment_counts.most_common(10)]
values = [v for _, v in treatment_counts.most_common(10)]

plt.figure(figsize=(10, 6))
plt.barh(labels, values, color="mediumseagreen")
plt.title("What Women Do on Their Own to Manage PCOS")
plt.xlabel("Number of Mentions")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("SelfManagement_EmojiBar.png")
plt.show(block=False)
plt.pause(3)
plt.close()


from textblob import TextBlob

def classify_sentiment(x):
    if pd.isnull(x): return "Neutral"
    polarity = TextBlob(str(x)).sentiment.polarity
    return "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"

df["WISH_SENTIMENT"] = df[MAGIC_WAND_COL].apply(classify_sentiment)
df["MANAGEMENT_SENTIMENT"] = df[MANAGEMENT_COL].apply(classify_sentiment)

sent_compare = pd.DataFrame({
    "Magic Wand Wishes": df["WISH_SENTIMENT"].value_counts(),
    "Current Management": df["MANAGEMENT_SENTIMENT"].value_counts()
}).fillna(0)

sent_compare.plot(kind='bar', figsize=(8, 5), colormap='Set2')
plt.title("How Women Feel: Now vs What They Wish")
plt.ylabel("Number of Responses")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("Sentiment_Comparison.png")
plt.show(block=False)
plt.pause(3)
plt.close()


def has_pcos(text):
    return bool(re.search(r'\bpcos|pcod\b', str(text).lower()))

def has_slight_pain(tags):
    if not isinstance(tags, list): return False
    return any(re.search(r'slight|mild|minor', tag.lower()) for tag in tags)

def has_heavy_pain(tags):
    if not isinstance(tags, list): return False
    return any(re.search(r'cramp|heavy|severe|intense', tag.lower()) for tag in tags)

# Create combination groups
df["PCOS_MENTIONED"] = df["SUPPORTED HP (What health problems/experiences support your view?)"].apply(has_pcos)
df["SLIGHT_PAIN"] = df["SYMPTOM_TAGS"].apply(has_slight_pain)
df["HEAVY_PAIN"] = df["SYMPTOM_TAGS"].apply(has_heavy_pain)

# Categorize into 4 groups
def categorize_group(row):
    if row["SLIGHT_PAIN"] and row["PCOS_MENTIONED"]:
        return "Slight Pain + PCOS"
    elif row["HEAVY_PAIN"] and not row["PCOS_MENTIONED"]:
        return "Heavy Pain + No PCOS"
    elif row["HEAVY_PAIN"] and row["PCOS_MENTIONED"]:
        return "Heavy Pain + PCOS"
    elif row["SLIGHT_PAIN"] and not row["PCOS_MENTIONED"]:
        return "Slight Pain + No PCOS"
    return "Other"

df["PAIN_PCOS_GROUP"] = df.apply(categorize_group, axis=1)

# Plot Bar Chart
pain_groups = df["PAIN_PCOS_GROUP"].value_counts()

plt.figure(figsize=(8, 5))
pain_groups.plot(kind='bar', color="tomato")
plt.title("Pain Intensity vs PCOS Mentions")
plt.xlabel("Group")
plt.ylabel("Number of Respondents")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("Pain_vs_PCOS_Grouping.png")
plt.show(block=False)
plt.pause(3)
plt.close()

def symptom_pie(symptom_name):
    has_symptom = df["SYMPTOM_TAGS"].apply(lambda tags: symptom_name in tags if isinstance(tags, list) else False)
    counts = has_symptom.value_counts()

    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=["Reported", "Not Reported"], autopct='%1.1f%%', colors=["skyblue", "lightgray"])
    plt.title(f"Reported {symptom_name.capitalize()}?")
    plt.tight_layout()
    plt.savefig(f"{symptom_name}_Pie.png")
    plt.show(block=False)
    plt.pause(3)
    plt.close()

# Generate for desired symptoms
symptom_pie("cramps")
symptom_pie("weight gain")
symptom_pie("acne")
symptom_pie("irregular periods")
symptom_pie("mood swings")
symptom_pie("hair loss")
symptom_pie("facial hair")
symptom_pie("fatigue")


# Flatten and count all symptom tags
all_symptoms = [s for tags in df["SYMPTOM_TAGS"].dropna() for s in tags]
symptom_freq = Counter(all_symptoms)

# Optional: pick top N symptoms
top_symptoms = dict(symptom_freq.most_common(10))

# Plot as pie chart
plt.figure(figsize=(8, 8))
plt.pie(top_symptoms.values(), labels=top_symptoms.keys(), autopct='%1.1f%%', startangle=140)
plt.title("Proportion of Most Reported PCOS Symptoms")
plt.tight_layout()
plt.savefig("All_Symptoms_PieChart.png")
plt.show(block=False)
plt.pause(3)
plt.close()


from wordcloud import WordCloud
import numpy as np
from collections import Counter

# Combine open-text fields
text_sources = [
    "Learned Something New (Did the interview change your perspective?)",
    "MAGIC WAND (If you had a magic wand, how would you solve your PCOS challenges?)",
    "Other Info/What Currently Doing (How are you currently managing your health?)"
]
combined_text = df[text_sources].fillna("").astype(str).apply(lambda x: ' '.join(x), axis=1).str.lower().str.cat(sep=" ")

# Tokenize words (ignore small fillers)
words = re.findall(r'\b[a-z]{5,}\b', combined_text)

# Count frequencies
word_counts = Counter(words)

# Reduce common/filler terms
common_words = set(["period", "health", "pain", "doctor", "treatment", "manage", "woman", "symptom", "feel", "help"])
non_obvious_counts = {word: freq for word, freq in word_counts.items() if word not in common_words and freq < 10}

# Boost rare/interesting keywords manually (if needed)
boost_words = ["ignored", "identity", "confidence", "shame", "overlooked", "empowered", "understood", "not believed", "normal girl"]
for word in boost_words:
    non_obvious_counts[word] = non_obvious_counts.get(word, 0) + 8  # Boost

# Create Word Cloud
wordcloud = WordCloud(width=1000, height=400, background_color='white', colormap='plasma').generate_from_frequencies(non_obvious_counts)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud: Non-Obvious Findings & Hidden Struggles", fontsize=16)
plt.tight_layout()
plt.savefig("Non_Obvious_Findings_WordCloud.png")
plt.show(block=False)
plt.pause(3)
plt.close()

# Define known/common terms to exclude
common_keywords = set(symptoms_keywords + treatment_keywords + [
    "pcos", "period", "pain", "doctor", "manage", "women", "health", "issue", "treatment", "feel", "experience", "support"
])

# Combine all open-response text fields
rare_text = df[
    ["SUPPORTED HP (What health problems/experiences support your view?)",
     "Disputed HP (What health problems/experiences do you question?)",
     "Other Info/What Currently Doing (How are you currently managing your health?)"]
].fillna("").astype(str).apply(lambda row: " ".join(row), axis=1).str.lower().str.cat(sep=" ")

# Tokenize and count all words (filter out common ones)
tokens = re.findall(r'\b[a-z]{5,}\b', rare_text)
word_freq = Counter(tokens)

# Filter rare and non-obvious terms
rare_conditions = {
    word: freq for word, freq in word_freq.items()
    if word_freq[word] <= 2 and word not in common_keywords
}

# Optional boost for specific manually-curated rare conditions
manual_rare = ["sticky", "blood clots", "misdiagnosed", "dry skin", "visual migraine", "insulin", "sleep apnea", "lump", "low energy", "brain fog"]
for word in manual_rare:
    rare_conditions[word] = rare_conditions.get(word, 0) + 6

# Generate the word cloud
rare_wc = WordCloud(width=1200, height=500, background_color='white', colormap='inferno').generate_from_frequencies(rare_conditions)

plt.figure(figsize=(14, 6))
plt.imshow(rare_wc, interpolation='bilinear')
plt.axis("off")
plt.title("Rare Patient-Reported Conditions", fontsize=18)
plt.tight_layout()
plt.savefig("Rare_Conditions_WordCloud.png")
plt.show(block=False)
plt.pause(3)
plt.close()

def pcod_status(row):
    mentioned_pcod = bool(re.search(r'\bpcos|pcod\b', str(row["SUPPORTED HP (What health problems/experiences support your view?)"]).lower()))
    has_symptoms = isinstance(row["SYMPTOM_TAGS"], list) and len(row["SYMPTOM_TAGS"]) > 0

    if mentioned_pcod and not has_symptoms:
        return "Diagnosed, Not Suffering Now"
    elif mentioned_pcod and has_symptoms:
        return "Diagnosed, Still Suffering"
    elif not mentioned_pcod and has_symptoms:
        return "Not Diagnosed, But Suffering"
    else:
        return "No PCOD Symptoms"

# Apply classification
df["PCOD_STATUS"] = df.apply(pcod_status, axis=1)

# Count groups
status_counts = df["PCOD_STATUS"].value_counts()

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140, colors=["lightgreen", "gold", "tomato", "lightblue"])
plt.title("PCOD Diagnosis and Suffering Status")
plt.tight_layout()
plt.savefig("PCOD_Status_Pie.png")
plt.show(block=False)
plt.pause(3)
plt.close()

