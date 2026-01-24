# UFC-AI-Fight-Predictor

### Project Overview

This project aims to bridge the gap between historical sports data and predictive analytics. By collecting raw UFC fight data from comprehensive online statistics, the system prepares high-dimensional athlete data to be processed by a Decision Tree AI. The goal is to identify patterns in fighter performance and use them to predict the outcomes of future matchups with statistical accuracy.

### Preprocessing

**Data Pipeline:**

* **Scraping:** Raw data is harvested from `ufcstats.com` using a custom scraper. This includes historical fight results, significant strike percentages, takedown accuracy, and physical attributes like reach and height.
* **Cleaning:** The raw CSV is processed to handle missing values, normalize fighter names, and convert categorical outcomes (e.g., "Decision - Unanimous", "KO/TKO") into numerical labels suitable for machine learning.
* **Feature Engineering:** Converts raw fight-by-fight stats into rolling averages or win-loss ratios to represent a fighter's current "form" rather than just their historical total.

### Models & Results

**1. Data Collection Scraper**

* **Logic:** Utilizes `requests` and `BeautifulSoup4` to navigate the hierarchical structure of UFC event pages and individual bout details.
* **Output:** A comprehensive `ufc_fight_data.csv` file containing the foundational training data.

**2. Data Preparation Module**

* **Logic:** Employs `pandas` to transform qualitative fight summaries into quantitative feature vectors.
* **Insight:** This step is crucial for the Decision Tree, as it reduces noise and ensures that features like "Reach Advantage" are explicitly calculated for the model to weigh.

**3. Decision Tree Classifier**

* **Architecture:** Implemented via `scikit-learn`, the model splits fighter data based on the most statistically significant features (e.g., Strike Accuracy vs. Takedown Defense).
* **Pros:** Highly interpretable; the resulting "tree" allows users to see exactly which metrics (like a 5-inch reach advantage) most heavily influenced the predicted win.
* **Result:** After training on `ufc_fight_data_cleaned.csv`, the AI evaluates upcoming bouts to generate a probability-based prediction of the winner.

### Conclusion

This project demonstrates the full lifecycle of a sports analytics tool, from web scraping to predictive modeling. The use of a Decision Tree provides a transparent view into the "logic" of a fight, suggesting that certain physical and technical advantages are reliable predictors of victory. Future iterations could involve expanding this to Random Forests or Gradient Boosting models to further refine accuracy across diverse weight classes.