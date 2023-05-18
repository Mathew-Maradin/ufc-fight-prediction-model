# UFC-AI-Fight-Predictor

This project aims to collect UFC fight data from a popular website, prepare it using Python, and then run it through a decision tree AI to predict the outcomes of future fights.

Getting Started
To get started, you will need to have Python 3.x installed on your computer. You can download it from the official Python website: https://www.python.org/downloads/

Once you have Python installed, you will need to install a few additional libraries to run this project. You can install them by running the following command in your terminal or command prompt:
pip install requests beautifulsoup4 pandas scikit-learn

Scraping UFC Fight Data
The first step in this project is to scrape UFC fight data from a popular website. We will be using the website https://www.ufcstats.com/ for this purpose.

The scraper is located in the ufc_scraper.py file. To use it, simply run the file using Python:
python ufc_scraper.py
This will scrape the data and save it to a CSV file named ufc_fight_data.csv.

Preparing UFC Fight Data
Once we have scraped the data, we need to prepare it for use with the decision tree AI. This involves cleaning the data and converting it into a format that can be used by the AI.

The data preparation code is located in the ufc_data_prep.py file. To use it, simply run the file using Python:
python ufc_data_prep.py
This will load the data from ufc_fight_data.csv, clean it, and save it to a new CSV file named ufc_fight_data_cleaned.csv.

Running UFC Fight Data Through a Decision Tree AI
The final step in this project is to run the cleaned data through a decision tree AI to predict the outcomes of future fights.

The decision tree AI code is located in the ufc_decision_tree.py file. To use it, simply run the file using Python:
python ufc_decision_tree.py
This will load the cleaned data from ufc_fight_data_cleaned.csv, train the decision tree AI, and then use it to predict the outcomes of future fights.

Conclusion
This project demonstrates how to scrape UFC fight data from a website, prepare it using Python, and then run it through a decision tree AI to predict the outcomes of future fights. With some modifications, this approach can be applied to other sports and prediction tasks.
