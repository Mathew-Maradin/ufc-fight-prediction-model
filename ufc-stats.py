from bs4 import BeautifulSoup
import time
import requests
import pandas as pd
# from spinners import Spinners
import openpyxl


#Creating blank excel files with headers 
def create_excel_file():

    workbook_upcoming = openpyxl.Workbook()
    sheet = workbook_upcoming.active
    sheet['A1'] = 'R_fighter'
    sheet['B1'] = 'B_fighter'
    sheet['C1'] = 'Date'
    sheet['D1'] = 'Location'
    workbook_upcoming.save('upcoming_fights.xlsx')
    print("upcoming fight book complete")

    workbook_completed = openpyxl.Workbook()
    sheet = workbook_completed.active
    sheet['A1'] = 'R_fighter'
    sheet['B1'] = 'B_fighter'
    sheet['D1'] = 'Date'
    sheet['E1'] = 'Date'
    sheet['F1'] = 'Date'
    sheet['G1'] = 'Date'
    sheet['H1'] = 'Date'
    sheet['I1'] = 'Date'
    sheet['J1'] = 'Date'
    sheet['K1'] = 'Date'
    sheet['C1'] = 'Date'
    sheet['C1'] = 'Date'
    sheet['C1'] = 'Date'
    sheet['C1'] = 'Date'
    sheet['C1'] = 'Date'
    sheet['C1'] = 'Date'
    workbook_completed.save('completed_fights.xlsx')

    # df.columns = dates
    print("Completed fight book complete")

##Returns a array with two objects, first a link of the upcoming event and an array of all completed events 
def get_event_links():
    event_list = []
    url = "http://www.ufcstats.com/statistics/events/completed?page=all"

    res = requests.get(url)
    doc = BeautifulSoup(res.text, "html.parser")

    table = doc.find('table', { 'class': 'b-statistics__table-events' })
    events = table.find("tbody")
    cards = events.find_all("a")

    for card in cards:
        href = card.get('href')
        event_list.append(href)

    upcoming_event = event_list.pop(0)

    return [upcoming_event, event_list]

##Takes in a link and returns an array of each fight that occured during that event
def get_fight_links(event_url):
    fight_links = []

    response = requests.get(event_url)
    doc = BeautifulSoup(response.text, "html.parser")

    table = doc.find('table', { 'class': 'b-fight-details__table b-fight-details__table_style_margin-top b-fight-details__table_type_event-details js-fight-table' })
    fights = table.find("tbody")
    all_fights = fights.find_all("tr", { 'class': 'b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click' })
 
    for fight in all_fights:
        link = fight.get('data-link')
        fight_links.append(link)
    
    return(fight_links)

def get_fight_data(link):
    ##Get data from each fights link and fill in according to headers 
    data = []

    response = requests.get(link)
    doc = BeautifulSoup(response.text, "html.parser")

    ## Preprocess data, convert fractions to decimals, strings to binary and maybe drop referees??

    print(link)
    return 0 

def add_data_to_excel(data):
    return 0

def __init__():
    start_time = time.time()
    print("Creating Excel Files")
    create_excel_file()  
    print("Excel Files made")

    print("Fetching links for each event!")
    # print(Spinners.line.value)
    links = get_event_links()
    upcoming_event = links[0]
    event_list = links[1]

    for event in event_list:
        fight_links = get_fight_links(event)
        print("The following fights occured during the following event: " + event)
        print(fight_links)

        # for link in fight_links:
        #     print(1)
        #     data = get_fight_data(link)
        #     break
        # break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

__init__()