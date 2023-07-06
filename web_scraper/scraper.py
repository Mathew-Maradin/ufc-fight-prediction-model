from bs4 import BeautifulSoup
import time
import requests
import pandas as pd
import openpyxl
from halo import Halo
from spinners import Spinners

#Creating blank excel files with headers 
def create_excel_file():

    workbook_upcoming = openpyxl.Workbook()
    sheet = workbook_upcoming.active
    sheet['A1'] = 'R_fighter'
    sheet['B1'] = 'B_fighter'
    sheet['C1'] = 'Date'
    sheet['D1'] = 'Location'
    workbook_upcoming.save('../data/upcoming_fights.xlsx')

    workbook_completed = openpyxl.Workbook()
    sheet = workbook_completed.active
    sheet['A1'] = 'R_fighter'
    sheet['B1'] = 'B_fighter'
    sheet['C1'] = 'R_KD'
    sheet['D1'] = 'B_KD'
    sheet['E1'] = 'R_SIG_STR.'
    sheet['F1'] = 'B_SIG_STR.'
    sheet['G1'] = 'R_SIG_STR_pct'
    sheet['H1'] = 'B_SIG_STR_pct'
    sheet['I1'] = 'R_TOTAL_STR.'
    sheet['J1'] = 'B_TOTAL_STR.'
    sheet['K1'] = 'R_TD'
    sheet['L1'] = 'B_TD'
    sheet['M1'] = 'R_TD_pct'
    sheet['N1'] = 'B_TD_pct'
    sheet['O1'] = 'R_SUB_ATT'
    sheet['P1'] = 'B_SUB_ATT'
    sheet['Q1'] = 'R_REV'
    sheet['R1'] = 'B_REV'
    sheet['S1'] = 'R_CTRL'
    sheet['T1'] = 'B_CTRL'
    sheet['U1'] = 'R_HEAD'
    sheet['V1'] = 'B_HEAD'
    sheet['W1'] = 'R_BODY'
    sheet['X1'] = 'B_BODY'
    sheet['Y1'] = 'R_LEG'
    sheet['Z1'] = 'B_LEG'
    sheet['AA1'] = 'R_DISTANCE'
    sheet['AB1'] = 'B_DISTANCE'
    sheet['AC1'] = 'R_CLINCH'
    sheet['AD1'] = 'B_CLINCH'
    sheet['AE1'] = 'R_GROUND'
    sheet['AF1'] = 'B_GROUND'
    sheet['AG1'] = 'win_by'
    sheet['AH1'] = 'last_round'
    sheet['AI1'] = 'last_round_time'
    sheet['AJ1'] = 'Format'
    sheet['AK1'] = 'Referee'
    sheet['AL1'] = 'date'
    sheet['AM1'] = 'location'
    sheet['AN1'] = 'Fight_type'
    sheet['AO1'] = 'Winner'
    workbook_completed.save('../data/completed_fights.xlsx')

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

    spinner = Halo(text='Creating Excel Files', spinner='dots')
    spinner.start()
    create_excel_file() 
    spinner.stop() 

    spinner = Halo(text='Fetching links for each event!', spinner='dots')
    spinner.start()
    links = get_event_links()
    upcoming_event = links[0]
    event_list = links[1]
    spinner.stop()

    spinner = Halo(text='Fetching fight links for each fight!', spinner='dots')
    spinner.start()
    for event in event_list:
        fight_links = get_fight_links(event)
        # print("The following fights occured during the following event: " + event)
        # print(fight_links)

        # for link in fight_links:
        #     print(1)
        #     data = get_fight_data(link)
        #     break
        # break
    spinner.stop()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

__init__()