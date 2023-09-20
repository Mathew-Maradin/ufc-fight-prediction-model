from bs4 import BeautifulSoup
import time
import requests
import pandas as pd
import openpyxl
from halo import Halo
from spinners import Spinners
from openpyxl import load_workbook
from fractions import Fraction

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
    fight_data = []

    response = requests.get(link)
    doc = BeautifulSoup(response.text, "html.parser")

    tables = doc.findAll('table')
    total_table = tables[0]

    data = total_table.findAll("p", {"class": "b-fight-details__table-text"})
    for i in data:
        i = i.getText().strip()
        try:
            frac = i.split(" of ")
            fraction = float(frac[0])/float(frac[1])
            fight_data.append(fraction)
        except:
            fight_data.append(i)


    specific_strikes_table = tables[2]
    strike_data = specific_strikes_table.findAll("p", {"class": "b-fight-details__table-text"})
    strike_data = strike_data[6:]
    for i in strike_data:
        i = i.getText().strip()
        try:
            frac = i.split(" of ")
            fraction = float(frac[0])/float(frac[1])
            fight_data.append(fraction)
        except:
            fight_data.append(i)

    # fight_details = doc.find("div",{"class": "b-fight-details__fight"})
    # details = fight_details.findAll("i")
    # details = [details[3].getText().strip(), details[4].getText().split(":")[1].strip(), details[6].getText().split(":",1)[1].strip(),
    #             details[8].getText().split(":")[1].strip(), details[10].getText().split(":")[1].strip()]
    
    # for i in details:
    #     fight_data.append(i)
    
    # data = doc.find("div", {"class": "b-fight-details__persons clearfix"})
    # data = data.findAll("div", {"class": "b-fight-details__person"})

    # for div in data:
    #     i_element = div.find('i', class_='b-fight-details__person-status b-fight-details__person-status_style_green')
    #     p_element = div.find('a')
    #     try:
    #         if (str(i_element.getText().strip()) == "W"):
    #             fight_data.append(str(p_element.getText().strip()))
    #     except:
    #         continue
                
    return fight_data 

def add_data_to_excel(data):
    wb = load_workbook("../data/completed_fights.xlsx")
    page = wb.active
    page.append(data)

def __init__():
    start_time = time.time()

    master_df = pd.DataFrame(columns=['R_fighter','B_fighter','R_KD','B_KD','R_SIG_STR.','B_SIG_STR.','R_SIG_STR_pct','B_SIG_STR_pct',
                                      'R_TOTAL_STR.','B_TOTAL_STR.','R_TD','B_TD','R_TD_pct','B_TD_pct','R_SUB_ATT','B_SUB_ATT','R_REV',
                                      'B_REV','R_CTRL','B_CTRL','R_HEAD','B_HEAD','R_BODY','B_BODY','R_LEG','B_LEG','R_DISTANCE','B_DISTANCE',
                                      'R_CLINCH','B_CLINCH','R_GROUND','B_GROUND','win_by','last_round','last_round_time','Format','Referee',
                                      'date','location','Fight_type','Winner'])
    print(master_df)

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

    # spinner = Halo(text='Fetching fight links for each fight!', spinner='dots')
    # spinner.start()
    for event in event_list:
        fight_links = get_fight_links(event)
        # print("The following fights occured during the following event: " + event)
        # print(fight_links)

        for link in fight_links:
            data = get_fight_data(link)
            print(data)
            # master_df.concat(data)
        # break
    print(master_df)
    # spinner.stop()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

__init__()