import pandas as pd 
import random
from typing import Literal, TypeAlias

FightType: TypeAlias = Literal["UPCOMING", "COMPLETED"]

class FightProcessor:
    def __init__(self):
        self.completed_fights = pd.read_excel("./completed_fights.xlsx")
        self.upcoming_fights = pd.read_excel("./upcoming_fights.xlsx")


    def get_fights(self, type: FightType) -> dict:
        print("Fetching fighter matchups")

        if type == "UPCOMING":
            fights = list(zip(self.upcoming_fights['R_fighter'], self.upcoming_fights['B_fighter'], self.upcoming_fights['Date']))
        else:
            fights = list(zip(self.completed_fights['R_fighter'], self.completed_fights['B_fighter'], self.completed_fights['date']))

        fight_log = {}

        for fight in fights:
            fight_log[fight] = {
                fight[0]: [],
                fight[1]: [],
                "date": fight[2]
            }

            if type == "COMPLETED":
                fight_log[fight]["Winner"] = ""

        return fight_log

    def get_fighter_historical_data(self, fighter, fight_date) -> list:
        fight_data = []
        fights = self.completed_fights.loc[((self.completed_fights['R_fighter'] == fighter) | 
                                          (self.completed_fights['B_fighter'] == fighter)) &
                                          (self.completed_fights['date'] < fight_date)]

        for _, row in fights.iterrows():
            fight_row = []

            if row['R_fighter'] == fighter:
                fight_row.append(row['R_KD'])
                fight_row.append(row['R_SIG_STR'])
                fight_row.append(row['R_SIG_STR_pct'])
                fight_row.append(row['R_TOTAL_STR'])
                fight_row.append(row['R_TD'])
                fight_row.append(row['R_TD_pct'])
                fight_row.append(row['R_SUB_ATT'])
                fight_row.append(row['R_REV'])
                fight_row.append(row['R_CTRL'])
                fight_row.append(row['R_HEAD'])
                fight_row.append(row['R_BODY'])
                fight_row.append(row['R_LEG'])
                fight_row.append(row['R_DISTANCE'])
                fight_row.append(row['R_CLINCH'])
                fight_row.append(row['R_GROUND'])

            else:
                fight_row.append(row['B_KD'])
                fight_row.append(row['B_SIG_STR'])
                fight_row.append(row['B_SIG_STR_pct'])
                fight_row.append(row['B_TOTAL_STR'])
                fight_row.append(row['B_TD'])
                fight_row.append(row['B_TD_pct'])
                fight_row.append(row['B_SUB_ATT'])
                fight_row.append(row['B_REV'])
                fight_row.append(row['B_CTRL'])
                fight_row.append(row['B_HEAD'])
                fight_row.append(row['B_BODY'])
                fight_row.append(row['B_LEG'])
                fight_row.append(row['B_DISTANCE'])
                fight_row.append(row['B_CLINCH'])
                fight_row.append(row['B_GROUND'])

            fight_data.append(fight_row)

        df = pd.DataFrame(fight_data).T

        vertical_avg = df.mean(axis=1)

        return vertical_avg.tolist()
    
    def assign_winners(self, fighter_1, fighter_2) -> str:
        fight = self.completed_fights.loc[(
            (self.completed_fights['R_fighter'] == fighter_1) &
            (self.completed_fights['B_fighter'] == fighter_2)
            ) |
            (
                (self.completed_fights['R_fighter'] == fighter_2) &
                (self.completed_fights['B_fighter'] == fighter_1)
            )]

        if fight.empty:
            return ""
        
        try:
            name = fight.iloc[0]['Winner'].strip()
            return name ##Does not handle rematches well
        except:
            # print(fight)
            # print(fight.iloc[0])
            # print(fight.iloc[0]['Winner'])
            return ""

        
    
    def assemble_historical_df(self, fight_dict, fight_type: FightType) -> pd.DataFrame:
        print("Assembling historical fight logs")

        columns = [
                'R_fighter', 
                'B_fighter', 
                'AVG_DIFF_B_KD',
                'AVG_DIFF_B_SIG_STR',
                'AVG_DIFF_B_SIG_STR_pct',
                'AVG_DIFF_B_TOTAL_STR',
                'AVG_DIFF_B_TD',
                'AVG_DIFF_B_TD_pct',
                'AVG_DIFF_B_SUB_ATT',
                'AVG_DIFF_B_REV',
                'AVG_DIFF_B_CTRL',
                'AVG_DIFF_B_HEAD',
                'AVG_DIFF_B_BODY',
                'AVG_DIFF_B_LEG',
                'AVG_DIFF_B_DISTANCE',
                'AVG_DIFF_B_CLINCH',
                'AVG_DIFF_B_GROUND']
        
        if fight_type == "COMPLETED":
            columns.append("Winner")
        
        rows = []


        for i, fight in enumerate(fight_dict.keys()):
            R_fighter, B_fighter, _ = fight

            R_hist = fight_dict[fight][R_fighter]
            B_hist = fight_dict[fight][B_fighter]

            s1 = pd.Series(R_hist)
            s2 = pd.Series(B_hist)

            diffs = (s1 - s2)

            row = [R_fighter, B_fighter]
            row.extend(diffs.tolist()) 

            if fight_type == "COMPLETED":
                winner_binary = 1 if fight_dict[fight]["Winner"] == R_fighter else 0
                row.append(winner_binary)

            rows.append(row)

        df = pd.DataFrame(rows, columns=columns)
        return df

    def process_data(self, fight_type: FightType) -> pd.DataFrame:
        fights = self.get_fights(fight_type)

        count = 0

        for fight in fights.keys():

            fights[fight][fight[0]] = self.get_fighter_historical_data(fight[0], fight[2])
            fights[fight][fight[1]] = self.get_fighter_historical_data(fight[1], fight[2])

            if fight_type == "COMPLETED":
                fights[fight]["Winner"] = self.assign_winners(fight[0], fight[1])

            count += 1

            if count % 1000 == 0:
                print(f" {count} fights processed")
        
        return self.assemble_historical_df(fights, fight_type)
     

if __name__ == "__main__":
    processor = FightProcessor()
    processor.process_data("COMPLETED")