import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import json

def digestData(firstYear, lastYear):

    #if a result grants 500 rows limit should be expanded in url

    url = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/mens-college-basketball/statistics/byteam?region=us&lang=en&contentorigin=espn&sort=team.offensive.avgPoints%3Adesc&limit=500&conference=50&page=1"

    statDict = dict()
    glossDict = dict()

    #  modifying for particular year requires just appending "&season=2022" or whatever year you need
    for year in range(firstYear, lastYear + 1):
        #print(year)
        #gets json data from api url
        r2 = requests.get(url + "&season=" + str(year))
        #print(r2)
        #loads json to python structure
        data = json.loads(r2.content)

        #glossary will eventually contain the ordered list of statistic names given by the api, including offensive/defensive and opponent tags
        glossary = ["teamName"]
        #"categories" stores the name data, in which are indexes containing the data for each statistic category, innermost containers include a list matching the team statistics with names for the matching statistic
        for index in range(len(data["categories"])):
            #opponent data labels are stored in their preceding container, if the check for data fails instead check the list before, append "opp_" to denote opponent statistics
            try:
                #add labels to the glossary (item) preceded by organizational information denoting statistic type
                for item in data["categories"][index]["names"]:
                    glossary.append(data["categories"][index]["name"] + "_" + item)
            except:
                #unlisted labels, must check preceding container, preceded by "opp_" tag
                for item in data["categories"][index - 1]["names"]:
                    glossary.append("opp_" + data["categories"][index]["name"] + "_" + item)

        #shows label names, mostly for debugging to confirm all is in order
        #print(glossary)
        #print(len(glossary))

        #stores glossary in year dictionary
        glossDict[year] = glossary

        #teamDictionary contains entries by team name storing all statistics in the same order as the label
        teamDictionary = dict()

        teams = data["teams"]
        for team in teams:
            #places all needed data items in an array to be added into the dictionary
            teamArray = [team["team"]["displayName"]]
            for datum in team["categories"]:
                teamArray += datum["values"]
            teamDictionary[team["team"]["displayName"]] = teamArray

        #print(teamDictionary)

        #stores dictionary in the dataframe with labels from the glossary
        dataFrame = pd.DataFrame.from_dict(teamDictionary, orient='index', columns=glossary)

        #print(dataFrame)
        #print(dataFrame.to_string())

        #stores dataFrame in year dictionary
        statDict[year] = dataFrame

    #print(statDict)
    #print(glossDict)
    return statDict, glossDict
    

        
