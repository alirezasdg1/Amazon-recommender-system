import pandas as pd
import numpy as np
import re

link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
def link(text):
    links = re.findall(link_regex,text)
    return [lnk[0] for lnk in links]