# All this does is display the meaning of the different features in the ESS dataset

def display_ess_table(ess_df):

    ESS_TABLE = """
    Core variables according to ESS documentation:
    - etfruit  : frequency of fruit consumption
    - eatveg   : frequency of vegetable consumption
    - cgtsmok  : smoking behavior
    - alcfreq  : alcohol consumption frequency
    - slprl    : sleep problems
    - paccnois : exposure to noise
    - bmi      : body mass index (newly created)
    - gndr     : gender
    Additionally included:
    - health   : self-rated health
    - dosprt   : sport/physical activity
    - sclmeet  : frequency of social meetings
    - inprdsc  : perceived discrimination
    - ctrlife  : perceived control over life
    """
    print(ESS_TABLE)