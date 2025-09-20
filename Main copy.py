import gspread
from google.oauth2.service_account import Credentials
import time
from Prompt_bot_GROK import run_bot_new
from Prompt_bot_GROK import find_volatility_grok


# 1️⃣  Authorise with your service-account key
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
creds   = Credentials.from_service_account_file("Credentials.json", scopes=SCOPES)
gc      = gspread.authorize(creds)
# 2️⃣  Open the spreadsheet (by ID or by title)
SHEET_ID   = "hidden"

ws = gc.open_by_key(SHEET_ID).sheet1  
symbols = [s.strip() for s in ws.col_values(1) if s.strip()]
cells = ws.get("A:A")          # returns list of lists
symbols = [row[0].strip() for row in cells if row and row[0].strip()]
symbols = [row[0].strip() for row in ws.get_all_values() if row and row[0].strip()]



for sym in symbols:
    # risk overrides volatility_threshold, so 0.55 becomes the cut-off
    decision, score = find_volatility_grok(sym, risk=0.4, verbose=True)

    # Always pass both pieces of information to run_bot;
    # inside run_bot a NO_TRADE (Volatility) decision skips pattern detection. 
    run_bot_new(sym,score)

    if decision.startswith("NO_TRADE"):
        print(decision, score)
    
print(symbols)



