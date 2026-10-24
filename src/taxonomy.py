"""Fixed risk categories, described in plain words.

Each chunk is matched to the closest description (zero-shot), so the same
yardstick is used for a bank, a phone maker and a car maker.
"""
CATEGORIES = {
    "Macroeconomic & Market": "Economic downturns, recessions, inflation, interest rate changes and financial market volatility reduce demand and hurt results.",
    "Credit & Liquidity": "Borrowers default on loans, credit losses, funding and liquidity shortfalls, capital requirements and counterparty risk.",
    "Regulatory & Legal": "Changes in laws and government regulation, regulatory investigations, compliance obligations, lawsuits, litigation and fines.",
    "Cybersecurity & Data Privacy": "Cyberattacks, hacking, data breaches, ransomware, unauthorized access to personal or confidential data and privacy laws.",
    "Supply Chain & Manufacturing": "Dependence on suppliers and outsourcing partners, component shortages, manufacturing delays and production ramp problems.",
    "Competition & Demand": "Intense competition, pricing pressure, losing market share and changing customer demand for products and services.",
    "Product & Technology": "Product defects, recalls, failure to innovate, new technology such as artificial intelligence or autonomous driving, and intellectual property.",
    "International & Geopolitical": "Foreign currency exchange rates, tariffs, trade disputes, international operations, geopolitical conflict and political instability.",
    "People & Key Personnel": "Dependence on key executives and the ability to attract, retain and motivate talented employees.",
    "Stock & Governance": "Stock price volatility, shareholders, corporate governance, debt securities, dividends and share repurchases.",
    "Operational & Business Continuity": "Failures of systems and processes, natural disasters, pandemics such as COVID-19, outages and business disruption.",
    "Climate & ESG": "Climate change, extreme weather, environmental, social and governance expectations and sustainability regulation.",
    "General / Introduction": "Introductory statements that the following risk factors contain forward-looking statements and should be read together with the rest of this annual report.",
}

# shown in the app, but left out of the risk mix
NON_RISK = {"General / Introduction"}
