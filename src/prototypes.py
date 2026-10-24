"""Several typical 10-K style sentences per category.

Written from general knowledge of risk-factor wording, NOT copied from the
reference sample, so the evaluation stays fair.
"""
PROTOTYPES = {
    "Macroeconomic & Market": [
        "Adverse economic conditions, a recession or slower growth could reduce demand and hurt our results.",
        "Changes in interest rates, inflation and financial market volatility could adversely affect our business.",
        "Market conditions such as falling asset prices, commodity prices or real estate values could cause losses.",
        "Fluctuations in residual values and pricing could reduce the value of our assets and guarantees.",
    ],
    "Credit & Liquidity": [
        "Borrowers and counterparties may fail to repay, causing credit losses on loans and collateral.",
        "Our liquidity could be constrained and funding costs could rise if we cannot access capital markets.",
        "Regulatory capital requirements and stress capital buffers could require us to hold more capital.",
        "We may be unable to make payments on our indebtedness or to refinance our debt on acceptable terms.",
        "We may need to raise additional funds, which may not be available on favorable terms.",
    ],
    "Regulatory & Legal": [
        "We are subject to extensive and changing laws and government regulation that could increase our costs.",
        "Litigation, legal proceedings and government investigations could result in fines and penalties.",
        "Changes in tax laws or adverse outcomes of tax examinations could increase our effective tax rate.",
        "Government incentives, rebates and tax credits may be reduced or eliminated.",
        "Regulators could take enforcement actions and impose sanctions for compliance failures.",
    ],
    "Cybersecurity & Data Privacy": [
        "Cyber attacks, hacking and ransomware could compromise our systems and data.",
        "Unauthorized access to personal information or a data breach could harm our reputation.",
        "Failure to comply with privacy and data protection laws could result in legal liability.",
    ],
    "Supply Chain & Manufacturing": [
        "We depend on single or limited source suppliers and outsourcing partners for components.",
        "Component shortages and higher materials prices could disrupt production and increase costs.",
        "Delays in building factories and ramping up manufacturing could harm our business.",
    ],
    "Competition & Demand": [
        "We face intense competition that could reduce our market share and put pressure on prices.",
        "Our gross margins could decline due to pricing pressure and changes in product mix.",
        "If we cannot grow sales or anticipate customer demand, our results could suffer.",
        "Competitors may develop better products or new business models and take customers.",
    ],
    "Product & Technology": [
        "Our products may contain defects that lead to recalls, warranty claims and product liability.",
        "If we fail to innovate and introduce new products and technologies, our business could suffer.",
        "New technologies such as artificial intelligence and self-driving features may not perform as expected.",
        "We may be unable to protect our intellectual property, patents and trademarks.",
    ],
    "International & Geopolitical": [
        "Fluctuations in foreign currency exchange rates could reduce our revenue and margins.",
        "Tariffs, trade restrictions and trade disputes between countries could harm our business.",
        "War, hostilities, sanctions and political instability in other countries could disrupt operations.",
    ],
    "People & Key Personnel": [
        "We depend on the services of our key executives and the loss of them could harm us.",
        "We must attract, hire and retain qualified employees in a competitive labor market.",
    ],
    "Stock & Governance": [
        "The price of our common stock may be volatile and could decline.",
        "Our executives have pledged shares, and sales of stock could affect the share price.",
        "Provisions in our charter and governance structure could affect shareholders.",
    ],
    "Operational & Business Continuity": [
        "Failures of our information technology systems, processes or vendors could disrupt operations.",
        "Natural disasters, pandemics, power outages and other events could interrupt our business.",
        "Our risk management framework and models may not be effective in identifying all risks.",
        "Launching new products and platforms creates operational risks we may fail to manage.",
    ],
    "Climate & ESG": [
        "Climate change and extreme weather could damage our facilities and our customers.",
        "Expectations about environmental, social and governance practices could affect our reputation.",
    ],
    "General / Introduction": [
        "The following risk factors contain forward-looking statements and should be read with the rest of this report.",
        "You should carefully consider the risks described below, which could materially affect our business.",
    ],
}
