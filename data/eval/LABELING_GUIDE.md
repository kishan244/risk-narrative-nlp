# Labelling guide

Every passage in `labels.csv` has exactly one label. These rules keep the labels consistent.

## The one question

> What is the **cause** of the risk this passage is mainly about?

Almost every passage ends with "…which could harm our business and results". That ending is the
consequence, not the risk. Label the cause.

## Categories

| Category | Label it when the passage is mainly about… |
|---|---|
| Macroeconomic & Market | recession, inflation, interest rates, commodity prices (oil, gas), market swings, asset values |
| Financial & Credit | borrowers not paying, liquidity and funding, the company's own debt and ratings, capital rules for banks, pensions, impairments, accounting estimates, insurance cover |
| Regulatory & Legal | laws and regulation, investigations, lawsuits, tax, antitrust, government pricing and reimbursement, government contracts |
| Cybersecurity & Data Privacy | cyber attacks, data breaches, privacy and data-protection law |
| Supply Chain & Manufacturing | suppliers, components, raw materials, factories, production ramp, logistics and delivery |
| Competition & Demand | competitors, pricing pressure, customer demand, seasonality, brand and reputation with customers, dependence on a few customers or products |
| Product & Technology | product quality and safety, recalls, warranty, R&D and clinical trials, innovation, AI, patents and other IP |
| International & Geopolitical | exchange rates, tariffs and trade rules, war, sanctions, risks of operating in other countries |
| People & Key Personnel | hiring and keeping talent, key executives, unions and labour relations |
| Stock & Governance | share price swings, shareholder structure, dividends and buybacks, anti-takeover rules |
| Operational & Business Continuity | failures of systems or processes (not attacks), disasters, pandemics, failures of service providers, weak risk management or models, employee misconduct |
| Climate & ESG | climate change, extreme weather, energy transition, ESG expectations |
| Strategy & M&A | acquisitions, divestitures, joint ventures, restructuring, big new business bets |
| General / Introduction | introductions and summaries that name no specific risk |

## Tie-break rules

1. **Two risks in one passage:** pick the one with more text. If equal, the first one.
2. **Specific beats general:**
   - privacy law → Cybersecurity & Data Privacy (not Regulatory & Legal)
   - climate regulation → Climate & ESG
   - tariffs, export controls, sanctions → International & Geopolitical
   - anything about patents or IP, including IP lawsuits → Product & Technology
   - bank capital and liquidity rules → Financial & Credit
   - tax → Regulatory & Legal
   - pandemic → Operational & Business Continuity
3. **Reputation:** label by what causes the damage. If the passage is about brand or reputation in
   general, use Competition & Demand.
4. **Lists of many risks** ("our results may vary because of A, B, C, D…"): use the category most of the items belong to. If none dominates, use Competition & Demand
   when the list is about sales and results, otherwise General / Introduction.

## Known weak spot

About one passage in four really covers two risks. A single label cannot show that, so the
second-best guess is reported too (top-2 accuracy).
