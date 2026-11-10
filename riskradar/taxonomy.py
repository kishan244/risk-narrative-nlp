"""The risk categories. One passage gets exactly one of them."""
from __future__ import annotations

CATEGORIES: dict[str, str] = {
    "Macroeconomic & Market": "Recession, inflation, interest rates, commodity prices, financial-market swings.",
    "Financial & Credit": "Credit losses, liquidity and funding, own debt, capital rules, pensions, impairments, accounting.",
    "Regulatory & Legal": "Laws and regulation, government investigations, lawsuits, tax, government pricing policy.",
    "Cybersecurity & Data Privacy": "Cyber attacks, data breaches, privacy and data-protection rules.",
    "Supply Chain & Manufacturing": "Suppliers, components and raw materials, factories, production ramp, logistics.",
    "Competition & Demand": "Competitors, pricing pressure, customer demand, brand and reputation with customers.",
    "Product & Technology": "Product quality and safety, recalls, R&D and clinical trials, innovation, AI, patents and IP.",
    "International & Geopolitical": "Exchange rates, tariffs and trade rules, war, sanctions, country risk.",
    "People & Key Personnel": "Hiring and keeping talent, key executives, labour relations.",
    "Stock & Governance": "Share price swings, shareholders, dividends and buybacks, control of the company.",
    "Operational & Business Continuity": (
        "System and process failures, disasters, pandemics, third-party service failures, risk-management gaps."
    ),
    "Climate & ESG": "Climate change, extreme weather, the energy transition, ESG expectations.",
    "Strategy & M&A": "Acquisitions, divestitures, joint ventures, restructuring, new business bets.",
    "General / Introduction": "Introductory or summary text that names no specific risk.",
}

# shown in the app for transparency, but not counted in a company's risk mix
NON_RISK = {"General / Introduction"}
RISK_CATEGORIES = [c for c in CATEGORIES if c not in NON_RISK]
