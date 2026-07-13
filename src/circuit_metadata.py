"""
Circuit metadata: static reference data not available from FastF1 directly.

"""

import pandas as pd

CIRCUIT_METADATA = [
    # circuit_name, country, circuit_type, high_speed, historical_sc_tendency (qualitative placeholder)
    ("Bahrain International Circuit", "Bahrain", "permanent", False, "medium"),
    ("Jeddah Corniche Circuit", "Saudi Arabia", "street", True, "high"),
    ("Albert Park Circuit", "Australia", "hybrid", False, "medium"),
    ("Suzuka Circuit", "Japan", "permanent", True, "medium"),
    ("Shanghai International Circuit", "China", "permanent", False, "medium"),
    ("Miami International Autodrome", "USA", "hybrid", False, "medium"),
    ("Imola Circuit", "Italy", "permanent", False, "medium"),
    ("Circuit de Monaco", "Monaco", "street", False, "high"),
    ("Circuit de Barcelona-Catalunya", "Spain", "permanent", False, "low"),
    ("Circuit Gilles Villeneuve", "Canada", "hybrid", False, "high"),
    ("Red Bull Ring", "Austria", "permanent", True, "medium"),
    ("Silverstone Circuit", "United Kingdom", "permanent", True, "medium"),
    ("Hungaroring", "Hungary", "permanent", False, "low"),
    ("Circuit de Spa-Francorchamps", "Belgium", "permanent", True, "medium"),
    ("Circuit Zandvoort", "Netherlands", "permanent", False, "medium"),
    ("Monza Circuit", "Italy", "permanent", True, "low"),
    ("Baku City Circuit", "Azerbaijan", "street", True, "high"),
    ("Marina Bay Street Circuit", "Singapore", "street", False, "high"),
    ("Circuit of the Americas", "USA", "permanent", False, "medium"),
    ("Autodromo Hermanos Rodriguez", "Mexico", "permanent", False, "medium"),
    ("Interlagos Circuit", "Brazil", "permanent", False, "medium"),
    ("Las Vegas Strip Circuit", "USA", "street", True, "medium"),
    ("Losail International Circuit", "Qatar", "permanent", True, "low"),
    ("Yas Marina Circuit", "United Arab Emirates", "permanent", False, "low"),
    ("Nurburgring", "Germany", "permanent", False, "medium"),
    ("Istanbul Park", "Turkey", "permanent", True, "low"),
    ("Portimao Circuit", "Portugal", "permanent", False, "low"),
    ("Mugello Circuit", "Italy", "permanent", True, "low"),
    ("Sochi Autodrom", "Russia", "hybrid", False, "medium"),
    ("Sepang International Circuit", "Malaysia", "permanent", False, "medium"),
    ("Fuji Speedway", "Japan", "permanent", True, "low"),
    ("Valencia Street Circuit", "Spain", "street", False, "medium"),
    ("Korean International Circuit", "South Korea", "hybrid", False, "medium"),
    ("Buddh International Circuit", "India", "permanent", False, "low"),
]

COLUMNS = [
    "circuit_name", "country", "circuit_type", "high_speed",
    "historical_sc_tendency_qualitative"
]


def get_circuit_metadata():
    """Return the circuit metadata as a pandas DataFrame."""
    return pd.DataFrame(CIRCUIT_METADATA, columns=COLUMNS)


def save_circuit_metadata(path="data/processed/circuit_metadata.csv"):
    """Save circuit metadata to disk so feature engineering can join against it."""
    df = get_circuit_metadata()
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} circuits to {path}")
    return df


if __name__ == "__main__":
    save_circuit_metadata()
    print("\nNext: once data/raw is fully pulled, calculate safety car rates")
    print("per circuit from actual 2018-2024 race data, and replace the qualitative column above with real numbers.")