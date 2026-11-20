"""
Global configuration, dataset constants, and cybersecurity design settings
for Network Anomaly Detection V2.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CIC_RAW_DIR = RAW_DATA_DIR / "CIC-IDS2017"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DATA_DIR = DATA_DIR / "samples"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Ensure runtime directories exist
for p in [RAW_DATA_DIR, CIC_RAW_DIR, PROCESSED_DATA_DIR, SAMPLES_DATA_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# App Info
APP_TITLE = "Network Anomaly Detection V2"
APP_SUBTITLE = "Enterprise AI-Powered Intrusion & Anomaly Detection System"
APP_VERSION = "2.2.0"
DATASET_NAME = "CIC-IDS2017 (Canadian Institute for Cybersecurity)"
DATASET_SOURCE_URL = "https://www.unb.ca/cic/datasets/ids-2017.html"

# The 8 Official CIC-IDS2017 MachineLearningCSV Files
EXPECTED_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

# Total known raw rows in the complete official dataset
TOTAL_DATASET_ROWS = 2830743

# Standard CIC-IDS2017 Label and Benign identifiers
DEFAULT_LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"

# Cleaned Attack Categories Mapping to Unified Taxonomy
ATTACK_CATEGORIES = {
    "BENIGN": "Benign Traffic",
    "DoS Hulk": "DoS / DDoS",
    "PortScan": "Reconnaissance / Probe",
    "DDoS": "DoS / DDoS",
    "DoS GoldenEye": "DoS / DDoS",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "DoS slowloris": "DoS / DDoS",
    "DoS Slowhttptest": "DoS / DDoS",
    "Bot": "Botnet",
    "Web Attack - Brute Force": "Web Attacks",
    "Web Attack - XSS": "Web Attacks",
    "Infiltration": "Infiltration / Compromise",
    "Web Attack - Sql Injection": "Web Attacks",
    "Heartbleed": "Vulnerability Exploit",
}

# Threat Category Colors (Cyber Dark Theme)
CATEGORY_COLORS = {
    "Benign Traffic": "#10b981",            # Emerald
    "DoS / DDoS": "#ef4444",                # Red
    "Reconnaissance / Probe": "#f59e0b",     # Amber
    "Brute Force": "#f97316",               # Orange
    "Web Attacks": "#8b5cf6",               # Violet
    "Botnet": "#ec4899",                    # Pink
    "Infiltration / Compromise": "#06b6d4", # Cyan
    "Vulnerability Exploit": "#e11d48",     # Rose
}

# Key Predictive Flow Features (Top Network Behavior Indicators)
KEY_PREDICTIVE_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Average Packet Size",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "Down/Up Ratio",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

# Risk Levels & Alert Thresholds
RISK_LEVELS = {
    "CRITICAL": {"color": "#ef4444", "min_score": 0.85, "badge": "CRITICAL"},
    "HIGH": {"color": "#f97316", "min_score": 0.65, "badge": "HIGH"},
    "MEDIUM": {"color": "#eab308", "min_score": 0.40, "badge": "MEDIUM"},
    "LOW": {"color": "#10b981", "min_score": 0.00, "badge": "NORMAL"},
}

# Standard Port to Service Mapping for CIC-IDS2017
PORT_SERVICE_MAP = {
    80: "HTTP (80)",
    443: "HTTPS (443)",
    53: "DNS (53)",
    22: "SSH (22)",
    21: "FTP (21)",
    20: "FTP-Data (20)",
    25: "SMTP (25)",
    110: "POP3 (110)",
    123: "NTP (123)",
    137: "NetBIOS-NS (137)",
    138: "NetBIOS-DGM (138)",
    139: "NetBIOS-SSN (139)",
    143: "IMAP (143)",
    161: "SNMP (161)",
    389: "LDAP (389)",
    445: "SMB (445)",
    465: "SMTPS (465)",
    993: "IMAPS (993)",
    995: "POP3S (995)",
    1433: "MSSQL (1433)",
    3306: "MySQL (3306)",
    3389: "RDP (3389)",
    8080: "HTTP-Proxy (8080)",
    8443: "HTTPS-Alt (8443)",
    3268: "Global-Catalog (3268)",
    0: "Reserved (0)",
}

def resolve_port_service(port: object) -> str:
    """Infer application service name and port based on standard IANA destination port conventions."""
    try:
        p = int(port)
        return PORT_SERVICE_MAP.get(p, f"Port {p}")
    except (ValueError, TypeError):
        return str(port)

def resolve_protocol_group(port: object) -> str:
    """Infer high-level application/service category based on destination port conventions."""
    try:
        p = int(port)
        if p in (80, 8080):
            return "HTTP / Web"
        elif p in (443, 8443):
            return "HTTPS / TLS"
        elif p == 53:
            return "DNS"
        elif p in (22, 23, 3389):
            return "Remote Admin (SSH/RDP)"
        elif p in (20, 21):
            return "FTP Transfer"
        elif p in (137, 138, 139, 445):
            return "NetBIOS / SMB"
        elif p in (25, 110, 143, 465, 993, 995):
            return "Mail (SMTP/IMAP)"
        elif p in (88, 389, 3268):
            return "Auth / LDAP / Kerberos"
        elif p in (123, 161):
            return "Time / Monitoring (NTP/SNMP)"
        elif p in (1433, 3306):
            return "Database (SQL)"
        elif p > 1024:
            return "Ephemeral / High Ports"
        else:
            return "Other Standard Ports"
    except (ValueError, TypeError):
        return "Unknown"

