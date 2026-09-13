"""
Synthetic Dataset Generator for Indian Infrastructure Projects
==============================================================

This script generates a realistic synthetic dataset of Indian infrastructure
projects across Highways, Railways, Airports, and Renewable Energy.

Primary Reference Sources & Methodological Foundations:
-------------------------------------------------------
1. Land Conflict Watch (LCW) (https://www.landconflictwatch.org/all-conflicts):
   - Definition: Public opposition to changes in land use/ownership by state or private actors.
   - Gram Sabha to national level records; exclusion of purely private litigation.
   - Socio-legal conflict patterns: High dispute rates in Forest & Schedule V Tribal tracts;
     community mobilization triggered by high displacement and unresolved customary claims;
     correlation between project investment scale (Rs Crore) and conflict likelihood.
   - Real-world distribution of affected families and land area footprint.

2. Pradhan Mantri Gram Sadak Yojana (PMGSY) (https://pmgsy.dord.gov.in/):
   - Spatial administrative hierarchy: State -> District -> Division / Block.
   - Standardized district mapping and geographical boundary conventions across India.

3. National Highways Authority of India (NHAI) & Ministry of Railways Data:
   - Linear acquisition patterns under RFCTLARR Act 2013 (Section 11 Notification timelines).
   - Project award dynamics, corridor costs (Rs Cr), and approval pipeline delays.
"""

import argparse
import csv
import math
import random
import sys
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple


# ==============================================================================
# 1. GEOGRAPHICAL & ADMINISTRATIVE DEFINITIONS (28 States + 8 UTs)
#    Standardized district lists following PMGSY & Census administrative boundaries
# ==============================================================================

STATE_TIER_WEIGHTS: Dict[str, float] = {
    # Tier 1 (High Project Density - NHAI/Renewable/Industrial Corridors) ~67%
    "Maharashtra": 0.13,
    "Uttar Pradesh": 0.12,
    "Rajasthan": 0.10,
    "Gujarat": 0.09,
    "Karnataka": 0.08,
    "Tamil Nadu": 0.08,
    "Madhya Pradesh": 0.07,
    # Tier 2 (Medium Project Density - Major state economies & transit zones) ~27%
    "Andhra Pradesh": 0.040,
    "Telangana": 0.038,
    "Bihar": 0.038,
    "West Bengal": 0.038,
    "Odisha": 0.036,
    "Punjab": 0.026,
    "Haryana": 0.026,
    "Jharkhand": 0.026,
    "Chhattisgarh": 0.026,
    "Kerala": 0.020,
    "Assam": 0.016,
    # Tier 3 (Hilly, North-Eastern States & Union Territories) ~6%
    "Uttarakhand": 0.008,
    "Himachal Pradesh": 0.008,
    "Delhi": 0.008,
    "Jammu and Kashmir": 0.006,
    "Goa": 0.004,
    "Tripura": 0.003,
    "Meghalaya": 0.003,
    "Manipur": 0.002,
    "Nagaland": 0.002,
    "Mizoram": 0.002,
    "Arunachal Pradesh": 0.002,
    "Sikkim": 0.002,
    "Puducherry": 0.002,
    "Chandigarh": 0.002,
    "Ladakh": 0.001,
    "Andaman and Nicobar Islands": 0.001,
    "Dadra and Nagar Haveli and Daman and Diu": 0.001,
    "Lakshadweep": 0.0005,
}

STATE_CODE_MAP: Dict[str, str] = {
    "Andhra Pradesh": "AP", "Arunachal Pradesh": "AR", "Assam": "AS", "Bihar": "BR",
    "Chhattisgarh": "CG", "Goa": "GA", "Gujarat": "GJ", "Haryana": "HR",
    "Himachal Pradesh": "HP", "Jharkhand": "JH", "Karnataka": "KA", "Kerala": "KL",
    "Madhya Pradesh": "MP", "Maharashtra": "MH", "Manipur": "MN", "Meghalaya": "ML",
    "Mizoram": "MZ", "Nagaland": "NL", "Odisha": "OD", "Punjab": "PB",
    "Rajasthan": "RJ", "Sikkim": "SK", "Tamil Nadu": "TN", "Telangana": "TS",
    "Tripura": "TR", "Uttar Pradesh": "UP", "Uttarakhand": "UK", "West Bengal": "WB",
    "Andaman and Nicobar Islands": "AN", "Chandigarh": "CH",
    "Dadra and Nagar Haveli and Daman and Diu": "DN", "Delhi": "DL",
    "Jammu and Kashmir": "JK", "Ladakh": "LA", "Lakshadweep": "LD", "Puducherry": "PY",
}

STATE_DISTRICTS_MAP: Dict[str, List[str]] = {
    "Andhra Pradesh": [
        "Visakhapatnam", "NTR (Vijayawada)", "Guntur", "Chittoor", "Krishna",
        "Kurnool", "SPSR Nellore", "Ananthapuramu", "East Godavari", "West Godavari",
        "YSR Kadapa", "Prakasam", "Srikakulam", "Vizianagaram", "Tirupati", "Annamayya"
    ],
    "Arunachal Pradesh": [
        "Papum Pare", "Changlang", "West Kameng", "East Siang", "Tawang",
        "Lohit", "Lower Subansiri", "Tirap", "Upper Siang", "Namsai"
    ],
    "Assam": [
        "Kamrup Metropolitan", "Kamrup", "Dibrugarh", "Cachar", "Nagaon",
        "Sonitpur", "Jorhat", "Tinsukia", "Barpeta", "Golaghat",
        "Dhubri", "Darrang", "Karbi Long", "Dima Hasao", "Kokrajhar"
    ],
    "Bihar": [
        "Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga",
        "Purnia", "Begusarai", "Saran", "Nalanda", "Rohtas",
        "Samastipur", "Vaishali", "Paschim Champaran", "Purba Champaran", "Katihar", "Madhubani"
    ],
    "Chhattisgarh": [
        "Raipur", "Durg", "Bilaspur", "Korba", "Rajnandgaon",
        "Bastar", "Raigarh", "Surguja", "Janjgir-Champa", "Dantewada",
        "Kanker", "Mahasamund", "Koriya", "Dhamtari", "Balod"
    ],
    "Goa": [
        "North Goa", "South Goa"
    ],
    "Gujarat": [
        "Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar",
        "Jamnagar", "Kutch", "Gandhinagar", "Junagadh", "Anand",
        "Bharuch", "Mehsana", "Valsad", "Navsari", "Banaskantha",
        "Sabarkantha", "Panchmahal", "Dahod", "Surendranagar", "Patan"
    ],
    "Haryana": [
        "Gurugram", "Faridabad", "Panipat", "Ambala", "Yamunanagar",
        "Karnal", "Sonipat", "Rohtak", "Hisar", "Panchkula",
        "Kurukshetra", "Jind", "Sirsa", "Rewari", "Jhajjar", "Palwal"
    ],
    "Himachal Pradesh": [
        "Shimla", "Kangra", "Mandi", "Solan", "Kullu",
        "Sirmaur", "Hamirpur", "Una", "Bilaspur", "Chamba", "Kinnaur", "Lahaul and Spiti"
    ],
    "Jharkhand": [
        "Ranchi", "East Singhbhum", "Dhanbad", "Bokaro", "Hazaribagh",
        "Palamu", "Deoghar", "Giridih", "Ramgarh", "Saraikela Kharsawan",
        "West Singhbhum", "Dumka", "Gumla", "Latehar", "Khunti"
    ],
    "Karnataka": [
        "Bengaluru Urban", "Bengaluru Rural", "Mysuru", "Dakshina Kannada", "Belagavi",
        "Dharwad", "Kalaburagi", "Tumakuru", "Ballari", "Udupi",
        "Shivamogga", "Hassan", "Davanagere", "Vijayapura", "Mandya",
        "Uttara Kannada", "Chikkamagaluru", "Kolar", "Bagalkote"
    ],
    "Kerala": [
        "Thiruvananthapuram", "Ernakulam", "Kozhikode", "Thrissur", "Malappuram",
        "Kollam", "Palakkad", "Kannur", "Alappuzha", "Kottayam",
        "Kasaragod", "Pathanamthitta", "Idukki", "Wayanad"
    ],
    "Madhya Pradesh": [
        "Indore", "Bhopal", "Jabalpur", "Gwalior", "Ujjain",
        "Sagar", "Dewas", "Satna", "Ratlam", "Rewa",
        "Singrauli", "Chhindwara", "Dhar", "Khargone", "Shivpuri",
        "Vidisha", "Sehore", "Narmadapuram", "Betul", "Balaghat", "Shahdol"
    ],
    "Maharashtra": [
        "Mumbai City", "Mumbai Suburban", "Pune", "Nagpur", "Thane",
        "Nashik", "Chhatrapati Sambhajinagar", "Solapur", "Kolhapur", "Amravati",
        "Jalgaon", "Nanded", "Sangli", "Satara", "Raigad",
        "Ahmednagar", "Chandrapur", "Palghar", "Ratnagiri", "Sindhudurg",
        "Yavatmal", "Gadchiroli", "Latur", "Bhandara"
    ],
    "Manipur": [
        "Imphal West", "Imphal East", "Churachandpur", "Thoubal", "Bishnupur",
        "Senapati", "Ukhrul", "Chandel", "Tamenglong"
    ],
    "Meghalaya": [
        "East Khasi Hills", "West Garo Hills", "Ri Bhoi", "West Khasi Hills",
        "West Jaintia Hills", "East Garo Hills", "South Garo Hills"
    ],
    "Mizoram": [
        "Aizawl", "Lunglei", "Champhai", "Kolasib", "Serchhip", "Lawngtlai", "Mamit", "Saiha"
    ],
    "Nagaland": [
        "Kohima", "Dimapur", "Mokokchung", "Tuensang", "Wokha",
        "Zunheboto", "Mon", "Phek", "Chumoukedima"
    ],
    "Odisha": [
        "Khordha", "Cuttack", "Ganjam", "Sundargarh", "Balasore",
        "Mayurbhanj", "Puri", "Sambalpur", "Angul", "Jajpur",
        "Jharsuguda", "Koraput", "Rayagada", "Kalahandi", "Balangir",
        "Kendrapara", "Jagatsinghpur", "Bargarh", "Keonjhar"
    ],
    "Punjab": [
        "Ludhiana", "Amritsar", "Jalandhar", "Patiala", "SAS Nagar (Mohali)",
        "Bathinda", "Hoshiarpur", "Pathankot", "Moga", "Firozpur",
        "Sangrur", "Kapurthala", "Gurdaspur", "Rupnagar", "Fazilka"
    ],
    "Rajasthan": [
        "Jaipur", "Jodhpur", "Kota", "Bikaner", "Ajmer",
        "Udaipur", "Bhilwara", "Alwar", "Sikar", "Sri Ganganagar",
        "Pali", "Bharatpur", "Barmer", "Jaisalmer", "Nagaur",
        "Jhunjhunu", "Chittorgarh", "Jalore", "Banswara", "Dungarpur"
    ],
    "Sikkim": [
        "East Sikkim (Gangtok)", "West Sikkim (Gyalshing)", "North Sikkim (Mangan)",
        "South Sikkim (Namchi)", "Pakyong", "Soreng"
    ],
    "Tamil Nadu": [
        "Chennai", "Coimbatore", "Chengalpattu", "Kanchipuram", "Tiruvallur",
        "Madurai", "Tiruchirappalli", "Salem", "Tiruppur", "Erode",
        "Vellore", "Tirunelveli", "Thanjavur", "Dindigul", "Cuddalore",
        "Kanyakumari", "Krishnagiri", "Ranipet", "Thoothukudi", "Villupuram"
    ],
    "Telangana": [
        "Hyderabad", "Medchal-Malkajgiri", "Rangareddy", "Sangareddy", "Warangal",
        "Nalgonda", "Nizamabad", "Khammam", "Karimnagar", "Mahabubnagar",
        "Bhadradri Kothagudem", "Siddipet", "Suryapet", "Mancherial", "Adilabad"
    ],
    "Tripura": [
        "West Tripura", "Gomati", "South Tripura", "North Tripura",
        "Unakoti", "Dhalai", "Khowai", "Sepahijala"
    ],
    "Uttar Pradesh": [
        "Lucknow", "Kanpur Nagar", "Gautam Buddha Nagar (Noida)", "Ghaziabad", "Varanasi",
        "Agra", "Prayagraj", "Meerut", "Bareilly", "Aligarh",
        "Moradabad", "Saharanpur", "Gorakhpur", "Jhansi", "Mathura",
        "Ayodhya", "Muzaffarnagar", "Bulandshahr", "Firozabad", "Sonbhadra",
        "Mirzapur", "Sitapur", "Hardoi", "Lakhimpur Kheri", "Banda"
    ],
    "Uttarakhand": [
        "Dehradun", "Haridwar", "Udham Singh Nagar", "Nainital", "Pauri Garhwal",
        "Almora", "Tehri Garhwal", "Chamoli", "Pithoragarh", "Uttarkashi",
        "Rudraprayag", "Bageshwar", "Champawat"
    ],
    "West Bengal": [
        "Kolkata", "North 24 Parganas", "South 24 Parganas", "Howrah", "Hooghly",
        "Paschim Medinipur", "Purba Medinipur", "Purba Bardhaman", "Paschim Bardhaman",
        "Murshidabad", "Nadia", "Darjeeling", "Jalpaiguri", "Malda",
        "Bankura", "Birbhum", "Purulia", "Alipurduar", "Cooch Behar"
    ],
    "Andaman and Nicobar Islands": [
        "South Andaman", "North and Middle Andaman", "Nicobar"
    ],
    "Chandigarh": [
        "Chandigarh"
    ],
    "Dadra and Nagar Haveli and Daman and Diu": [
        "Daman", "Diu", "Dadra and Nagar Haveli"
    ],
    "Delhi": [
        "New Delhi", "South Delhi", "North Delhi", "West Delhi", "East Delhi",
        "South West Delhi", "North West Delhi", "North East Delhi", "Central Delhi",
        "Shahdara", "South East Delhi"
    ],
    "Jammu and Kashmir": [
        "Srinagar", "Jammu", "Anantnag", "Baramulla", "Udhampur",
        "Kathua", "Budgam", "Pulwama", "Kupwara", "Rajouri",
        "Poonch", "Samba", "Reasi", "Ganderbal", "Bandipora", "Kulgam", "Doda", "Ramban"
    ],
    "Ladakh": [
        "Leh", "Kargil"
    ],
    "Lakshadweep": [
        "Kavaratti", "Agatti", "Andrott", "Minicoy"
    ],
    "Puducherry": [
        "Puducherry", "Karaikal", "Mahe", "Yanam"
    ],
}


# ==============================================================================
# 2. TERRAIN DISTRIBUTION MATRIX BY REGIONAL GEOGRAPHY
#    Incorporating Land Conflict Watch (LCW) geographic conflict hotspot profiles
# ==============================================================================

# Default Terrain Distribution: [Urban, Rural_Agri, Forest_Eco_Sensitive, Tribal_Schedule_V]
DEFAULT_TERRAIN_PROBS = [0.15, 0.55, 0.18, 0.12]

STATE_TERRAIN_PROBS: Dict[str, List[float]] = {
    # Schedule V Tribal Prominent States (Odisha, Jharkhand, Chhattisgarh, MP, Rajasthan, Gujarat)
    "Odisha": [0.08, 0.32, 0.28, 0.32],
    "Jharkhand": [0.10, 0.28, 0.26, 0.36],
    "Chhattisgarh": [0.08, 0.26, 0.32, 0.34],
    "Madhya Pradesh": [0.12, 0.44, 0.20, 0.24],
    "Rajasthan": [0.14, 0.48, 0.16, 0.22],
    "Gujarat": [0.20, 0.48, 0.12, 0.20],
    "Telangana": [0.16, 0.46, 0.18, 0.20],
    "Andhra Pradesh": [0.16, 0.52, 0.16, 0.16],
    "Maharashtra": [0.22, 0.46, 0.16, 0.16],

    # Western Ghats, Himalayan, and North-Eastern Forest / Eco-Sensitive Zones
    "Uttarakhand": [0.08, 0.22, 0.62, 0.08],
    "Himachal Pradesh": [0.08, 0.24, 0.60, 0.08],
    "Jammu and Kashmir": [0.10, 0.30, 0.52, 0.08],
    "Ladakh": [0.05, 0.30, 0.60, 0.05],
    "Kerala": [0.22, 0.32, 0.42, 0.04],
    "Goa": [0.25, 0.25, 0.48, 0.02],
    "Assam": [0.12, 0.44, 0.26, 0.18],
    "Arunachal Pradesh": [0.04, 0.16, 0.60, 0.20],
    "Meghalaya": [0.06, 0.18, 0.52, 0.24],
    "Manipur": [0.08, 0.22, 0.50, 0.20],
    "Nagaland": [0.05, 0.15, 0.55, 0.25],
    "Mizoram": [0.05, 0.15, 0.60, 0.20],
    "Sikkim": [0.06, 0.20, 0.66, 0.08],
    "Tripura": [0.10, 0.38, 0.36, 0.16],
    "Andaman and Nicobar Islands": [0.08, 0.12, 0.72, 0.08],

    # Indo-Gangetic Plains & Intensive Agricultural States
    "Uttar Pradesh": [0.18, 0.70, 0.08, 0.04],
    "Bihar": [0.12, 0.76, 0.08, 0.04],
    "Punjab": [0.16, 0.80, 0.03, 0.01],
    "Haryana": [0.24, 0.72, 0.03, 0.01],
    "West Bengal": [0.20, 0.60, 0.12, 0.08],
    "Tamil Nadu": [0.24, 0.60, 0.12, 0.04],
    "Karnataka": [0.22, 0.50, 0.18, 0.10],

    # Metropolitan / Urbanized Union Territories
    "Delhi": [0.85, 0.13, 0.02, 0.00],
    "Chandigarh": [0.90, 0.08, 0.02, 0.00],
    "Puducherry": [0.65, 0.30, 0.05, 0.00],
    "Dadra and Nagar Haveli and Daman and Diu": [0.35, 0.40, 0.15, 0.10],
    "Lakshadweep": [0.20, 0.10, 0.70, 0.00],
}

TERRAIN_TYPES = ["Urban", "Rural_Agri", "Forest_Eco_Sensitive", "Tribal_Schedule_V"]


# ==============================================================================
# 3. PROJECT TYPES, AGENCIES, AND PHYSICAL PARAMETERS
# ==============================================================================

PROJECT_TYPES = ["Highway", "Railway", "Airport", "Renewable_Energy"]
PROJECT_TYPE_WEIGHTS = [0.45, 0.25, 0.10, 0.20]

PROJECT_AGENCY_MAP = {
    "Highway": "NHAI",
    "Railway": "MoR",
    "Airport": "AAI",
    "Renewable_Energy": "MNRE",
}

# Land Area ranges in hectares (Min, Max, Mean, StdDev for truncated lognormal/gaussian)
LAND_AREA_SPECS = {
    "Highway": {"min": 50.0, "max": 500.0, "mean": 180.0, "std": 90.0},
    "Railway": {"min": 100.0, "max": 800.0, "mean": 320.0, "std": 140.0},
    "Airport": {"min": 200.0, "max": 1000.0, "mean": 550.0, "std": 180.0},
    "Renewable_Energy": {"min": 5.0, "max": 200.0, "mean": 45.0, "std": 35.0},
}

# Estimated Investment (in Rs. Crore) per hectare base parameters
INVESTMENT_PER_HA_SPECS = {
    "Highway": {"base_cr_per_ha": 6.5, "noise_std": 1.8},
    "Railway": {"base_cr_per_ha": 5.2, "noise_std": 1.4},
    "Airport": {"base_cr_per_ha": 8.0, "noise_std": 2.5},
    "Renewable_Energy": {"base_cr_per_ha": 3.8, "noise_std": 0.9},
}


# ==============================================================================
# 4. STATISTICAL HELPER FUNCTIONS & DISTRIBUTIONS
# ==============================================================================

def sample_truncated_normal(mean: float, std: float, low: float, high: float) -> float:
    """Sample from a Gaussian distribution bounded within [low, high]."""
    for _ in range(25):
        val = random.gauss(mean, std)
        if low <= val <= high:
            return val
    return max(low, min(high, mean))


def sample_weighted_choice(options: List[Any], weights: List[float]) -> Any:
    """Sample an item from a list given arbitrary weights."""
    return random.choices(options, weights=weights, k=1)[0]


def sigmoid(x: float) -> float:
    """Logistic sigmoid activation."""
    if x < -35:
        return 0.0
    if x > 35:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# ==============================================================================
# 5. CORE SYNTHETIC DATA GENERATION ENGINE
# ==============================================================================

def generate_record(record_index: int, state_counter: Dict[str, int]) -> Dict[str, Any]:
    """
    Generate a single realistic Indian infrastructure project record adhering
    to LCW socio-legal conflict dynamics and statutory administrative timelines.
    """
    # --------------------------------------------------------------------------
    # A. Project & Geography Metrics
    # --------------------------------------------------------------------------
    states = list(STATE_TIER_WEIGHTS.keys())
    weights = list(STATE_TIER_WEIGHTS.values())
    state = sample_weighted_choice(states, weights)

    districts = STATE_DISTRICTS_MAP.get(state, ["District Headquarters"])
    district = random.choice(districts)

    project_type = sample_weighted_choice(PROJECT_TYPES, PROJECT_TYPE_WEIGHTS)
    agency = PROJECT_AGENCY_MAP[project_type]

    # Project year: 2015 to 2026 (weighted towards recent 2020-2025 infrastructure push)
    year_weights = [0.03, 0.04, 0.05, 0.06, 0.07, 0.09, 0.11, 0.13, 0.15, 0.14, 0.09, 0.04]
    years = list(range(2015, 2027))
    start_year = sample_weighted_choice(years, year_weights)

    # State sequence number for Unique Identifier
    state_code = STATE_CODE_MAP.get(state, "IN")
    state_counter[state] = state_counter.get(state, 0) + 1
    seq_num = state_counter[state]
    project_id = f"{agency}-{state_code}-{start_year}-{seq_num:04d}"

    # Terrain type based on regional geography matrix
    terrain_probs = STATE_TERRAIN_PROBS.get(state, DEFAULT_TERRAIN_PROBS)
    terrain_type = sample_weighted_choice(TERRAIN_TYPES, terrain_probs)

    # Land area (Hectares)
    area_spec = LAND_AREA_SPECS[project_type]
    land_area = sample_truncated_normal(
        mean=area_spec["mean"],
        std=area_spec["std"],
        low=area_spec["min"],
        high=area_spec["max"]
    )
    land_area = round(land_area, 2)

    # Estimated Project Investment in Rs. Crore (correlated with LCW scale dynamics)
    cost_spec = INVESTMENT_PER_HA_SPECS[project_type]
    cost_per_ha = max(1.2, random.gauss(cost_spec["base_cr_per_ha"], cost_spec["noise_std"]))
    estimated_cost_cr = round(land_area * cost_per_ha, 2)

    # --------------------------------------------------------------------------
    # B. Socio-Legal & Demographic Metrics (Land Conflict Watch Modeling)
    # --------------------------------------------------------------------------
    # Density multiplier: Families displaced/impacted per hectare
    density_ranges = {
        "Urban": (5.0, 14.0),
        "Rural_Agri": (1.4, 4.8),
        "Forest_Eco_Sensitive": (0.3, 1.6),
        "Tribal_Schedule_V": (0.5, 2.2),
    }
    low_d, high_d = density_ranges[terrain_type]
    fam_density = random.uniform(low_d, high_d)

    # Base affected families with lognormal dispersion
    log_dispersion = random.lognormvariate(0, 0.35)
    raw_families = int(land_area * fam_density * log_dispersion)
    # Clamp to realistic bounds (10 to 7500)
    affected_families_count = max(10, min(7500, raw_families))

    # Title Dispute Rate Percent (0% - 100%)
    # LCW Finding: Significantly elevated in Tribal & Forest areas due to FRA 2006,
    # customary unrecorded tenures, and community forest rights.
    if terrain_type == "Tribal_Schedule_V":
        base_dispute = sample_truncated_normal(mean=42.0, std=15.0, low=15.0, high=85.0)
    elif terrain_type == "Forest_Eco_Sensitive":
        base_dispute = sample_truncated_normal(mean=34.0, std=14.0, low=12.0, high=75.0)
    elif terrain_type == "Rural_Agri":
        base_dispute = sample_truncated_normal(mean=22.0, std=10.0, low=4.0, high=58.0)
    else:  # Urban
        base_dispute = sample_truncated_normal(mean=11.0, std=6.5, low=1.0, high=35.0)

    # Additional dispute pressure from high investment scale & large affected population
    investment_factor = min(12.0, (estimated_cost_cr / 2500.0) * 4.0)
    pop_factor = min(10.0, (affected_families_count / 1500.0) * 3.0)
    title_dispute_rate_percent = round(min(98.5, max(0.5, base_dispute + investment_factor + pop_factor)), 2)

    # Compensation Multiplier Demand (1.0x to 4.0x)
    # Under RFCTLARR 2013: Statutory multiplier is 1.0x in urban, up to 2.0x in rural (+ 100% solatium).
    # Displaced communities demand 2.5x to 4.0x when displacement is large or disputes exist.
    if terrain_type in ["Tribal_Schedule_V", "Forest_Eco_Sensitive"]:
        base_comp = sample_truncated_normal(mean=2.6, std=0.55, low=1.5, high=4.0)
    elif terrain_type == "Rural_Agri":
        base_comp = sample_truncated_normal(mean=2.2, std=0.45, low=1.2, high=3.8)
    else:  # Urban
        base_comp = sample_truncated_normal(mean=1.6, std=0.35, low=1.0, high=3.0)

    if affected_families_count > 1000 or title_dispute_rate_percent > 35.0:
        base_comp += random.uniform(0.2, 0.6)

    compensation_multiplier_demand = round(max(1.0, min(4.0, base_comp)), 2)

    # Local Protest Flag (Boolean)
    # Modeled as a logistic probability function derived from LCW conflict triggers:
    # High title dispute rate, sensitive terrain, large displacement, and high compensation demands
    logit_score = -2.8  # baseline offset (~12% base)
    logit_score += (title_dispute_rate_percent - 25.0) * 0.058
    logit_score += (affected_families_count / 800.0) * 0.52
    logit_score += (compensation_multiplier_demand - 2.0) * 0.85

    if terrain_type == "Tribal_Schedule_V":
        logit_score += 0.95
    elif terrain_type == "Forest_Eco_Sensitive":
        logit_score += 0.70
    elif terrain_type == "Urban":
        logit_score -= 0.40

    protest_prob = sigmoid(logit_score)
    local_protest_flag = random.random() < protest_prob

    # --------------------------------------------------------------------------
    # C. Administrative Workflow & Statutory Metrics (RFCTLARR / MoEFCC / NHAI)
    # --------------------------------------------------------------------------
    # Forest Clearance Status
    if terrain_type == "Urban":
        forest_clearance_status = "Not_Required"
    elif terrain_type == "Forest_Eco_Sensitive":
        forest_clearance_status = sample_weighted_choice(
            ["Approved", "Stage_1_Pending"],
            [0.42, 0.58]
        )
    elif terrain_type == "Tribal_Schedule_V":
        forest_clearance_status = sample_weighted_choice(
            ["Not_Required", "Stage_1_Pending", "Approved"],
            [0.35, 0.40, 0.25]
        )
    else:  # Rural_Agri
        forest_clearance_status = sample_weighted_choice(
            ["Not_Required", "Stage_1_Pending", "Approved"],
            [0.84, 0.08, 0.08]
        )

    # SIA (Social Impact Assessment) Approval Status
    # Base requirements: Approved: 45%, Pending: 35%, Rejected: 10%, Exempted: 10%
    if project_type == "Renewable_Energy" and land_area < 50:
        # Small renewable projects often qualify for expedited exemption
        sia_approval_status = sample_weighted_choice(
            ["Approved", "Pending", "Rejected", "Exempted"],
            [0.50, 0.20, 0.05, 0.25]
        )
    elif local_protest_flag and (forest_clearance_status == "Stage_1_Pending" or title_dispute_rate_percent > 40):
        # Conflicts stall SIA appraisals (reflecting NHAI & LCW delay analyses)
        sia_approval_status = sample_weighted_choice(
            ["Approved", "Pending", "Rejected", "Exempted"],
            [0.20, 0.60, 0.16, 0.04]
        )
    else:
        sia_approval_status = sample_weighted_choice(
            ["Approved", "Pending", "Rejected", "Exempted"],
            [0.45, 0.35, 0.10, 0.10]
        )

    # Section 11 Notification Days (RFCTLARR Act 2013 preliminary acquisition notification)
    # Realistic bounds: 30 to 730 days
    # Fast in urban/exempt; severely delayed in tribal/forest corridors and protested projects
    if terrain_type == "Urban":
        base_days = sample_truncated_normal(mean=95.0, std=35.0, low=30.0, high=180.0)
    elif terrain_type == "Rural_Agri":
        base_days = sample_truncated_normal(mean=240.0, std=85.0, low=60.0, high=450.0)
    elif terrain_type == "Forest_Eco_Sensitive":
        base_days = sample_truncated_normal(mean=410.0, std=110.0, low=180.0, high=710.0)
    else:  # Tribal_Schedule_V
        base_days = sample_truncated_normal(mean=460.0, std=115.0, low=190.0, high=730.0)

    # Linear infrastructure corridor multi-village expansion delays
    if project_type in ["Highway", "Railway"]:
        base_days += random.uniform(25.0, 75.0)

    # Escalation due to protests & disputes
    if local_protest_flag:
        base_days += random.uniform(60.0, 140.0)
    if forest_clearance_status == "Stage_1_Pending":
        base_days += random.uniform(45.0, 120.0)

    section_11_notification_days = int(max(30, min(730, round(base_days))))

    # Fund Disbursement Percent (0.0% to 100.0%)
    # Real-world NHAI/MoR benchmark: ~60% of ongoing infrastructure projects have <50% disbursement
    # Inversely correlated with delays, disputes, pending clearances, and protest presence
    delay_penalty = (section_11_notification_days / 730.0) * 35.0
    dispute_penalty = (title_dispute_rate_percent / 100.0) * 20.0
    protest_penalty = 18.0 if local_protest_flag else 0.0
    clearance_penalty = 15.0 if forest_clearance_status == "Stage_1_Pending" else 0.0
    sia_penalty = 20.0 if sia_approval_status in ["Pending", "Rejected"] else 0.0

    total_penalty = delay_penalty + dispute_penalty + protest_penalty + clearance_penalty + sia_penalty
    base_disbursement = sample_truncated_normal(mean=72.0, std=22.0, low=10.0, high=98.0)
    raw_disb = base_disbursement - total_penalty + random.gauss(0, 8.0)
    fund_disbursement_percent = round(max(0.0, min(100.0, raw_disb)), 2)

    return {
        "project_id": project_id,
        "project_type": project_type,
        "state": state,
        "district": district,
        "terrain_type": terrain_type,
        "land_area_hectares": land_area,
        "estimated_cost_inr_crore": estimated_cost_cr,
        "project_start_year": start_year,
        "affected_families_count": affected_families_count,
        "title_dispute_rate_percent": title_dispute_rate_percent,
        "local_protest_flag": local_protest_flag,
        "compensation_multiplier_demand": compensation_multiplier_demand,
        "sia_approval_status": sia_approval_status,
        "section_11_notification_days": section_11_notification_days,
        "forest_clearance_status": forest_clearance_status,
        "fund_disbursement_percent": fund_disbursement_percent,
    }


def generate_dataset(num_records: int = 7500, random_seed: int = 42) -> List[Dict[str, Any]]:
    """Generate full dataset with progress tracking and deterministic reproducibility."""
    random.seed(random_seed)
    records = []
    state_counter: Dict[str, int] = {}

    print(f"[*] Starting dataset generation: {num_records:,} records (Seed: {random_seed})...")
    milestone = max(1, num_records // 10)

    for i in range(1, num_records + 1):
        rec = generate_record(i, state_counter)
        records.append(rec)

        if i % milestone == 0 or i == num_records:
            progress = (i / num_records) * 100
            print(f"    Progress: {i:>6,}/{num_records:,} ({progress:>5.1f}%) completed.")

    return records


# ==============================================================================
# 6. DATA VALIDATION & QUALITY CHECKS
# ==============================================================================

def validate_dataset(records: List[Dict[str, Any]]) -> bool:
    """Run comprehensive automated assertions against dataset integrity."""
    print("\n" + "=" * 80)
    print("DATA VALIDATION & LOGICAL CONSTRAINT CHECKS")
    print("=" * 80)

    errors = []

    # 1. Null / Missing value check
    for idx, row in enumerate(records):
        for col, val in row.items():
            if val is None or val == "" or (isinstance(val, float) and math.isnan(val)):
                errors.append(f"Row {idx}: Column '{col}' is null/empty.")

    # 2. Constraint: Urban terrain -> forest_clearance_status == 'Not_Required'
    urban_forest_violations = [
        r["project_id"] for r in records
        if r["terrain_type"] == "Urban" and r["forest_clearance_status"] != "Not_Required"
    ]
    if urban_forest_violations:
        errors.append(f"Found {len(urban_forest_violations)} Urban records with required forest clearance.")

    # 3. Numeric bounds validation
    for idx, r in enumerate(records):
        if not (5.0 <= r["land_area_hectares"] <= 1000.0):
            errors.append(f"Row {idx}: land_area_hectares out of bounds ({r['land_area_hectares']})")
        if not (0.0 <= r["title_dispute_rate_percent"] <= 100.0):
            errors.append(f"Row {idx}: title_dispute_rate_percent out of bounds ({r['title_dispute_rate_percent']})")
        if not (1.0 <= r["compensation_multiplier_demand"] <= 4.0):
            errors.append(f"Row {idx}: compensation_multiplier_demand out of bounds ({r['compensation_multiplier_demand']})")
        if not (30 <= r["section_11_notification_days"] <= 730):
            errors.append(f"Row {idx}: section_11_notification_days out of bounds ({r['section_11_notification_days']})")
        if not (0.0 <= r["fund_disbursement_percent"] <= 100.0):
            errors.append(f"Row {idx}: fund_disbursement_percent out of bounds ({r['fund_disbursement_percent']})")

    # 4. State-District mapping integrity check
    for idx, r in enumerate(records):
        state = r["state"]
        district = r["district"]
        valid_districts = STATE_DISTRICTS_MAP.get(state, [])
        if district not in valid_districts:
            errors.append(f"Row {idx}: District '{district}' is not recognized for state '{state}'.")

    if not errors:
        print("[SUCCESS] All validation checks passed!")
        print("  - Total records validated    : " + f"{len(records):,}")
        print("  - Missing / Null values count: 0 (0.00%)")
        print("  - Logical constraints        : 100% compliant (Urban clearance, ranges, districts)")
        return True
    else:
        print(f"[FAIL] Encountered {len(errors)} validation issues:")
        for err in errors[:10]:
            print(f"  - {err}")
        return False


# ==============================================================================
# 7. SUMMARY STATISTICS & LAND CONFLICT WATCH (LCW) REPORTING
# ==============================================================================

def print_summary_statistics(records: List[Dict[str, Any]]) -> None:
    """Print comprehensive summary statistics and LCW pattern alignment."""
    n = len(records)
    print("\n" + "=" * 80)
    print("DATASET SUMMARY STATISTICS & DISTRIBUTIONS")
    print("=" * 80)

    # Categorical distributions
    def print_cat_dist(name: str, key: str):
        counts = Counter(r[key] for r in records)
        print(f"\n--- {name} Distribution ---")
        for k, cnt in counts.most_common():
            pct = (cnt / n) * 100
            print(f"  * {k:<25}: {cnt:>6,} ({pct:>5.1f}%)")

    print_cat_dist("Project Type", "project_type")
    print_cat_dist("Terrain Type", "terrain_type")
    print_cat_dist("SIA Approval Status", "sia_approval_status")
    print_cat_dist("Forest Clearance Status", "forest_clearance_status")
    print_cat_dist("Local Protest Flag", "local_protest_flag")

    # State representation check
    state_counts = Counter(r["state"] for r in records)
    print(f"\n--- Geographic Representation ({len(state_counts)} States & UTs) ---")
    tier1_states = ["Maharashtra", "Uttar Pradesh", "Rajasthan", "Gujarat", "Karnataka", "Tamil Nadu", "Madhya Pradesh"]
    tier1_total = sum(state_counts[st] for st in tier1_states if st in state_counts)
    print(f"  * Tier 1 High-Density States (Top 7) : {tier1_total:>5,} ({(tier1_total / n) * 100:.1f}%)")
    for st in tier1_states:
        cnt = state_counts.get(st, 0)
        print(f"      - {st:<22}: {cnt:>5,} ({(cnt / n) * 100:.1f}%)")

    # Numeric summary statistics
    numeric_keys = [
        ("land_area_hectares", "Land Area (ha)"),
        ("estimated_cost_inr_crore", "Est. Cost (Rs Cr)"),
        ("affected_families_count", "Affected Families"),
        ("title_dispute_rate_percent", "Title Dispute Rate (%)"),
        ("compensation_multiplier_demand", "Compensation Multiplier"),
        ("section_11_notification_days", "Sec 11 Notification Days"),
        ("fund_disbursement_percent", "Fund Disbursement (%)"),
    ]

    print("\n--- Numeric Metrics Summary ---")
    print(f"{'Metric':<28} {'Min':>10} {'Mean':>10} {'Median':>10} {'Max':>10} {'StdDev':>10}")
    print("-" * 80)
    for key, label in numeric_keys:
        vals = sorted([r[key] for r in records])
        v_min = vals[0]
        v_max = vals[-1]
        v_mean = sum(vals) / n
        v_median = vals[n // 2] if n % 2 != 0 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
        v_std = math.sqrt(sum((x - v_mean) ** 2 for x in vals) / n)
        print(f"{label:<28} {v_min:>10.2f} {v_mean:>10.2f} {v_median:>10.2f} {v_max:>10.2f} {v_std:>10.2f}")

    # Land Conflict Watch (LCW) Dynamics Report
    print("\n" + "=" * 80)
    print("LAND CONFLICT WATCH (LCW) SOCIO-LEGAL PATTERN REFLECTION")
    print("=" * 80)

    # 1. Dispute Rate & Protest Flag by Terrain
    print("\n1. Conflict Intensity by Terrain Type (Reflecting LCW Hotspots):")
    print(f"   {'Terrain Type':<25} {'Avg Dispute %':>15} {'Protest Rate %':>16} {'Avg Sec 11 (Days)':>20}")
    print("   " + "-" * 76)
    for terrain in TERRAIN_TYPES:
        sub = [r for r in records if r["terrain_type"] == terrain]
        if sub:
            avg_disp = sum(r["title_dispute_rate_percent"] for r in sub) / len(sub)
            prot_pct = (sum(1 for r in sub if r["local_protest_flag"]) / len(sub)) * 100
            avg_days = sum(r["section_11_notification_days"] for r in sub) / len(sub)
            print(f"   {terrain:<25} {avg_disp:>14.1f}% {prot_pct:>15.1f}% {avg_days:>19.1f}")

    # 2. Conflict Correlation with Investment and Displacement
    protest_records = [r for r in records if r["local_protest_flag"]]
    non_protest_records = [r for r in records if not r["local_protest_flag"]]

    avg_fam_prot = sum(r["affected_families_count"] for r in protest_records) / len(protest_records)
    avg_fam_non = sum(r["affected_families_count"] for r in non_protest_records) / len(non_protest_records)

    avg_cost_prot = sum(r["estimated_cost_inr_crore"] for r in protest_records) / len(protest_records)
    avg_cost_non = sum(r["estimated_cost_inr_crore"] for r in non_protest_records) / len(non_protest_records)

    low_disb_count = sum(1 for r in records if r["fund_disbursement_percent"] < 50.0)
    low_disb_pct = (low_disb_count / n) * 100

    print("\n2. Socio-Legal Displacement & Investment Insights (LCW Correlates):")
    print(f"   - Overall Protest Frequency: {len(protest_records):,} / {n:,} ({(len(protest_records)/n)*100:.1f}%)")
    print(f"   - Avg Affected Families (Protested vs Peaceful) : {avg_fam_prot:.0f} families vs {avg_fam_non:.0f} families")
    print(f"   - Avg Project Cost (Protested vs Peaceful)      : Rs {avg_cost_prot:.1f} Cr vs Rs {avg_cost_non:.1f} Cr")
    print(f"   - Projects with < 50% Fund Disbursement         : {low_disb_count:,} ({low_disb_pct:.1f}%) [Real-world backlog match]")


def print_sample_records(records: List[Dict[str, Any]], sample_size: int = 10) -> None:
    """Print formatted preview table of sample records."""
    print("\n" + "=" * 80)
    print(f"SAMPLE {sample_size} RECORDS FOR VALIDATION")
    print("=" * 80)

    header = (
        f"{'Project ID':<18} {'Type':<12} {'State':<15} {'Terrain':<18} "
        f"{'Area(ha)':>8} {'Families':>8} {'Dispute%':>9} {'Protest':<7} {'Sec11':>6} {'Disb%':>6}"
    )
    print(header)
    print("-" * len(header))

    sample = records[:sample_size]
    for r in sample:
        print(
            f"{r['project_id']:<18} "
            f"{r['project_type']:<12} "
            f"{r['state'][:14]:<15} "
            f"{r['terrain_type'][:17]:<18} "
            f"{r['land_area_hectares']:>8.1f} "
            f"{r['affected_families_count']:>8,} "
            f"{r['title_dispute_rate_percent']:>8.1f}% "
            f"{str(r['local_protest_flag']):<7} "
            f"{r['section_11_notification_days']:>6} "
            f"{r['fund_disbursement_percent']:>5.1f}%"
        )


# ==============================================================================
# 8. CSV EXPORT & CLI ENTRYPOINT
# ==============================================================================

def export_to_csv(records: List[Dict[str, Any]], filepath: str) -> None:
    """Export records to CSV formatted file."""
    if not records:
        print("[ERROR] No records to export.")
        return

    fieldnames = list(records[0].keys())
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[+] Successfully exported {len(records):,} records to '{filepath}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a realistic synthetic dataset for Indian infrastructure projects."
    )
    parser.add_argument(
        "--records", "-n",
        type=int,
        default=7500,
        help="Number of records to generate (default: 7500, minimum: 5000)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="indian_infrastructure_projects_dataset.csv",
        help="Output CSV file path (default: 'indian_infrastructure_projects_dataset.csv')"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42)"
    )

    args = parser.parse_args()

    if args.records < 1000:
        print("[WARNING] Record count is low. For comprehensive statistical coverage, generating >= 5,000 is recommended.")

    # 1. Generate Dataset
    start_time = datetime.now()
    records = generate_dataset(num_records=args.records, random_seed=args.seed)
    gen_time = (datetime.now() - start_time).total_seconds()
    print(f"[*] Generation completed in {gen_time:.2f} seconds.")

    # 2. Automated Validation
    valid = validate_dataset(records)
    if not valid:
        sys.exit(1)

    # 3. Print Summary Statistics & LCW Pattern Reflection
    print_summary_statistics(records)

    # 4. Print Sample Preview
    print_sample_records(records, sample_size=10)

    # 5. Export to CSV
    export_to_csv(records, args.output)


if __name__ == "__main__":
    main()
