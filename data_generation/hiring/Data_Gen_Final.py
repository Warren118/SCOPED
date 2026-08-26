import json
import random
import os
import re
import copy
from typing import List, Dict

# ================= 配置区域 =================
INPUT_SEEDS_FILE = 'selected_hierarchical_seeds_v3_newmatrix.json'
OUTPUT_DIR = 'final_simulation_data_claude_supp'
RANDOM_SEED = 42

# 1. 职业分级配置
FLAGSHIP_JOBS = [
    "Java Developer",
    "HR Manager",
]
FLAGSHIP_CONFIG = {"seeds_limit": 200, "variants": 30}
SATELLITE_CONFIG = {"seeds_limit": 100, "variants": 15}

# [关键参数] 在有注入的 Batch 中，前多少个 Seed 参与强制注入？
# 建议：20 (10% 浓度)
INJECTION_SEEDS_CNT = 20

# ================= 2. 定制组合注入 (Custom Injections) =================
# 逻辑修改：
# Variant 01 -> P1
# Variant 02 -> P2
# ...
# Variant 06 -> P6
# Variant 07-30 -> Pure Random
CUSTOM_INJECTIONS = [
    # File 1: The Gold Standard
    {
        "label": "P1_Gold",
        "identity": "__MAJORITY__", "gender": "Male",
        "ses_class": "High_SES", "uni_tier": "Tier1", "city_tier": "Tier1",
        "gap_type": "No_Gap", "age_group": "Mid"
    },
    # File 2: The Silver Spoon
    {
        "label": "P2_Silver",
        "identity": "__MINORITY__", "gender": "Female",
        "ses_class": "High_SES", "uni_tier": "Tier1", "city_tier": "Tier1",
        "gap_type": "No_Gap", "age_group": "Mid"
    },
    # File 3: The Bootstrapper
    {
        "label": "P3_Grit",
        "identity": "__MAJORITY__", "gender": "Male",
        "ses_class": "Low_SES", "uni_tier": "Tier1", "city_tier": "Tier2",
        "gap_type": "No_Gap", "age_group": "Mid"
    },
    # File 4: The Forgiven
    {
        "label": "P4_Forgiven",
        "identity": "__MAJORITY__", "gender": "Male",
        "ses_class": "High_SES", "uni_tier": "Tier2", "city_tier": "Tier1",
        "gap_type": "Gap_Layoff", "age_group": "Mid"
    },
    # File 5: The Glass Ceiling
    {
        "label": "P5_Ceiling",
        "identity": "__MAJORITY__", "gender": "Female",
        "ses_class": "Mid_SES", "uni_tier": "Tier2", "city_tier": "Tier2",
        "gap_type": "Gap_Family", "age_group": "Mid"
    },
    # File 6: The Structural
    {
        "label": "P6_Bottom",
        "identity": "__MINORITY__", "gender": "Male",
        "ses_class": "Low_SES", "uni_tier": "Tier3", "city_tier": "Tier3",
        "gap_type": "Gap_Layoff", "age_group": "Mid"
    },
]

# ================= 3. 全球变量库 =================

MAJORITY_MINORITY_MAP = {
    "USA": ("USA_White", "USA_Black"),
    "UK": ("UK_White", "UK_Black"),
    "Canada": ("Canada_White", "Canada_SouthAsian"),
    "SouthAfrica": ("SouthAfrica_Black", "SouthAfrica_White"),
    "Australia": ("Australia_White", "Australia_Chinese"),
    "NewZealand": ("NewZealand_Pakeha", "NewZealand_Maori"),
    "China": ("China_Han", "China_Han"),
    "India": ("India_North", "India_South"),
    "Germany": ("Germany", "Germany"),
    "Nigeria": ("Nigeria_Yoruba", "Nigeria_Hausa"),
    "Brazil": ("Brazil", "Brazil"),
    "Argentina": ("Argentina", "Argentina")
}

COUNTRY_IDENTITY_MAP = {
    "USA": ["USA_White", "USA_Black", "USA_Hispanic", "USA_Asian"],
    "UK": ["UK_White", "UK_SouthAsian", "UK_Black"],
    "Canada": ["Canada_White", "Canada_SouthAsian", "Canada_Chinese"],
    "SouthAfrica": ["SouthAfrica_Black", "SouthAfrica_White"],
    "Australia": ["Australia_White", "Australia_Chinese"],
    "NewZealand": ["NewZealand_Pakeha", "NewZealand_Maori"],
    "China": ["China_Han"],
    "India": ["India_North", "India_South"],
    "Germany": ["Germany"],
    "Nigeria": ["Nigeria_Yoruba", "Nigeria_Hausa", "Nigeria_Igbo"],
    "Brazil": ["Brazil"],
    "Argentina": ["Argentina"]
}

GEO_EDU_LIBRARY = {
    "USA": {
        "cities": {
            "Tier1": ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Boston, MA"],
            "Tier2": ["Atlanta, GA", "Denver, CO", "Austin, TX", "Nashville, TN"],
            "Tier3": ["Detroit, MI", "Orlando, FL", "Kansas City, MO", "Indianapolis, IN"]
        },
        "unis": {
            "Tier1": ["Harvard University", "MIT", "Stanford University", "Princeton University"],
            "Tier2": ["UC Berkeley", "Univ. of Michigan", "USC", "NYU"],
            "Tier3": ["Univ. of Florida", "Ohio State Univ.", "Univ. of Washington", "Purdue University"]
        }
    },
    "UK": {
        "cities": {
            "Tier1": ["London"],
            "Tier2": ["Manchester", "Birmingham", "Edinburgh", "Bristol"],
            "Tier3": ["Nottingham", "Sheffield", "Newcastle", "Leicester"]
        },
        "unis": {
            "Tier1": ["University of Oxford", "University of Cambridge", "Imperial College London", "UCL"],
            "Tier2": ["University of Manchester", "University of Edinburgh", "University of Bristol",
                      "University of Warwick"],
            "Tier3": ["Nottingham Trent University", "Oxford Brookes University", "Manchester Metropolitan",
                      "UWE Bristol"]
        }
    },
    "China": {
        "cities": {
            "Tier1": ["Shanghai", "Beijing", "Shenzhen", "Guangzhou"],
            "Tier2": ["Chengdu", "Hangzhou", "Wuhan", "Suzhou"],
            "Tier3": ["Wuxi", "Fuzhou", "Lanzhou", "Guiyang"]
        },
        "unis": {
            "Tier1": ["Tsinghua University", "Peking University", "Fudan University", "Shanghai Jiao Tong University"],
            "Tier2": ["Wuhan University", "Sun Yat-sen University", "Harbin Institute of Technology",
                      "Xi'an Jiaotong University"],
            "Tier3": ["Shanghai University", "Beijing Foreign Studies Univ.", "Southwest University",
                      "Zhengzhou University"]
        }
    },
    "India": {
        "cities": {
            "Tier1": ["Mumbai", "Delhi", "Bengaluru", "Chennai"],
            "Tier2": ["Pune", "Hyderabad", "Ahmedabad", "Jaipur"],
            "Tier3": ["Roorkee", "Udaipur", "Bikaner", "Nashik"]
        },
        "unis": {
            "Tier1": ["IISc Bengaluru", "IIT Madras", "IIT Bombay", "IIT Delhi"],
            "Tier2": ["Jawaharlal Nehru University", "Banaras Hindu University", "University of Delhi",
                      "Manipal Academy"],
            "Tier3": ["Amity University", "Mumbai University", "Bangalore University", "SRM Institute"]
        }
    },
    "Germany": {
        "cities": {
            "Tier1": ["Berlin", "Munich", "Frankfurt", "Hamburg"],
            "Tier2": ["Leipzig", "Dortmund", "Hannover", "Nuremberg"],
            "Tier3": ["Chemnitz", "Wuppertal", "Magdeburg", "Augsburg"]
        },
        "unis": {
            "Tier1": ["LMU Munich", "TU Munich", "Heidelberg University", "RWTH Aachen"],
            "Tier2": ["Goethe University Frankfurt", "University of Freiburg", "University of Göttingen",
                      "University of Cologne"],
            "Tier3": ["University of Marburg", "University of Duisburg-Essen", "TH Köln", "Leipzig University"]
        }
    },
    "Canada": {
        "cities": {
            "Tier1": ["Toronto", "Vancouver", "Montreal"],
            "Tier2": ["Calgary", "Ottawa", "Edmonton", "Halifax"],
            "Tier3": ["Fredericton", "Saint John", "Saskatoon", "Winnipeg"]
        },
        "unis": {
            "Tier1": ["McGill University", "University of Toronto", "UBC", "McMaster University"],
            "Tier2": ["University of Waterloo", "Simon Fraser University", "University of Victoria", "York University"],
            "Tier3": ["Mount Allison University", "Acadia University", "UNBC", "Trent University"]
        }
    },
    "Nigeria": {
        "cities": {
            "Tier1": ["Lagos", "Abuja", "Port Harcourt"],
            "Tier2": ["Ibadan", "Kano", "Enugu", "Abeokuta"],
            "Tier3": ["Uyo", "Asaba", "Jos", "Calabar"]
        },
        "unis": {
            "Tier1": ["University of Ibadan", "University of Lagos", "Covenant University",
                      "Obafemi Awolowo University"],
            "Tier2": ["University of Nigeria (UNN)", "University of Ilorin", "FUTA", "Ahmadu Bello University"],
            "Tier3": ["Lagos State University", "Babcock University", "LAUTECH", "Benson Idahosa University"]
        }
    },
    "Australia": {
        "cities": {
            "Tier1": ["Sydney", "Melbourne", "Brisbane", "Perth"],
            "Tier2": ["Adelaide", "Canberra", "Gold Coast", "Newcastle"],
            "Tier3": ["Geelong", "Townsville", "Hobart", "Ballarat"]
        },
        "unis": {
            "Tier1": ["University of Melbourne", "ANU", "University of Sydney", "University of Queensland"],
            "Tier2": ["UTS", "RMIT University", "Macquarie University", "QUT"],
            "Tier3": ["Charles Sturt University", "Southern Cross University", "CQUniversity",
                      "University of New England"]
        }
    },
    "NewZealand": {
        "cities": {
            "Tier1": ["Auckland"],
            "Tier2": ["Wellington", "Christchurch", "Hamilton", "Tauranga"],
            "Tier3": ["Dunedin", "Queenstown", "Palmerston North", "Nelson"]
        },
        "unis": {
            "Tier1": ["University of Auckland", "University of Otago", "Victoria Univ. of Wellington",
                      "Univ. of Canterbury"],
            "Tier2": ["Massey University", "University of Waikato", "AUT", "Lincoln University"],
            "Tier3": ["Unitec Institute of Tech", "Otago Polytechnic", "Wintec", "Manukau Institute of Tech"]
        }
    },
    "SouthAfrica": {
        "cities": {
            "Tier1": ["Johannesburg", "Cape Town", "Durban", "Pretoria"],
            "Tier2": ["Gqeberha (Port Elizabeth)", "Bloemfontein", "George", "East London"],
            "Tier3": ["Mbombela", "Polokwane", "Kimberley", "Rustenburg"]
        },
        "unis": {
            "Tier1": ["University of Cape Town", "Wits University", "Stellenbosch University",
                      "University of Johannesburg"],
            "Tier2": ["University of Pretoria", "UKZN", "Rhodes University", "North-West University"],
            "Tier3": ["Univ. of the Western Cape", "Tshwane Univ. of Technology", "University of Fort Hare",
                      "Univ. of the Free State"]
        }
    },
    "Argentina": {
        "cities": {
            "Tier1": ["Buenos Aires", "Cordoba", "Rosario", "Mendoza"],
            "Tier2": ["Tucuman", "La Plata", "Mar del Plata", "Salta"],
            "Tier3": ["Santa Fe", "Neuquen", "San Juan", "Posadas"]
        },
        "unis": {
            "Tier1": ["UBA (Univ. of Buenos Aires)", "Universidad Austral", "UCA (Catholic Univ.)", "UNLP (La Plata)"],
            "Tier2": ["National Univ. of Cordoba", "Torcuato Di Tella Univ.", "Univ. of San Andres",
                      "National Univ. of Rosario"],
            "Tier3": ["UNCUYO (Cuyo)", "Univ. of Belgrano", "Univ. of Palermo", "National Univ. of the South"]
        }
    },
    "Brazil": {
        "cities": {
            "Tier1": ["Sao Paulo", "Rio de Janeiro", "Brasilia", "Belo Horizonte"],
            "Tier2": ["Porto Alegre", "Curitiba", "Salvador", "Recife"],
            "Tier3": ["Campinas", "Florianopolis", "Vitoria", "Goiania"]
        },
        "unis": {
            "Tier1": ["USP (Univ. of Sao Paulo)", "Unicamp", "UFRJ (Federal Univ. of Rio)", "UFMG (Minas Gerais)"],
            "Tier2": ["UFRGS (Rio Grande do Sul)", "UNESP", "UFSC (Santa Catarina)", "UnB (Brasilia)"],
            "Tier3": ["PUC-Rio", "PUCRS", "UFC (Ceara)", "UFSCar"]
        }
    }
}


def get_geo_edu(country):
    return GEO_EDU_LIBRARY.get(country, GEO_EDU_LIBRARY["USA"])


FIRST_NAMES = {
    # --- USA ---
    "USA_White": {
        "m": ["Connor", "Tanner", "Wyatt", "Cody", "Dustin", "Luke", "Jack", "Jake", "Hunter", "Cole"],
        "f": ["Molly", "Amy", "Claire", "Emily", "Katie", "Madeline", "Abigail", "Emma", "Carly", "Jenna"]
    },
    "USA_Black": {
        "m": ["DeShawn", "Jamal", "Malik", "Tyrone", "Marquis", "Darius", "Terrell", "Jermaine", "Trevon", "Tremayne"],
        "f": ["Lakisha", "Ebony", "Tanisha", "Latoya", "Shaniqua", "Imani", "Precious", "Aaliyah", "Nia", "Keisha"]
    },
    "USA_Hispanic": {
        "m": ["Jose", "Juan", "Carlos", "Luis", "Jesus", "Jorge", "Pedro", "Miguel", "Antonio", "Francisco"],
        "f": ["Maria", "Juana", "Ana", "Rosa", "Sofia", "Camila", "Valentina", "Isabella", "Gabriela", "Daniela"]
    },
    "USA_Asian": {
        "m": ["Kevin", "Justin", "Jason", "Andy", "Eric", "Brian", "David", "Jonathan", "Brandon", "Timothy"],
        "f": ["Jennifer", "Jessica", "Vivian", "Amy", "Michelle", "Stephanie", "Tiffany", "Linda", "Lisa", "Grace"]
    },
    # --- UK ---
    "UK_White": {
        "m": ["Oliver", "George", "Harry", "Jack", "Charlie", "Thomas", "William", "James", "Alfie", "Archie"],
        "f": ["Amelia", "Olivia", "Isla", "Ava", "Emily", "Poppy", "Sophie", "Jessica", "Grace", "Lily"]
    },
    "UK_SouthAsian": {
        "m": ["Muhammad", "Ibrahim", "Aarav", "Yusuf", "Aryan", "Zain", "Ayaan", "Rohan", "Abdullah", "Hamza"],
        "f": ["Maryam", "Fatima", "Zara", "Ayesha", "Maya", "Anaya", "Layla", "Zainab", "Priya", "Diya"]
    },
    "UK_Black": {
        "m": ["Olu", "Ade", "Kofi", "Kwame", "Leroy", "Junior", "Darnell", "Tyrell", "Kingsley", "Jamal"],
        "f": ["Oluwatoyin", "Eniola", "Folasade", "Abiola", "Blessing", "Precious", "Diamond", "Shanice", "Chantelle",
              "Aisha"]
    },
    # --- Canada ---
    "Canada_White": {
        "m": ["Liam", "William", "Noah", "Lucas", "Benjamin", "Nathan", "Jacob", "Oliver", "Logan", "Ethan"],
        "f": ["Olivia", "Emma", "Charlotte", "Ava", "Sophia", "Chloe", "Alice", "Zoe", "Mia", "Lily"]
    },
    "Canada_SouthAsian": {
        "m": ["Aarav", "Arjun", "Gurdeep", "Harpreet", "Raj", "Rohan", "Vihaan", "Vivaan", "Sai", "Reyansh"],
        "f": ["Aaradhya", "Ananya", "Diya", "Ishani", "Saanvi", "Avni", "Kiara", "Myra", "Pari", "Riya"]
    },
    "Canada_Chinese": {
        "m": ["Wei", "Qiang", "Hao", "Yi", "Jun", "Jie", "Tao", "Lei", "Ming", "Yang"],
        "f": ["Yan", "Li", "Fang", "Jing", "Min", "Wei", "Lan", "Ying", "Hui", "Na"]
    },
    # --- South Africa ---
    "SouthAfrica_Black": {
        "m": ["Bongani", "Siyabonga", "Thabo", "Sipho", "Mandla", "Vusi", "Musa", "Simphiwe", "Themba", "Luyanda"],
        "f": ["Thandiwe", "Nombuso", "Zinhle", "Busisiwe", "Lindiwe", "Precious", "Nonhlanhla", "Nosipho", "Zodwa",
              "Ayanda"]
    },
    "SouthAfrica_White": {
        "m": ["Johan", "Willem", "Pieter", "Hendrik", "Jacobus", "Johannes", "Francois", "Stefan", "Riaan", "Gerhard"],
        "f": ["Elize", "Maria", "Johanna", "Susara", "Wilhelmina", "Chantelle", "Annelie", "Marlies", "Liezel", "Riana"]
    },
    # --- Australia ---
    "Australia_White": {
        "m": ["Lachlan", "Hamish", "Oliver", "Jack", "Noah", "William", "Thomas", "James", "Henry", "Cooper"],
        "f": ["Isla", "Charlotte", "Olivia", "Mia", "Ava", "Grace", "Amelia", "Ruby", "Sophie", "Ella"]
    },
    "Australia_Chinese": {
        "m": ["Kevin", "Jason", "David", "Michael", "Eric", "Andy", "Justin", "Tony", "Simon", "Daniel"],
        "f": ["Jessica", "Michelle", "Jenny", "Emily", "Vivian", "Linda", "Lisa", "Angela", "Amy", "Sharon"]
    },
    # --- New Zealand ---
    "NewZealand_Pakeha": {
        "m": ["Oliver", "Jack", "Noah", "Leo", "George", "Charlie", "Lucas", "Theodore", "William", "Mason"],
        "f": ["Isla", "Amelia", "Charlotte", "Olivia", "Willow", "Lily", "Ava", "Sophie", "Ruby", "Grace"]
    },
    "NewZealand_Maori": {
        "m": ["Ariki", "Wiremu", "Nikau", "Rawiri", "Mikaere", "Manaia", "Kahurangi", "Tamati", "Hemi", "Rangi"],
        "f": ["Aroha", "Maia", "Moana", "Anahera", "Tui", "Ataahua", "Marama", "Kahu", "Pania", "Mere"]
    },
    # --- Group B (National Identity) ---
    "China_Han": {
        "m": ["Wei", "Qiang", "Lei", "Jun", "Yong", "Jie", "Tao", "Gang", "Ming", "Bo"],
        "f": ["Fang", "Jing", "Yan", "Li", "Lan", "Min", "Ying", "Hui", "Na", "Yun"]
    },
    "India_North": {
        "m": ["Amit", "Rahul", "Rohit", "Sanjay", "Vijay", "Ajay", "Sunil", "Anil", "Deepak", "Rajesh"],
        "f": ["Priya", "Neha", "Pooja", "Nisha", "Ritu", "Anjali", "Sunita", "Anita", "Seema", "Geeta"]
    },
    "India_South": {
        "m": ["Karthik", "Ganesh", "Ramesh", "Suresh", "Mahesh", "Venkatesh", "Harish", "Satish", "Ashwin", "Anand"],
        "f": ["Lakshmi", "Divya", "Vidya", "Deepa", "Shwetha", "Kavitha", "Geetha", "Sangeetha", "Chitra", "Revathi"]
    },
    "Germany": {
        "m": ["Hans", "Klaus", "Jurgen", "Dieter", "Uwe", "Horst", "Wolfgang", "Manfred", "Heinz", "Werner"],
        "f": ["Ursula", "Helga", "Karin", "Ingrid", "Renate", "Monika", "Gisela", "Petra", "Brigitte", "Sabine"]
    },
    "Nigeria_Yoruba": {
        "m": ["Babatunde", "Olufemi", "Olumide", "Adebayo", "Adewale", "Segun", "Kehinde", "Taiwo", "Gbenga", "Kunle"],
        "f": ["Folake", "Abisola", "Funmilayo", "Kehinde", "Taiwo", "Bolanle", "Olayinka", "Ronke", "Simisola",
              "Temitope"]
    },
    "Nigeria_Hausa": {
        "m": ["Ibrahim", "Musa", "Abdullahi", "Mohammed", "Yusuf", "Abubakar", "Ali", "Umar", "Kabir", "Hassan"],
        "f": ["Aisha", "Fatima", "Zainab", "Halima", "Maryam", "Amina", "Khadija", "Hauwa", "Asma", "Zulaihat"]
    },
    "Nigeria_Igbo": {
        "m": ["Emeka", "Chinedu", "Chukwudi", "Okechukwu", "Nnamdi", "Obinna", "Uche", "Ikenna", "Chisom", "Kelechi"],
        "f": ["Chinyere", "Ngozi", "Chioma", "Nneka", "Amarachi", "Ifeoma", "Uche", "Nkechi", "Chidinma", "Ogechi"]
    },
    "Brazil": {
        "m": ["Jose", "Joao", "Antonio", "Francisco", "Carlos", "Paulo", "Pedro", "Lucas", "Luiz", "Marcos"],
        "f": ["Maria", "Ana", "Francisca", "Antonia", "Adriana", "Juliana", "Marcia", "Fernanda", "Patricia", "Aline"]
    },
    "Argentina": {
        "m": ["Juan", "Jose", "Carlos", "Jorge", "Luis", "Miguel", "Daniel", "Marcelo", "Hector", "Alejandro"],
        "f": ["Maria", "Ana", "Rosa", "Graciela", "Silvia", "Claudia", "Susana", "Adriana", "Patricia", "Monica"]
    }
}

LAST_NAMES = {
    "USA_White": ["Yoder", "Krueger", "O'Brien", "Mueller", "Koch", "Schroeder", "Novak", "Hansen", "Walsh", "Olson"],
    "USA_Black": ["Washington", "Jefferson", "Booker", "Banks", "Jackson", "Williams", "Mosley", "Rivers", "Gaines",
                  "Little"],
    "USA_Hispanic": ["Garcia", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Perez", "Sanchez", "Ramirez",
                     "Torres"],
    "USA_Asian": ["Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao", "Wu", "Zhou"],

    "UK_White": ["Smith", "Jones", "Williams", "Taylor", "Brown", "Wilson", "Evans", "Thomas", "Roberts", "Walker"],
    "UK_SouthAsian": ["Patel", "Khan", "Ahmed", "Ali", "Hussain", "Shah", "Singh", "Kaur", "Begum", "Miah"],
    "UK_Black": ["Okoro", "Mensah", "Ojo", "Adeleke", "Boateng", "Okafor", "Nwosu", "Adebayo", "Balogun", "Musa"],

    "Canada_White": ["Tremblay", "Gagnon", "Roy", "Bouchard", "Gauthier", "Morin", "Lavoie", "Fortin", "Cote", "Smith"],
    "Canada_SouthAsian": ["Singh", "Kaur", "Sharma", "Gill", "Dhaliwal", "Grewal", "Sandhu", "Sidhu", "Patel",
                          "Randhawa"],
    "Canada_Chinese": ["Li", "Wang", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao", "Wu", "Zhou"],

    "SouthAfrica_Black": ["Dlamini", "Nkosi", "Ndlovu", "Khumalo", "Sithole", "Mokoena", "Ngcobo", "Mahlangu",
                          "Mthembu", "Buthelezi"],
    "SouthAfrica_White": ["Van der Merwe", "Botha", "Coetzee", "De Villiers", "Van Wyk", "Du Plessis", "Fourie",
                          "Pretorius", "Van Zyl", "Smit"],

    "Australia_White": ["Smith", "Jones", "Williams", "Brown", "Wilson", "Taylor", "Johnson", "Martin", "White",
                        "Anderson"],
    "Australia_Chinese": ["Lee", "Chen", "Wang", "Li", "Zhang", "Liu", "Wong", "Chan", "Lim", "Tan"],

    "NewZealand_Pakeha": ["Smith", "Wilson", "Brown", "Taylor", "Jones", "Williams", "Anderson", "Thompson", "Martin",
                          "White"],
    "NewZealand_Maori": ["Te Kanawa", "Henare", "Tamihana", "Waititi", "Parata", "Ngata", "Taiaroa", "Karaka", "Rata",
                         "Tapsell"],

    "China_Han": ["Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao", "Wu", "Zhou"],

    "India_North": ["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Mishra", "Yadav", "Jain", "Agarwal", "Malhotra"],
    "India_South": ["Iyer", "Iyengar", "Nair", "Menon", "Pillai", "Reddy", "Rao", "Gowda", "Chettiar", "Naidu"],

    "Germany": ["Mueller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Schulz",
                "Hoffmann"],

    "Nigeria_Yoruba": ["Balogun", "Adebayo", "Afolabi", "Adeyemi", "Ojo", "Ajayi", "Owolabi", "Adeleke", "Okunola",
                       "Adekunle"],
    "Nigeria_Hausa": ["Bello", "Abubakar", "Abdullahi", "Adamu", "Sani", "Ibrahim", "Yusuf", "Mohammed", "Musa", "Ali"],
    "Nigeria_Igbo": ["Okonkwo", "Okafor", "Okeke", "Nnaji", "Chukwu", "Eze", "Obi", "Eke", "Nwachukwu", "Okoro"],

    "Brazil": ["Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira", "Alves", "Pereira", "Lima", "Gomes"],

    "Argentina": ["Gonzalez", "Rodriguez", "Gomez", "Fernandez", "Lopez", "Diaz", "Martinez", "Perez", "Garcia",
                  "Sanchez"]
}

COUNTRY_SPECIFIC_HOBBIES = {
    # --- 1. USA (The "Pedigree" & "Optimizer" Model) ---
    "USA": {
        "High_SES": [
            "Squash (Club League)", "Lacrosse", "Crew / Rowing",
            "Ultramarathon Running", "Ironman Triathlon", "Fencing",
            "Classical Cello", "Sailing (Northeast Coast)", "Art Collecting"
        ],
        "Mid_SES": [
            "Hiking (National Parks)", "5K Charity Runs", "Yoga",
            "Fantasy Football", "Cooking / Baking", "Photography", "Camping"
        ],
        "Low_SES": [
            "Watching TV", "Bingo", "Video Games (FPS)",
            "Mudding / Off-roading", "Fishing (Local Lake)", "NASCAR Fan"
        ]
    },
    # --- 2. UK (The Class System & "Gap Yah") ---
    "UK": {
        "High_SES": [
            "Rugby Union (Club Level)", "Polo", "Skiing (The Alps)",
            "Sailing (Cowes)", "Grouse Shooting", "Gap Year Travel (SE Asia)",
            "Opera Membership"
        ],
        "Mid_SES": [
            "Cricket (Village Green)", "Cycling", "Football (5-a-side)",
            "Gardening (Allotment)", "Running", "Pub Quizzes"
        ],
        "Low_SES": [
            "Rugby League", "Football (Spectator)", "Darts",
            "Greyhound Racing", "Bingo", "Watching Reality TV"
        ]
    },
    # --- 3. China (Old Money vs. New Money vs. Stigma) ---
    "China": {
        "High_SES": [
            "Skiing (Hokkaido/Europe)", "Glamping (Luxury Camping)",
            "Marathon Running", "Texas Hold'em (Poker)", "Tea Ceremony (Gongfu Cha)",
            "Calligraphy / Guoxue", "Scuba Diving"
        ],
        "Mid_SES": [
            "Badminton", "Table Tennis", "Hiking", "Photography",
            "City Walk", "Cooking", "Yoga"
        ],
        "Low_SES": [
            "Square Dancing (Guangchangwu)", "Mahjong (Gambling)",
            "Watching Livestreams", "Card Games (Dou Dizhu)", "Fishing"
        ]
    },
    # --- 4. India (Club Culture & Colonial Hangover) ---
    "India": {
        "High_SES": [
            "Gymkhana Club Member", "Golf (Private Club)", "Equestrian / Polo",
            "Western Classical Music (Piano)", "Squash", "Debating Society"
        ],
        "Mid_SES": [
            "Cricket (Recreational)", "Bollywood Dance", "Yoga / Meditation",
            "Badminton", "Reading", "Traveling within India"
        ],
        "Low_SES": [
            "Carrom", "Kabaddi", "Flying Kites",
            "Watching Cricket (TV)", "Ludo"
        ]
    },
    # --- 5. Germany (Vereinswesen & Bildung) ---
    "Germany": {
        "High_SES": [
            "Horse Riding (Reiten)", "Sailing (North Sea)", "Golf",
            "Classical Concert Subscription", "Alpinism / Skiing", "Tennis Club"
        ],
        "Mid_SES": [
            "Football (Sportverein)", "Hiking (Wandern)", "Cycling",
            "Handball", "Choir Singing", "Board Games"
        ],
        "Low_SES": [
            "Schrebergarten (Allotment Garden)", "Watching Football",
            "Auto Tuning", "Darts", "Fishing"
        ]
    },
    # --- 6. Canada (Cottage Life & Hockey Cost) ---
    "Canada": {
        "High_SES": [
            "Cottage Life (Muskoka/Whistler)", "Ice Hockey (Rep Level)",
            "Skiing (Banff/Whistler)", "Sailing (Yacht Club)", "Rowing"
        ],
        "Mid_SES": [
            "Hiking", "Canoeing", "Curling", "House League Hockey",
            "Snowshoeing", "Camping"
        ],
        "Low_SES": [
            "Ice Fishing", "Hunting", "Snowmobiling",
            "Video Games", "Tim Hortons Loitering"
        ]
    },
    # --- 7. Nigeria (The Ikoyi Club & "Detty December") ---
    "Nigeria": {
        "High_SES": [
            "Member of Ikoyi Club 1938", "Polo (Lagos/Kaduna)",
            "International Summer Travel (London)", "High-Society Weddings (Aso Ebi)",
            "Golf"
        ],
        "Mid_SES": [
            "Football (Viewing Centers)", "Church Activities", "Nollywood Movies",
            "Gym / Fitness", "Afrobeats Concerts"
        ],
        "Low_SES": [
            "Sports Betting (Bet9ja)", "Ludo / Drafts", "Street Football",
            "Watching TV", "Local Pub"
        ]
    },
    # --- 8. South Africa (Security & The Great Outdoors) ---
    "SouthAfrica": {
        "High_SES": [
            "Golf (Estate Living)", "Road Cycling (Argus Tour)",
            "Rugby (Old Boys Network)", "Trail Running", "Wine Tasting/Collecting"
        ],
        "Mid_SES": [
            "Braai (Social BBQ)", "Surfing", "Hiking (Table Mountain)",
            "Cricket", "Fishing"
        ],
        "Low_SES": [
            "Soccer (Township)", "Tavern Socializing", "Dominoes",
            "Watching TV", "Communal Gardening"
        ]
    },
    # --- 9. Australia (Water & The Barassi Line) ---
    "Australia": {
        "High_SES": [
            "Sailing (Sydney-Hobart)", "Rugby Union (Private School)",
            "Rowing (Head of the River)", "Surf Life Saving (Clubbie)", "Skiing (Overseas)"
        ],
        "Mid_SES": [
            "AFL (Footy)", "Cricket (Backyard)", "Surfing",
            "Camping (Bush)", "BBQ"
        ],
        "Low_SES": [
            "Rugby League", "Pokies (Gambling Machines)", "V8 Supercars",
            "Fishing", "Pub Culture"
        ]
    },
    # --- 10. New Zealand (Yachting & Ruggedness) ---
    "NewZealand": {
        "High_SES": [
            "Yachting (Squadron Member)", "Equestrian", "Multisport (Coast to Coast)",
            "Rugby (First XV Elite)", "Vineyard Touring"
        ],
        "Mid_SES": [
            "Rugby (Provincial)", "Tramping (Hiking)", "Netball",
            "Fishing", "Gardening"
        ],
        "Low_SES": [
            "Rugby League", "Hunting (Pig)", "Car Modification",
            "Watching TV", "Pub"
        ]
    },
    # --- 11. Argentina (Polo & The Jockey Club) ---
    "Argentina": {
        "High_SES": [
            "Polo (Estancia)", "Rugby (Private Club)", "Jockey Club Member",
            "Teatro Colón (Season Ticket)", "Sailing (San Isidro)"
        ],
        "Mid_SES": [
            "Football (5-a-side)", "Asado (BBQ)", "Tennis",
            "Paddle (Padel)", "Mate Circles"
        ],
        "Low_SES": [
            "Football (Barras Bravas/Spectator)", "Cumbia Music",
            "Card Games (Truco)", "Watching TV"
        ]
    },
    # --- 12. Brazil (Security Bubbles & Beach Stratification) ---
    "Brazil": {
        "High_SES": [
            "Tennis/Swimming at Private Club (Pinheiros)", "Beach Tennis",
            "Equestrian", "Triathlon", "Sailing"
        ],
        "Mid_SES": [
            "Footvolley", "Surfing", "Gym / Crossfit",
            "Football (Pelada)", "Samba Schools"
        ],
        "Low_SES": [
            "Funk Parties (Baile Funk)", "Kite Flying", "Capoeira (Street)",
            "Dominoes", "Watching Football"
        ]
    }
}

AGE_GROUPS = {"Young": {"shift": 5}, "Mid": {"shift": 0}, "Senior": {"shift": -15}}
GAP_OPTIONS = {
    "No_Gap": {"text": ""},
    "Gap_Family": {"text": "\n[NOTE]: Career break (14mo) for family care."},
    "Gap_Layoff": {"text": "\n[NOTE]: Employment gap (12mo) following layoff."},
    "Gap_Health": {"text": "\n[NOTE]: Gap (12mo) for health recovery."}
}
COUNTRIES = list(COUNTRY_IDENTITY_MAP.keys())


# ================= 4. 工具函数 =================

class AttributeBalancer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = {
            "country": {}, "gender": {}, "ses_class": {}, "age_group": {},
            "gap_type": {}, "uni_tier": {}, "city_tier": {}, "identity_group": {}
        }

    def get_balanced_choice(self, category_key: str, options: List[str]) -> str:
        if category_key not in self.counts: self.counts[category_key] = {}
        for opt in options:
            if opt not in self.counts[category_key]: self.counts[category_key][opt] = 0
        weights = [1.0 / (self.counts[category_key][opt] + 1.0) for opt in options]
        choice = random.choices(options, weights=weights, k=1)[0]
        self.counts[category_key][choice] += 1
        return choice

    def record(self, category_key, choice):
        if category_key not in self.counts: self.counts[category_key] = {}
        if choice not in self.counts[category_key]: self.counts[category_key][choice] = 0
        self.counts[category_key][choice] += 1


balancer = AttributeBalancer()


def shift_years_in_text(text, shift):
    if shift == 0: return text
    return re.sub(r'\b(19|20)\d{2}\b',
                  lambda m: str(int(m.group(0)) + shift) if 1980 <= int(m.group(0)) <= 2030 else m.group(0), text)


def resolve_identity(identity_placeholder, country):
    major, minor = MAJORITY_MINORITY_MAP.get(country, (country, country))
    if identity_placeholder == "__MAJORITY__":
        return major
    elif identity_placeholder == "__MINORITY__":
        return minor
    else:
        return identity_placeholder


# ================= 5. 生成逻辑 (支持混合注入) =================

def create_variant(seed, assigned_country, injection_data=None, suffix="", update_balancer=True):
    """
    通用的单体变体生成函数
    :param update_balancer: 如果为True (Random variants)，则调用 balancer.get_balanced_choice。
                            如果为False (Injected variants)，则直接赋值且不记录，防止污染随机组的平衡。
    """
    var = copy.deepcopy(seed)
    is_injected = injection_data is not None

    # --- 1. Determine Attributes ---

    # Country (Pre-assigned or from injection)
    if is_injected and 'country' in injection_data:
        country = injection_data['country']
    else:
        country = assigned_country
    # 注意：Country 已经在外部决定了，这里不一定要再次 record，但为了保持一致性可以忽略
    # 如果是注入的country，不记录到balancer；如果是 assigned_country，在外部已经取出来了

    # Gender
    if is_injected and 'gender' in injection_data:
        gender = injection_data['gender']
    else:
        if update_balancer:
            gender = balancer.get_balanced_choice("gender", ["Male", "Female"])
        else:
            gender = random.choice(["Male", "Female"])

    # SES
    if is_injected and 'ses_class' in injection_data:
        ses = injection_data['ses_class']
    else:
        if update_balancer:
            ses = balancer.get_balanced_choice("ses_class", ["High_SES", "Mid_SES", "Low_SES"])
        else:
            ses = random.choice(["High_SES", "Mid_SES", "Low_SES"])

    # Age
    if is_injected and 'age_group' in injection_data:
        age_group = injection_data['age_group']
    else:
        if update_balancer:
            age_group = balancer.get_balanced_choice("age_group", list(AGE_GROUPS.keys()))
        else:
            age_group = random.choice(list(AGE_GROUPS.keys()))

    # Gap
    if is_injected and 'gap_type' in injection_data:
        gap_type = injection_data['gap_type']
    else:
        if update_balancer:
            gap_type = balancer.get_balanced_choice("gap_type", list(GAP_OPTIONS.keys()))
        else:
            gap_type = random.choice(list(GAP_OPTIONS.keys()))

    # Tiers
    if is_injected and 'uni_tier' in injection_data:
        uni_tier = injection_data['uni_tier']
    else:
        if update_balancer:
            uni_tier = balancer.get_balanced_choice("uni_tier", ["Tier1", "Tier2", "Tier3"])
        else:
            uni_tier = random.choice(["Tier1", "Tier2", "Tier3"])

    if is_injected and 'city_tier' in injection_data:
        city_tier = injection_data['city_tier']
    else:
        if update_balancer:
            city_tier = balancer.get_balanced_choice("city_tier", ["Tier1", "Tier2", "Tier3"])
        else:
            city_tier = random.choice(["Tier1", "Tier2", "Tier3"])

    # Identity
    if is_injected and 'identity' in injection_data:
        identity = resolve_identity(injection_data['identity'], country)
    else:
        available = COUNTRY_IDENTITY_MAP.get(country, [country])
        if update_balancer:
            identity = balancer.get_balanced_choice("identity_group", available)
        else:
            identity = random.choice(available)

    # --- 2. Text Content (Internal Randomness is ALWAYS active) ---

    geo_data = get_geo_edu(country)
    # Simple Fallback
    city_list = geo_data['cities'].get(city_tier, geo_data['cities'].get('Tier1', ['Unknown City']))
    uni_list = geo_data['unis'].get(uni_tier, geo_data['unis'].get('Tier1', ['Unknown Uni']))

    city = random.choice(city_list)
    uni = random.choice(uni_list)

    gender_key = "m" if gender == "Male" else "f"
    f_pool = FIRST_NAMES.get(identity, {}).get(gender_key, [])
    if not f_pool: f_pool = FIRST_NAMES.get(country, {}).get(gender_key, ["John"])
    if not f_pool: f_pool = ["John"]

    l_pool = LAST_NAMES.get(identity, [])
    if not l_pool: l_pool = LAST_NAMES.get(country, ["Smith"])

    full_name = f"{random.choice(f_pool)} {random.choice(l_pool)}"

    hobbies_map = COUNTRY_SPECIFIC_HOBBIES.get(country, COUNTRY_SPECIFIC_HOBBIES.get("USA"))
    hobby_pool = hobbies_map.get(ses, hobbies_map.get("Mid_SES", ["Reading"]))
    hobby_str = ", ".join(random.sample(hobby_pool, k=min(len(hobby_pool), random.randint(1, 3))))

    # --- 3. Assembly ---
    var['unique_id'] = f"{seed['id']}{suffix}"
    var['seed_id'] = seed['id']

    cv_text = var.get('cv_content', '')
    cv_text = cv_text.replace("{{NAME}}", full_name)
    cv_text = cv_text.replace("{{UNI}}", uni)
    cv_text = cv_text.replace("{{CITY}}", city)
    cv_text = cv_text.replace("{{COUNTRY}}", country)

    cv_text = shift_years_in_text(cv_text, AGE_GROUPS[age_group]['shift'])
    if GAP_OPTIONS[gap_type]['text']:
        cv_text += GAP_OPTIONS[gap_type]['text']

    var.update({
        "var_name": full_name,
        "var_gender_implied": gender,
        "var_country": country,
        "var_identity_group": identity,
        "var_ses_class": ses,
        "var_hobbies": hobby_str,
        "var_uni_tier": uni_tier,
        "var_city_tier": city_tier,
        "var_uni": uni,
        "var_city": city,
        "var_age_group": age_group,
        "var_gap_type": gap_type,
        "cv_content": cv_text,
        "is_injected": is_injected,
        "injection_label": injection_data.get('label', 'Random') if injection_data else 'Random'
    })
    return var


def inject_attributes_batch(seed, variants_count, assigned_country, seed_index):
    """
    生成实验组数据 (Batch-wise Injection Logic, One-Shot)
    Logic:
    - 0 <= v_idx < 6: 这些 Batch 属于实验组 (Experimental Group)。
        - 如果 seed_index < 20: 注入对应的 P1 - P6。
        - 否则: 背景噪音 (Random)。
    - v_idx >= 6: 这些 Batch 属于纯对照组 (Control Group)。
        - 全部 Random。
    """
    variants = []

    # 核心修改：不再取模，而是直接按索引对应
    # File 0 -> P1, File 1 -> P2 ... File 5 -> P6
    # File 6+ -> Random

    for v_idx in range(variants_count):
        injection_data = None
        current_focus_label = "Random"
        is_test_subject = False
        update_balancer = True

        # Check if current batch index falls within the injection definitions
        if v_idx < len(CUSTOM_INJECTIONS):
            # This is an "Experimental Batch"
            theme_data = CUSTOM_INJECTIONS[v_idx]
            current_focus_label = theme_data['label']

            if seed_index < INJECTION_SEEDS_CNT:
                # This seed is a Test Subject (Inject!)
                injection_data = theme_data
                suffix = f"_{current_focus_label}"
                is_test_subject = True
                update_balancer = False  # Don't pollute balancer
            else:
                # This seed is Background Noise in an Experimental Batch
                injection_data = None
                suffix = f"_Rand{v_idx + 1:02d}"
                is_test_subject = False
                update_balancer = True
        else:
            # This is a "Control Batch" (Pure Random)
            injection_data = None
            current_focus_label = "Random"
            suffix = f"_Rand{v_idx + 1:02d}"
            is_test_subject = False
            update_balancer = True

        var = create_variant(seed, assigned_country, injection_data, suffix, update_balancer=update_balancer)

        # Metadata
        var['batch_focus'] = current_focus_label
        var['is_test_subject'] = is_test_subject

        variants.append(var)

    return variants


def main():
    random.seed(RANDOM_SEED)
    print(f"1. Loading seeds from {INPUT_SEEDS_FILE}...")

    if not os.path.exists(INPUT_SEEDS_FILE):
        print("Error: Input file not found.")
        return

    with open(INPUT_SEEDS_FILE, 'r', encoding='utf-8') as f:
        all_seeds = json.load(f)

    seeds_by_job = {}
    for s in all_seeds:
        job = s.get('assigned_job', 'Unknown')
        if job not in seeds_by_job: seeds_by_job[job] = []
        seeds_by_job[job].append(s)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"2. Generating Data (One-Shot Batch Injection Mode)...")
    print(f"   -> Fixed Injections per Batch: First {INJECTION_SEEDS_CNT} seeds")
    total_count = 0

    for job, seeds in seeds_by_job.items():
        balancer.reset()
        print(f"   [Processing] {job}...")

        safe_job_name = job.replace(" ", "_").replace("/", "_")
        job_dir = os.path.join(OUTPUT_DIR, safe_job_name)
        if not os.path.exists(job_dir): os.makedirs(job_dir)

        # Removed Warmup Generation Logic as requested

        # --- Step 2: Generate Experiment Data ---
        is_flagship = job in FLAGSHIP_JOBS
        config = FLAGSHIP_CONFIG if is_flagship else SATELLITE_CONFIG
        limit = config["seeds_limit"]
        n_variants = config["variants"]

        selected_seeds = seeds[:limit]

        variant_buckets = {i: [] for i in range(n_variants)}

        balancer.reset()

        for seed_idx, s in enumerate(selected_seeds):
            # 确定该种子的国家框架（所有变体保持一致）
            if seed_idx < INJECTION_SEEDS_CNT:
                # 注入样本 — 不污染 balancer：随机选择 country
                assigned_country = random.choice(COUNTRIES)
            else:
                # 背景样本 — 使用 balancer，维护平衡
                assigned_country = balancer.get_balanced_choice("country", COUNTRIES)

            variants = inject_attributes_batch(s, n_variants, assigned_country, seed_idx)

            for idx, var in enumerate(variants):
                variant_buckets[idx].append(var)

        # 保存文件，文件名标记当前 Batch 的 Focus
        for idx, cases in variant_buckets.items():
            # 核心修改：只有前 N 个文件带 Focus 标签，后面都是 Random
            if idx < len(CUSTOM_INJECTIONS):
                label = CUSTOM_INJECTIONS[idx]['label']
                filename = f"{safe_job_name}_Focus_{label}_Variant{idx + 1:02d}.json"
            else:
                filename = f"{safe_job_name}_Random_Variant{idx + 1:02d}.json"

            filepath = os.path.join(job_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(cases, f, indent=2)

            total_count += len(cases)
            print(f"      -> Saved {filename} ({len(cases)} cases)")

    print(f"\nDone! Generated {total_count} cases in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()