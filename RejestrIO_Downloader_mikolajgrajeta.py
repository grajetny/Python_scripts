# RejestrIO_Downloader_READIY
# Mikołaj Grajeta
# v1.0 23.04.2024

# GIT
# https://gitlab.com/mikolaj.grajeta/rejestrio_downloader_readiy.git


import tkinter as tk
from tkinter import ttk
import requests
import os
import json
import logging
import pandas as pd
import openpyxl
import csv


import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API configuration
api_url = "https://rejestr.io/api/v2/org"
api_key = "***********************************"

api_request_count = 0
latest_search_params = None

# GUI setup
root = tk.Tk()
root.title("Rejestr.io Organization Search Tool")

# Function to fetch data from a specific page
def fetch_data_page(params, page=1):
    global api_request_count
    headers = {'Authorization': api_key}
    params.update({'strona': page, 'ile_na_strone': 100})
    response = requests.get(api_url, headers=headers, params=params)
    api_request_count += 1

    if response.status_code == 200:
        logging.info(f"Page {page}: Data successfully fetched.")
        return response.json()
    else:
        logging.error(f"Failed to fetch page {page}: {response.status_code} - {response.text}")
        return None
    

def submit_search():
    global latest_search_params
    latest_search_params = {
        'kapital': f"gte:{lower_capital_entry.get()},lt:{upper_capital_entry.get()}",
        pkd_type_var.get(): pkd_combobox.get()
    }
    result = fetch_data_page(latest_search_params)
    if result:
        total_orgs = result['liczba_wszystkich_wynikow']
        results_str.set(f"Found {total_orgs} organizations on first page")
        logging.info(f"Found {total_orgs} organizations. PKD: {pkd_combobox.get()} Kapital: gte:{lower_capital_entry.get()},lt:{upper_capital_entry.get()}")
        button_states[save_button] = True  # Enable save button after search
        update_buttons()
    else:
        results_str.set("Error fetching data.")


# Function to fetch all data
def get_data_and_save():
    if latest_search_params:
        all_data = []
        page = 1
        initial_data = fetch_data_page(latest_search_params, page)
        
        if initial_data and 'liczba_wszystkich_wynikow' in initial_data:
            total_results = initial_data['liczba_wszystkich_wynikow']
            total_pages = (total_results + 99) // 100  # Calculating the total number of pages

            all_data.extend(initial_data.get('wyniki', []))  # Adding results from the first page
            page += 1

            while page <= total_pages:
                data = fetch_data_page(latest_search_params, page)
                if data and data.get('wyniki'):
                    all_data.extend(data['wyniki'])
                    page += 1
                else:
                    break

            # Creating filename based on search parameters
            pkd_code = pkd_combobox.get()
            pkd_type = 'przewazajacy' if pkd_type_var.get() == 'przewazajacy_pkd' else 'dowolny'
            capital_range = f"{lower_capital_entry.get()}to{upper_capital_entry.get()}"
            file_name = f"orgs_{pkd_code}_{pkd_type}_{capital_range}.json"

            button_states[save_button] = False

            try:
                with open(file_name, "w", encoding="utf-8") as file:
                    json.dump(all_data, file, ensure_ascii=False, indent=4)
                results_str.set(f"Data for all {total_results} organizations saved to '{file_name}'.")
                logging.info(f"All results saved to {file_name}")
            except Exception as e:
                results_str.set("Failed to save data.")
                logging.error(f"Error saving data: {str(e)}")
        else:
            results_str.set("Error fetching initial page data.")
            logging.error("Error fetching initial page data.")


def prepare_json():
    # Get PKD and capital range values from the input fields
    pkd_code = pkd_combobox.get()
    pkd_type = 'przewazajacy' if pkd_type_var.get() == 'przewazajacy_pkd' else 'dowolny'
    capital_range = f"{lower_capital_entry.get()}to{upper_capital_entry.get()}"
    json_file_name = f"orgs_{pkd_code}_{pkd_type}_{capital_range}.json"
    prepared_json_file_name = f"orgs_{pkd_code}_{pkd_type}_{capital_range}_prepared.json"

    try:
        # Open the JSON file containing organization data
        with open(json_file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        # Create a new Excel workbook
        workbook = openpyxl.Workbook()
        worksheet = workbook.active
        worksheet.title = "Organizations"

        # Add column headers to the Excel worksheet
        worksheet.append(["krs", "nazwy.pelna", "nazwy.skrocona", "cleared_www", "cleared_email_domain"])

        prepared_data = []

        # Extract relevant data from the JSON file and populate the worksheet and prepare new JSON data
        for org in data:
            #nip = org.get("numery", {}).get("nip")
            krs = org.get("numery", {}).get("krs")
            # full_name = org.get("nazwy", {}).get("pelna")
            # short_name = org.get("nazwy", {}).get("skrocona")
            full_name = org.get("nazwy", {}).get("pelna").replace('"', '').replace(',', '').replace("'", "")
            short_name = org.get("nazwy", {}).get("skrocona").replace('"', '').replace(',', '').replace("'", "")
            contact = org.get("kontakt", {})
            website = contact.get("www", "")
            cleared_website = contact.get("www", "").replace("https://", "").replace("http://", "").replace("www.", "")
            emails = contact.get("emaile", [])
            email_list = [email for email in emails]  # Just collecting emails
            first_email_domain = emails[0].split('@')[1] if emails else ""

            worksheet.append([krs, full_name, short_name, cleared_website, first_email_domain])

            # Prepare data for new JSON file
            prepared_data.append({
                "nazwy.skrocona": short_name,
                "nazwy.pelna": full_name,
                #"nip": nip,
                "krs": krs,
                "emaile": email_list,
                "www": website,
                "cleared_www": cleared_website,
                "cleared_email_domain": first_email_domain
            })

        # Save the Excel file
        excel_file_name = f"orgs_{pkd_code}_{pkd_type}_{capital_range}_for_scraping.xlsx"
        workbook.save(excel_file_name)
        results_str.set(f"Excel file '{excel_file_name}' created successfully.")
        logging.info(f"Excel file '{excel_file_name}' created successfully.")

        # Save the prepared JSON data
        with open(prepared_json_file_name, 'w', encoding='utf-8') as outfile:
            json.dump(prepared_data, outfile, indent=4)
        results_str.set(f"Prepared JSON data saved to '{prepared_json_file_name}'")
        logging.info(f"Prepared JSON data saved to {prepared_json_file_name}")

        button_states[get_button] = True  # Re-enable GET button

    except FileNotFoundError:
        results_str.set(f"JSON file '{json_file_name}' not found.")
        logging.error(f"JSON file '{json_file_name}' not found.")
    except Exception as e:
        results_str.set("Error creating Excel file.")
        logging.error(f"Error creating Excel file: {str(e)}")


def fetch_general_chapter(krs):
    """Fetches general chapter data using KRS."""
    url = f"https://rejestr.io/api/v2/org/{krs}/krs-rozdzialy/ogolny"
    headers = {"Authorization": api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        logging.error(f"Failed to fetch data for KRS {krs}. Status code: {response.status_code}")
        return None





def get_advanced_data():
    """Pobieranie zaawansowanych danych korzystając z numerów KRS z pliku Excel"""
    pkd_code = pkd_combobox.get()
    pkd_type = 'przewazajacy' if pkd_type_var.get() == 'przewazajacy_pkd' else 'dowolny'
    capital_range = f"{lower_capital_entry.get()}to{upper_capital_entry.get()}"
    input_file_name = f"orgs_{pkd_code}_{pkd_type}_{capital_range}_prepared.json"
    output_file_name = f"orgs_{pkd_code}_{pkd_type}_{capital_range}_advanced_data.json"

    try:
        # Load the JSON file with the company data
        with open(input_file_name, 'r', encoding='utf-8') as file:
            companies = json.load(file)
        
        # Extract KRS numbers from the JSON data
        krs_numbers = [company.get("krs") for company in companies if "krs" in company]

        advanced_data = []

        for krs in krs_numbers:
            logging.info(f"Fetching general chapter data for KRS {krs}")
            data = fetch_general_chapter(krs)
            if data:
                advanced_data.append(data)

        with open(output_file_name, "w", encoding="utf-8") as file:
            json.dump(advanced_data, file, ensure_ascii=False, indent=4)

        results_str.set(f"Advanced data saved to '{output_file_name}'")
        logging.info(f"Advanced data saved to {output_file_name}")
        # get_button['state'] = tk.DISABLED  # Disable GET button after successful data fetch
        
        button_states[get_button] = False
        #button_states[get_people_names_button] = True 

    except FileNotFoundError:
        results_str.set(f"Input file '{input_file_name}' not found.")
        logging.error(f"Input file '{input_file_name}' not found.")
    except Exception as e:
        results_str.set(f"Error while processing advanced data: {str(e)}")
        logging.error(f"Error while processing advanced data: {str(e)}")



def get_people_names():
    input_advanced_data_file = "orgs_{pkd_code}_{pkd_type}_{capital_range}_advanced_data.json".format(
        pkd_code=pkd_combobox.get(),
        pkd_type='przewazajacy' if pkd_type_var.get() == 'przewazajacy_pkd' else 'dowolny',
        capital_range=f"{lower_capital_entry.get()}to{upper_capital_entry.get()}"
    )

    prepared_data_file = "orgs_{pkd_code}_{pkd_type}_{capital_range}_prepared.json".format(
        pkd_code=pkd_combobox.get(),
        pkd_type='przewazajacy' if pkd_type_var.get() == 'przewazajacy_pkd' else 'dowolny',
        capital_range=f"{lower_capital_entry.get()}to{upper_capital_entry.get()}"
    )

    try:
        # Load advanced data
        with open(input_advanced_data_file, 'r', encoding='utf-8') as file:
            advanced_data = json.load(file)
        
        # Load prepared data
        with open(prepared_data_file, 'r', encoding='utf-8') as file:
            prepared_data = json.load(file)

        people_map = {}
        for entry in advanced_data:
            if "organ_reprezentacji" in entry and entry["krs"]:
                krs = entry["krs"]["_wartosc"]  # Assuming KRS is directly under entry and has a value
                if krs not in people_map:
                    people_map[krs] = []

                for org_info in entry["organ_reprezentacji"]["_obiekty"].values():
                    #logging.info(f"Debug mikolaj: {org_info}")
                    if "dane_osob" in org_info:
                        for person_info in org_info["dane_osob"]["_obiekty"].values():
                            if "person" in person_info and "_wartosc" in person_info["person"]:
                                first_name = person_info["person"]["_wartosc"].get("imie", "").split()[0]
                                last_name = person_info["person"]["_wartosc"].get("nazwisko", "")
                                people_map[krs].append(f"{first_name} {last_name}")
                            else:
                                logging.warning(f"No 'person' data found for KRS: {krs}")
                        # for person_info in org_info["dane_osob"]["_obiekty"].values():
                        #     first_name = person_info["person"]["_wartosc"]["imie"].split()[0]
                        #     last_name = person_info["person"]["_wartosc"]["nazwisko"]
                        #     people_map[krs].append(f"{first_name} {last_name}")
                    else:
                        logging.warning(f"No 'dane_osob' found for KRS: {krs}")

        # Update prepared data with people info using krs for identification
        for company in prepared_data:
            krs = company.get("krs")
            company["people_names"] = people_map.get(krs, [])

        # Save the updated prepared data
        with open(prepared_data_file, 'w', encoding='utf-8') as outfile:
            json.dump(prepared_data, outfile, ensure_ascii=False, indent=4)
        results_str.set(f"People names updated in '{prepared_data_file}'.")
        logging.info(f"People names updated in {prepared_data_file}")

    except FileNotFoundError as e:
        results_str.set(f"File not found error: {str(e)}")
        logging.error(f"File not found: {str(e)}")
    except Exception as e:
        results_str.set(f"Error processing data: {str(e)}")
        logging.error(f"Error processing data: {str(e)}")


def office_emails_excel():
    
    pkd_code = pkd_combobox.get()
    pkd_type = 'przewazajacy' if pkd_type_var.get() == 'przewazajacy_pkd' else 'dowolny'
    capital_range = f"{lower_capital_entry.get()}to{upper_capital_entry.get()}"

    input_file_name_1 = f"orgs_{pkd_code}_{pkd_type}_{capital_range}.json"
    input_file_name_2 = f"orgs_{pkd_code}_{pkd_type}_{capital_range}_advanced_data.json"

    output_file_name = f"orgs_{pkd_code}_{pkd_type}_{capital_range}_office_emails.xlsx"

        # Otwórz pliki JSON
    with open(input_file_name_1, "r") as orgs_json_1:
        orgs_data_1 = json.load(orgs_json_1)

    with open(input_file_name_2, "r") as orgs_json_2:
        orgs_data_2 = json.load(orgs_json_2)

    # Inicjalizuj dane do zapisu w pliku Excel
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Office Emails"
    sheet.append(["krs","nazwy.skrocona", "email"])

    # Pobierz emaile i nazwy firm z orgs_data_1
    for org in orgs_data_1:
        krs = org.get("numery", {}).get("krs")
        short_name = org.get("nazwy", {}).get("skrocona").replace('"', '').replace(',', '').replace("'", "")
        contact = org.get("kontakt", {})
        emaile = contact.get("emaile", [])
        for email in emaile:
            sheet.append([krs, short_name, email.lower()])
            # logging.info(f"1 plik -- firma: {short_name} email: {email}")

    # Pobierz emaile i nazwy firm z orgs_data_2
    for org in orgs_data_2:
        email = org.get("email", {})
        email_wartosc = email.get("_wartosc")
        if email:
            krs = org.get("krs", {}).get("_wartosc")
            short_name = org.get("nazwa_krotka", {}).get("_wartosc").replace('"', '').replace(',', '').replace("'", "")
            #nazwa_firmy = org.get("nazwa_krotka", {}).get("_wartosc")
            sheet.append([krs, short_name, email_wartosc.lower()])
            # logging.info(f"2 plik +++++++++++ firma: {short_name} email: {email_wartosc}")
 
    workbook.save(output_file_name)
    logging.info(f"Office emails saved in {output_file_name}")

    # Przetwarzanie danych w zapisanym pliku Excel
    df = pd.read_excel(output_file_name)
    before_duplicates = len(df)
    df.drop_duplicates(subset=["email"], inplace=True)  # Usunięcie duplikatów na podstawie adresu email
    after_duplicates = len(df)
    duplicates_removed = before_duplicates - after_duplicates
    
    logging.info(f"Usunięto {duplicates_removed} duplikatów.")
   
   
    # #
    # # Wypisywanie zduplikowanych i niepoprawnych adresów email
    # invalid_emails = [email.value for email in sheet["C"] if not re.match(r'^[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email.value)]

    # if invalid_emails:
    #     logging.info(f"Niepoprawne adresy email: {', '.join(invalid_emails)}")
    # #
    # #


    # Usunięcie wierszy z niepoprawnymi adresami email
    df = df[df['email'].str.match(r'^[a-zA-Z0-9._+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')]
    invalid_emails_removed = after_duplicates - len(df)

    df.to_excel(output_file_name, index=False)  # Nadpisanie pliku z usuniętymi duplikatami i usunietymi niepoprawnymi adreasami

    logging.info(f"Zapisano {len(df)} unikatowych adresów email spelniajacych wymogi")

    results_str.set(f"Office emails saved in {output_file_name} (without duplicates and invalid emails)")
    logging.info(f"Dane zapisane w pliku {output_file_name} (bez duplikatów i niepoprawnych emaili)")

    logging.info(f"Usunięto {invalid_emails_removed} niepoprawnych adresów email.")




def office_emails_csv():
    
    pkd_code = pkd_combobox.get()
    pkd_type = 'przewazajacy' if pkd_type_var.get() == 'przewazajacy_pkd' else 'dowolny'
    capital_range = f"{lower_capital_entry.get()}to{upper_capital_entry.get()}"

    input_file_name_1 = f"orgs_{pkd_code}_{pkd_type}_{capital_range}.json"
    input_file_name_2 = f"orgs_{pkd_code}_{pkd_type}_{capital_range}_advanced_data.json"

    output_file_name = f"orgs_{pkd_code}_{pkd_type}_{capital_range}_office_emails.csv"

    # Otwórz pliki JSON
    with open(input_file_name_1, "r") as orgs_json_1:
        orgs_data_1 = json.load(orgs_json_1)

    with open(input_file_name_2, "r") as orgs_json_2:
        orgs_data_2 = json.load(orgs_json_2)

    # Inicjalizuj dane do zapisu w pliku CSV
    with open(output_file_name, mode='w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["krs", "nazwy.skrocona", "email"])

        # Pobierz emaile i nazwy firm z orgs_data_1
        for org in orgs_data_1:
            krs = org.get("numery", {}).get("krs")
            short_name = org.get("nazwy", {}).get("skrocona").replace('"', '').replace(',', '').replace("'", "")
            contact = org.get("kontakt", {})
            emaile = contact.get("emaile", [])
            for email in emaile:
                csvwriter.writerow([krs, short_name, email.lower()])
                # logging.info(f"1 plik -- firma: {short_name} email: {email}")

        # Pobierz emaile i nazwy firm z orgs_data_2
        for org in orgs_data_2:
            email = org.get("email", {})
            email_wartosc = email.get("_wartosc")
            if email:
                krs = org.get("krs", {}).get("_wartosc")
                short_name = org.get("nazwa_krotka", {}).get("_wartosc").replace('"', '').replace(',', '').replace("'", "")
                csvwriter.writerow([krs, short_name, email_wartosc.lower()])
                # logging.info(f"2 plik +++++++++++ firma: {short_name} email: {email_wartosc}")

    logging.info(f"Office emails saved in {output_file_name}")

    # Przetwarzanie danych w zapisanym pliku CSV
    df = pd.read_csv(output_file_name)
    before_duplicates = len(df)
    df.drop_duplicates(subset=["email"], inplace=True)  # Usunięcie duplikatów na podstawie adresu email
    after_duplicates = len(df)
    duplicates_removed = before_duplicates - after_duplicates
    
    logging.info(f"Usunięto {duplicates_removed} duplikatów.")
    
    # Usunięcie wierszy z niepoprawnymi adresami email
    df = df[df['email'].str.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')]
    invalid_emails_removed = after_duplicates - len(df)

    df.to_csv(output_file_name, index=False)  # Nadpisanie pliku z usuniętymi duplikatami i niepoprawnymi adresami email

    logging.info(f"Zapisano {len(df)} unikatowych adresów email spełniających wymogi")

    results_str.set(f"Office emails saved in {output_file_name} (without duplicates and invalid emails)")
    logging.info(f"Dane zapisane w pliku {output_file_name} (bez duplikatów i niepoprawnych emaili)")

    logging.info(f"Usunięto {invalid_emails_removed} niepoprawnych adresów email.")


def update_buttons():
    for button, active in button_states.items():
        button['state'] = tk.NORMAL if active else tk.DISABLED
    root.after(100, update_buttons)

# GUI setup
# root = tk.Tk()
# root.title("Rejestr.io Organization Search Tool")
button_states = {}
labelframe1 = tk.LabelFrame(root, text="Wyszukiwanie organizacji", borderwidth=2, relief="groove")
labelframe1.pack(fill="both", expand=True, padx=10, pady=10)

labelframe2 = tk.LabelFrame(root, text="Przygotowanie pliku JSON - prepared", borderwidth=2, relief="groove")
labelframe2.pack(fill="both", expand=True, padx=10, pady=10)

labelframe3 = tk.LabelFrame(root, text="Zaawansowane dane organizacji - advanced", borderwidth=2, relief="groove")
labelframe3.pack(fill="both", expand=True, padx=10, pady=10)

labelframe4 = tk.LabelFrame(root, text="EXCEL dla Hunter.io", borderwidth=2, relief="groove")
labelframe4.pack(fill="both", expand=True, padx=10, pady=10)

labelframe5 = tk.LabelFrame(root, text="Office Emails EXCEL", borderwidth=2, relief="groove")
labelframe5.pack(fill="both", expand=True, padx=10, pady=10)

labelframe_Treminal = tk.LabelFrame(root, text="Treminal", borderwidth=2, relief="groove")
labelframe_Treminal.pack(fill="both", expand=True, padx=10, pady=10)

# PKD type and capital entry setup
tk.Label(labelframe1, text="Select PKD Type:").grid(row=0, column=0)
pkd_type_var = tk.StringVar(value='przewazajacy_pkd')
tk.Radiobutton(labelframe1, text="Przeważający PKD", variable=pkd_type_var, value='przewazajacy_pkd').grid(row=0, column=1)
tk.Radiobutton(labelframe1, text="Dowolny PKD", variable=pkd_type_var, value='dowolny_pkd').grid(row=0, column=2)
pkd_combobox = ttk.Combobox(labelframe1, values=['24', '24.51.Z', '24.52.Z', '26', '26.30.Z', '27.12.Z', '29', '29.10.Z', '82.30.Z'])
pkd_combobox.grid(row=1, column=1)
pkd_combobox.set('27.12.Z')

tk.Label(labelframe1, text="Minimum Capital (gte):").grid(row=2, column=0)
lower_capital_entry = tk.Entry(labelframe1)
lower_capital_entry.grid(row=2, column=1)
lower_capital_entry.insert(0, '5000')

tk.Label(labelframe1, text="Maximum Capital (lt):").grid(row=3, column=0)
upper_capital_entry = tk.Entry(labelframe1)
upper_capital_entry.grid(row=3, column=1)
upper_capital_entry.insert(0, '1005000')

# Submit and save buttons
search_button = tk.Button(labelframe1, text="Search", command=submit_search)
search_button.grid(row=4, column=1)
save_button = tk.Button(labelframe1, text="Get and Save JSON", command=get_data_and_save)
button_states[save_button] = False  # Initially disabled
save_button.grid(row=4, column=2)

# Advanced companys data panel buttons
message_button = tk.Button(labelframe2, text="Prepare JSON for GET & Excel for scraping", command=prepare_json)
message_button.grid(row=0, column=0)

get_button = tk.Button(labelframe3, text="GET", command=get_advanced_data)
# get_button['state'] = tk.DISABLED
button_states[get_button] = False 
get_button.grid(row=0, column=1)

get_people_names_button = tk.Button(labelframe4, text="Get people names", command=get_people_names)
get_people_names_button.grid(row=0, column=1)
#button_states[get_people_names_button] = False 

button = tk.Button(labelframe5, text="Office Emails EXCEL", command=office_emails_excel)
button.grid(row=0, column=1)

button = tk.Button(labelframe5, text="Office Emails CSV", command=office_emails_csv)
button.grid(row=0, column=2)

# get_button = tk.Button(labelframe2, text="Advanced data to excel", command=advanced_data_to_excel)
# get_button['state'] = tk.DISABLED
# get_button.grid(row=0, column=2)

# Result display
results_str = tk.StringVar()
tk.Label(labelframe_Treminal, textvariable=results_str).grid(row=5, column=0, columnspan=3)


update_buttons()

root.mainloop()


