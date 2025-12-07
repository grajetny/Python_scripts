import os

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog

import pandas as pd
import json
import random
from bs4 import BeautifulSoup
import requests
import re
import time

class ScraperApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Web Scraper Interface')
        self.geometry('800x500')  # Initial size

        self.configure_grid()
        self.style_application()
        self.create_widgets()
        self.minsize(800, 500)

        self.data = None
        self.file_path = None

        # Iterowanie przez rekordy danych
        self.successful_count = 0
        self.failed_count = 0

    def configure_grid(self):
        # Configure the grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def style_application(self):
        # Set general style for the application
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use a theme that supports background coloring

        # Font configuration
        self.default_font = ('Helvetica', 12)
        self.button_font = ('Helvetica', 12, 'bold')

        # Background and foreground configuration
        self.style.configure('.', background='white', foreground='black', font=self.default_font)
        self.style.configure('TButton', font=self.button_font, padding=5)
        self.style.configure('GoogleBlue.TButton', background='#4285F4', foreground='white')
        self.style.configure('GoogleRed.TButton', background='#DB4437', foreground='white')
        self.style.configure('GoogleYellow.TButton', background='#F4B400', foreground='black')
        self.style.configure('GoogleGreen.TButton', background='#0F9D58', foreground='white')
        self.style.configure('Danger.TButton', foreground='red', background='light gray')

    def create_widgets(self):
        # Left frame for buttons
        self.left_frame = ttk.Frame(self)
        self.left_frame.grid(row=0, column=0, sticky='ns', padx=10, pady=10)
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(7, weight=1)

        # Right frame for text displays
        self.right_frame = ttk.Frame(self)
        self.right_frame.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        self.right_frame.grid_columnconfigure(0, weight=1)
        for i in range(4):
            self.right_frame.grid_rowconfigure(i, weight=1)

        # Buttons with dynamic resizing and colored styles
        self.import_button = ttk.Button(self.left_frame, text="1. Import", style='GoogleBlue.TButton', command=lambda: perform_import(self))
        self.import_button.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        self.prepare_file_button = ttk.Button(self.left_frame, text="2. Prepare File", style='GoogleRed.TButton', command=lambda: prepare_file(self))
        self.prepare_file_button.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        self.scraping_button = ttk.Button(self.left_frame, text="3. Scraping", style='GoogleYellow.TButton', command=lambda: start_scraping(self))
        self.scraping_button.grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        self.action_button = ttk.Button(self.left_frame, text="Convert JSON to CSV", style='GoogleGreen.TButton', command=lambda: json_to_csv(self))
        self.action_button.grid(row=4, column=0, sticky='ew', padx=5, pady=5)
        self.clear_program_button = ttk.Button(self.left_frame, text="Clear Program", style='Danger.TButton', command=lambda: clear_program(self))
        self.clear_program_button.grid(row=9, column=0, sticky='ew', padx=5, pady=5)

        # Labels for display
        self.text_area_import = tk.Label(self.right_frame, text="Ready to import data.", font=self.default_font)
        self.text_area_import.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.text_area_prepare = tk.Label(self.right_frame, text="Prepare your files here.", font=self.default_font)
        self.text_area_prepare.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        self.text_area_scraping = tk.Label(self.right_frame, text="Scraping status will be shown here.", font=self.default_font)
        self.text_area_scraping.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)
        self.text_area_clear = tk.Label(self.right_frame, text="Logs and actions will be cleared.", font=self.default_font)
        self.text_area_clear.grid(row=3, column=0, sticky='nsew', padx=5, pady=5)

# Define functions outside of the class
# def perform_import(app):
#     app.text_area_import.config(text="Importing data...")
#     messagebox.showinfo("Import", "Data import completed successfully!")

###


def perform_import(app):
    app.file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("JSON files", "*.json"), ("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
    )
    if not app.file_path:
        app.text_area_import.config(text="No file selected.")
        return

    try:
        required_fields = {'nazwy.skrocona', 'nazwy.pelna', 'cleared_www', 'cleared_email_domain'}
        file_name = os.path.basename(app.file_path)  # Extract the filename from the full file path

        if app.file_path.endswith('.json'):
            with open(app.file_path, 'r') as file:
                app.data = json.load(file)
            # Check each item for required fields
            if all(required_fields <= set(item.keys()) for item in app.data):
                app.text_area_import.config(text=f"File loaded successfully. Found all required fields. JSON type file. \n\n '{file_name}'")
                display_first_record_JSON(app, app.data[0])
            else:
                missing = [field for field in required_fields if all(field not in item for item in app.data)]
                app.text_area_import.config(text=f"Missing fields: {', '.join(missing)}")
                messagebox.showerror("Import Error", f"Missing fields in JSON data: {', '.join(missing)}")

        elif app.file_path.endswith('.xlsx'):  # TODO:
            df = pd.read_excel(app.file_path)
            check_fields(df, app, required_fields, file_name, 'XLSX')

            # display_first_record_xlsx_csv() # TODO: TU MASZ DOKONZCYC MG FIXME:

        elif app.file_path.endswith('.csv'):  # TODO:
            df = pd.read_csv(app.file_path)
            check_fields(df, app, required_fields, file_name, 'CSV')

    except Exception as e:
        app.text_area_import.config(text="Failed to load the file.")
        messagebox.showerror("Import Error", f"An error occurred: {str(e)}")


# def prepare_file(app):
#     app.text_area_prepare.config(text="Preparing file...")
#     messagebox.showinfo("Prepare File", "File prepared successfully!")

def prepare_file(app):
    if not hasattr(app, 'data'):
        app.text_area_prepare.config(text="No JSON data loaded to prepare. Please import data first.")
        return
    
    # Przetwarzanie danych JSON
    #data = app.data  # Zakładamy, że dane JSON są już załadowane jako lista słowników
    for record in app.data:
        record['url_scraped_nazwy.skrocona'] = ''
        record['url_scraped_nazwy.pelna'] = ''
        record['final_url'] = record['cleared_www'] if record['cleared_www'] else record['cleared_email_domain']
        record['scraping_flag'] = 1 if record['final_url'] else 0

    #app.data = data  # Aktualizacja danych

    if app.file_path.endswith('.json'):
        with open(app.file_path, 'w') as file:
            json.dump(app.data, file, indent=4)
        messagebox.showinfo("Save Data", "Data saved as JSON successfully.")
    elif app.file_path.endswith('.xlsx'):
        df = pd.DataFrame(app.data)
        df.to_excel(app.file_path, index=False)
        messagebox.showinfo("Save Data", "Data saved as Excel successfully.")
    elif app.file_path.endswith('.csv'):
        df = pd.DataFrame(app.data)
        df.to_csv(app.file_path, index=False)
        messagebox.showinfo("Save Data", "Data saved as CSV successfully.")
    else:
        messagebox.showerror("Save Error", "Unsupported file format for saving.")

    app.text_area_prepare.config(text="JSON data prepared successfully with new fields.")
    messagebox.showinfo("Prepare File", "JSON data has been prepared with new fields successfully.")

# def start_scraping(app):
#     app.text_area_scraping.config(text="Scraping started...")
#     messagebox.showinfo("Scraping", "Scraping completed successfully!")

def start_scraping(app):
    if not hasattr(app, 'data') or not app.data:
        app.text_area_scraping.config(text="No data loaded. Please load and prepare data first.")
        messagebox.showerror("Scraping Error", "No data loaded. Please import and prepare data first.")
        return

    # Sprawdzenie, czy istnieją wymagane pola
    required_fields = {'url_scraped_nazwy.skrocona', 'url_scraped_nazwy.pelna', 'final_url', 'scraping_flag'}
    first_record = app.data[0] if isinstance(app.data, list) else app.data.iloc[0]

    # Dla JSON:
    if isinstance(app.data, list):
        missing_fields = [field for field in required_fields if field not in first_record]
    # Dla DataFrame (Excel/CSV):
    elif isinstance(app.data, pd.DataFrame):
        missing_fields = [field for field in required_fields if field not in app.data.columns]
    else:
        app.text_area_scraping.config(text="Unsupported data format.")
        messagebox.showerror("Scraping Error", "Unsupported data format.")
        return

    if missing_fields:
        error_message = "Missing required fields for scraping: " + ", ".join(missing_fields) + ". Run 'Prepare File' first."
        app.text_area_scraping.config(text=error_message)
        messagebox.showerror("Scraping Error", error_message)
        return

    # Logika scrapowania
    app.text_area_scraping.config(text="Starting scraping process...")
    messagebox.showinfo("Scraping Info", "Scraping process started successfully. Processing data...")
    


    # for record in app.data:
    #     if record['scraping_flag'] == 0:
    #         if company_scrape(record):
    #             final_url_choice(record)
    #             successful_count += 1
    #         else:
    #             failed_count += 1
    #     messagebox.showinfo("Scraping ... ", f"Success: {successful_count}, Failed: {failed_count}")


    try:
        for index, record in enumerate(app.data):
            if record['scraping_flag'] == 0:
                time.sleep(random.randint(7,12))  # Delay to avoid 429 Too Many Requests error
                company_scrape(record)
                final_url_choice(record)

            app.text_area_scraping.config(text=f"Scraping ... | Success: {app.successful_count} | Failed: {app.failed_count} |")
            print(f"Scraping ... | Success: {app.successful_count} | Failed: {app.failed_count} |")

            # Periodically save data every 10 records or at the end of data
            if index % 10 == 0 or index == len(app.data) - 1:
                save_data(app)

        app.text_area_scraping.config(text=f"Scraping completed. Success: {app.successful_count}, Failed: {app.failed_count}")
        #messagebox.showinfo("Scraping Completed", f"Scraping process completed successfully. Success: {successful_count}, Failed: {failed_count}")

    except Exception as e:
        app.text_area_scraping.config(text="Scraping interrupted. Saving progress...")
        save_data(app)
        messagebox.showerror("Scraping Interrupted", f"Scraping was interrupted. Error: {str(e)}")




def save_data(app):
    if app.file_path.endswith('.json'):
        with open(app.file_path, 'w') as file:
            json.dump(app.data, file, indent=4)
    elif app.file_path.endswith('.xlsx'):
        pd.DataFrame(app.data).to_excel(app.file_path, index=False)
    elif app.file_path.endswith('.csv'):
        pd.DataFrame(app.data).to_csv(app.file_path, index=False)
    print("Data has been saved successfully.")
    # messagebox.showinfo("Data Saved", "Data has been saved successfully.")

#############################
#----------------------------
def company_scrape(record):
    

    # List of URL patterns to exclude using regular expressions
    exclusion_list = [
        r'https://rejestr.io/krs/.*' , # Regex pattern to match any URL starting with 'https://rejestr.io/krs/'
        r'http://rejestr.io/krs/.*' ,
        r'http://rejestrkrs.pl/.*'
        r'https://aleo.com/.*' ,
        r'https://krs-pobierz.pl/.*' ,
        r'https://krs-firma.pl/.*' ,
        r'https://www.bizraport.pl/.*' ,
        r'https://sprawozdaniaonline.pl/.*' ,
        r'https://www.emis.com/.*' ,
        r'https://pl.wikipedia.org/.*' ,
        r'https://en.wikipedia.org/.*' ,
        r'https://www.gowork.pl/.*' ,
        r'https://sjp.pwn.pl/.*' ,
        r'https://www.wyszukiwarkakrs.pl/.*' ,
        r'https://www.filmweb.pl/.*' ,
        r'https://www.imsig.pl/.*' ,
        r'https://aleo.com*' ,
        r'https://panoramafirm.pl.*' ,
        r'https://www.owg.pl.*' ,
        r'https://www.oferteo.pl/.*'
        r'https://www.cabb.pl/.*'
        r'https://www.sa.kompass.com/.*'
        r'https://www.pracodawcy.pracuj.pl/.*'
        r'https://www.cabb.pl/.*'
        r'https://cabb.pl/.*'
        r'https://www.polskiedane.io/.*'
        # Add more regex patterns as needed
    ]




    record['url_scraped_nazwy.skrocona'] = search_company_website(record['nazwy.skrocona'] ,exclusion_list)
    record['url_scraped_nazwy.pelna'] = search_company_website(record['nazwy.pelna'] ,exclusion_list)




def search_company_website(company_name, exclusion_list):

    search_url = f"https://www.google.com/search?q={company_name}&gl=PL"

    #In this modification, &gl=PL is added to the search URL to specify that the search results should be more relevant to Poland.
    user_agent_list = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36' ,
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36' ,
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36' ,
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36' ,
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36' ,
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15' ,
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.1 Safari/605.1.15' ,
    ]
    user_agent = random.choice(user_agent_list)
    headers = {'User-Agent': user_agent}

    try:
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.select('.tF2Cxc .yuRUbf a')  # Updated CSS selector
            for link in links:
                href = link.get('href')
                if not should_exclude_url(href, exclusion_list): #and company_name.lower() in href.lower() - to jest dodatkowy warunek ktory sie nie sprawdzil
                    app.successful_count += 1
                    return href  # Returns the first found link containing the company name and not in the exclusion list
        else:
            print(f"Error: Company name: {company_name}  Status code {response.status_code}")
            app.failed_count += 1
            return "" #zmienione ze starej komendy return None
    except Exception as e:
        print(f"Scraping failed for {company_name}: {str(e)}")
        return False

# Function to check if a URL should be excluded based on the exclusion list
def should_exclude_url(url, exclusion_list):
    for pattern in exclusion_list:
        if re.match(pattern, url):
            return True
    return False

# def final_url_choice(record):
#     # Logika wyboru final_url, może być oparta na różnych kryteriach
#     # if record['url_scraped_nazwy.skrocona'] and record['url_scraped_nazwy.pelna']:
#     #     record['final_url'] = record['url_scraped_nazwy.pelna']  # Wybór wartości przykładowej
#     #     record['scraping_flag'] = 1  # Ustawienie flagi na 1 po udanym scrapowaniu

#     record['final_url'] = record['url_scraped_nazwy.pelna']  # Wybór wartości przykładowej
#     record['scraping_flag'] = 1  # Ustawienie flagi na 1 po udanym scrapowaniu

def final_url_choice(record):
    # Define a simple scoring system to determine the best URL
    def score_url(url, name):
        score = 0
        if url:
            if name.lower().replace(' ', '') in url.lower().replace('www.', '').replace('.com', ''):
                score += 10  # Boost score if the company name is part of the URL
            if 'https://' in url:
                score += 2  # Prefer secure URLs
        return score

    # Calculate scores for both scraped URLs
    short_name_score = score_url(record['url_scraped_nazwy.skrocona'], record['nazwy.skrocona'])
    full_name_score = score_url(record['url_scraped_nazwy.pelna'], record['nazwy.skrocona'])

    # Choose the URL with the highest score
    if short_name_score > full_name_score:
        chosen_url = record['url_scraped_nazwy.skrocona']
    elif full_name_score > short_name_score:
        chosen_url = record['url_scraped_nazwy.pelna']
    else:
        # If scores are the same, prefer the full name URL, or default to any if both are None
        chosen_url = record['url_scraped_nazwy.pelna'] if record['url_scraped_nazwy.pelna'] else record['url_scraped_nazwy.skrocona']

    # Clean the chosen URL to store a simplified version if not None
    if chosen_url:
        simplified_url = re.sub(r"https?://(www\.)?", "", chosen_url).split('/')[0]  # Simplify URL to basic domain
    else:
        simplified_url = None

    record['final_url'] = simplified_url
    print(f"Company: {record['nazwy.skrocona']} Final url: {simplified_url} ")
    record['scraping_flag'] = 1 if simplified_url else 0



#------------------------------
###############################


def json_to_csv(app):
# Ask the user to select a JSON file
    json_file_path = filedialog.askopenfilename(
        title="Select a JSON file",
        filetypes=[("JSON files", "*.json")]
    )
    if not json_file_path:
        messagebox.showinfo("Information", "No file selected.")
        return

    # Load the JSON data
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load the file: {str(e)}")
        return

    # Check if all required fields are present
    required_fields = {'nazwy.skrocona', 'nazwy.pelna', 'krs', 'www', 'cleared_www', 'url_scraped_nazwy.skrocona', 'url_scraped_nazwy.pelna', 'final_url', 'scraping_flag', 'people_names'}
    missing_fields = required_fields - set(data[0].keys())
    if missing_fields:
        messagebox.showerror("Error", f"Missing fields in the JSON file: {', '.join(missing_fields)}")
        return

    # Create CSV data
    csv_data = []
    for company in data:
        for person in company.get('people_names', []):
            row = {
                'Person Name': person,
                'Short Name': company['nazwy.skrocona'],
                'Full Name': company['nazwy.pelna'],
                'KRS': company['krs'],
                'Website': company['www'],
                'Cleared WWW': company['cleared_www'],
                'URL Scraped Short': company['url_scraped_nazwy.skrocona'],
                'URL Scraped Full': company['url_scraped_nazwy.pelna'],
                'Final URL': company['final_url'],
                'Scraping Flag': company['scraping_flag']
            }
            csv_data.append(row)

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(csv_data)

    # Ask for location to save the CSV file
    csv_file_path = filedialog.asksaveasfilename(
        title="Save as",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if not csv_file_path:
        return

    # Save DataFrame to CSV
    df.to_csv(csv_file_path, index=False)
    messagebox.showinfo("Success", f"Data exported successfully to {csv_file_path}")


def clear_program(app):
    app.text_area_import.config(text="Ready to import data.")
    app.text_area_prepare.config(text="Prepare your files here.")
    app.text_area_scraping.config(text="Scraping status will be shown here.")
    app.text_area_clear.config(text="Logs and actions will be cleared.")
    messagebox.showinfo("Clear", "All data cleared!")


###################################################################################################
###------------------------------------------IMPORT---------------------------------------------###

def display_preview(app, data, file_name, data_format='JSON'):  # TODO: usunac te funkcje
    # Configure the success message and display first few company names
    success_message = f"File '{file_name}' loaded successfully. Found all required fields. {data_format} type file."
    preview_data = '\n'.join([item['nazwy.skrocona'] for item in data[:3]])  # Preview first 3 records
    app.text_area_import.config(text=success_message)
    app.text_area_clear.config(text=f"Preview of '{file_name}':\n{preview_data}")
    messagebox.showinfo("Import", f"{data_format} Data import completed successfully and all required fields are present in {file_name}.")

def display_first_record_xlsx_csv():  # TODO:
    preview_data = '\n'.join(df['nazwy.skrocona'].head(3).tolist())  # Preview first 3 records
    app.text_area_clear.config(text=f"Preview of '{file_name}':\n{preview_data}")

def display_first_record_JSON(app, record):
    # Pobieranie do czterech pierwszych emaili z listy
    email_list = record['emaile'][:4]  # Zwraca pierwsze cztery emaile lub mniej, jeśli ich liczba jest mniejsza
    formatted_emails = ', '.join(f'"{email}"' for email in email_list)  # Formatuje emaile jako ciąg znaków oddzielonych przecinkami

    # Przygotowanie formatowanego tekstu
    formatted_text = (
        f"\"nazwy.skrocona\": \"{record['nazwy.skrocona']}\",\n"
        f"\"nazwy.pelna\": \"{record['nazwy.pelna']}\",\n"
        f"\"krs\": {record['krs']},\n"
        f"\"emaile\": [{formatted_emails}],\n"  # Użycie sformatowanej listy emaili
        f"\"www\": \"{record['www']}\",\n"
        f"\"cleared_www\": \"{record['cleared_www']}\",\n"
        f"\"cleared_email_domain\": \"{record['cleared_email_domain']}\""
    )
    
    # Aktualizacja treści etykiety w interfejsie użytkownika
    app.text_area_clear.config(text=formatted_text)


def check_fields(df, app, required_fields, file_name, file_type): # TODO: poprawic wyswietlanie dla xlsx oraz csv
    data_keys = set(df.columns)
    if required_fields <= data_keys:
        # preview_data = '\n'.join(df['nazwy.skrocona'].head(3).tolist())  # Preview first 3 records
        app.text_area_import.config(text=f"File loaded successfully. Found all required fields. XLSX/CSV type file. \n\n {file_type} ")
        # app.text_area_clear.config(text=f"Preview of '{file_name}':\n{preview_data}")
        messagebox.showinfo("Import", f"Data import completed successfully and all required fields are present in {file_name}.")
    else:
        missing = required_fields - data_keys
        app.text_area_import.config(text=f"Missing fields: {', '.join(missing)} in {file_name}")
        messagebox.showerror("Import Error", f"Missing fields in {file_name}: {', '.join(missing)}")

def missing_fields(app, data, required_fields):
    missing = [field for field in required_fields if not any(field in item for item in data)]
    app.text_area_import.config(text=f"Missing fields: {', '.join(missing)}")
    messagebox.showerror("Import Error", f"Missing fields in data: {', '.join(missing)}")

###------------------------------------------IMPORT---------------------------------------------###
###################################################################################################

if __name__ == "__main__":
    app = ScraperApp()
    app.mainloop()


