from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

def scrollToElement(driver, element):
    y = element.location['y']
    window_height = driver.execute_script('return window.innerHeight')
    new_y = y - window_height / 2
    driver.execute_script(f'window.scrollTo(0, {new_y})')
    print(f"Scrolled to element at y={y}")

# Wyświetlenie komunikatu powitalnego i opisu działania programu
print("Witaj w programie do scrapowania stron WWW z listami wystawcow Warsaw PTAK EXPO!")
print("Program ten pobiera informacje o wystawcach z podanej strony targowej - nazwa firmy i adres url.")
url = input("Proszę podać adres URL strony do scrapowania: ")  # Pobranie URL od użytkownika

# Inicjalizacja WebDrivera Chrome
driver = webdriver.Chrome()
print("WebDriver initialized.")

# Otworzenie podanego URL w przeglądarce
driver.get(url)
print(f"Page loaded: {url}")

try:
    accept_cookies_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "cookie_action_close_header"))
    )
    accept_cookies_button.click()
    print("Cookie acceptance button clicked.")
except Exception as e:
    print("Cookie button not found or not clickable. Proceeding without clicking it.", e)

try:
    exhibitors_container = WebDriverWait(driver, 20).until(
        EC.presence_of_element_located((By.CLASS_NAME, "exhibitors__container"))
    )
    print("Exhibitors container found.")

    companies = []
    websites = []

    exhibitors = exhibitors_container.find_elements(By.CLASS_NAME, 'exhibitors__container-list')
    print(f"Found {len(exhibitors)} exhibitors.")
    for exhibitor in exhibitors:
        scrollToElement(driver, exhibitor)
        
        time.sleep(0.4)
        exhibitor.click()
        print("Exhibitor clicked.")

        modal_elements = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "modal__elements"))
        )
        print("Modal elements found.")

        h3_elements = modal_elements.find_elements(By.TAG_NAME, 'h3')
        a_elements = modal_elements.find_elements(By.TAG_NAME, 'a')

        for h3 in h3_elements:
            companies.append(h3.text)
            print(f"Company found: {h3.text}")

        for a in a_elements:
            websites.append(a.get_attribute('href'))
            print(f"Website found: {a.get_attribute('href')}")

        close_button = modal_elements.find_element(By.CLASS_NAME, "close")
        close_button.click()
        print("Modal window closed.")

    print("Data collection completed.")
    
    print(f"Number of companies found: {len(companies)}")
    print(f"Number of websites found: {len(websites)}")

    if len(companies) != len(websites):
        print("Warning: Mismatch in number of companies and websites extracted.")
        max_length = max(len(companies), len(websites))
        companies.extend([None] * (max_length - len(companies)))
        websites.extend([None] * (max_length - len(websites)))

    df = pd.DataFrame({
        'Company': companies,
        'Website': websites
    })
    df.to_excel('HVAC_EXPO_companies_and_websites.xlsx', index=False)
    print("Data saved to Excel. Task Completed Successfully.")

finally:
    driver.quit()
    print("WebDriver closed.")
