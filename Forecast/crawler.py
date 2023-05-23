import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from datetime import date, timedelta

options = Options()
prefs = {"download.default_directory" : "./home/cellery/Desktop/ML_for_Beginning/Forecast/csvs"}
options.add_experimental_option("prefs",prefs)
options.add_argument("--disable-notifications")
craw_chrome = webdriver.Chrome('./chromedriver_linux64/chromedriver', chrome_options=options)


start_date = date(2016, 1, 1)   # start date
end_date = date(2023, 5, 1)     # end date


delta = end_date - start_date
for i in range(delta.days + 1):
    this_date = start_date + timedelta(days=i)
    url = "https://e-service.cwb.gov.tw/HistoryDataQuery/DayDataController.do?command=viewMain&station=467410&stname=%25E8%2587%25BA%25E5%258D%2597&datepicker="+str(this_date)+"&altitude=40.8m#"
    r = craw_chrome.get(url)
    try:
        downloadCSV = craw_chrome.find_element(By.ID, "downloadCSV")
        nexItem = craw_chrome.find_elements(By.TAG_NAME, "input")
        print("now crawling: ", this_date)
        downloadCSV.click()
        time.sleep(0.5)
        nexItem[1].click
        time.sleep(0.5)
    except Exception as e:
        print(e)
        print("Failed at ", this_date)
        break

print("Done!, {} files were downloaded.".format(delta.days + 1))