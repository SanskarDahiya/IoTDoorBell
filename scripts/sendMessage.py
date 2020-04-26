import requests
from credentials import ifttt

print(ifttt)
url = ifttt["url"]
values = ifttt['values']


def sendSMS(msg=False, phone=False):
    values['value2'] = msg or "Np Message given"
    if(phone):
        values['value1'] = phone
    x = requests.post(url, values)
    print(values, x.text)
